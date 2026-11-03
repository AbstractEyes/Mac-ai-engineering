"""A timeline of marks on the Apple GPU's command stream, for timing a model's training step by component (series D3-D5,
step 0; docs/series_layer.md).

A mark records a Metal timing event (the point the GPU reached in the stream, as PyTorch's MPS event reports it) and the
CPU's clock (time.perf_counter_ns) when the mark was issued. The time between two consecutive marks belongs to the innermost
component open at that point (exclusive time: a parent's time is what its timed children leave).

The backward is marked by identity autograd nodes placed at the forward's marks (on the tensors that need a gradient). The
autograd engine runs one device's nodes in the reverse of their creation order, so these nodes run in the reverse order of
the forward's marks, interleaved with the backward of the operations between them: the backward segment that starts at the
node of forward mark j belongs to the component that was innermost just before mark j in the forward.

    tl = Timeline()
    inst = Instrument(tl)
    inst.module(model.layers[0].mlp, "mlp")          # forward pre / post hooks, and the backward nodes
    inst.function(module_namespace, "fn", "fn")      # a module-level function wrapped the same way
    tl.start(); loss = model(x).sum(); loss.backward(); tl.stop()
    rows = tl.attribute()                            # {(component, "fwd" | "bwd"): [gpu ms, cpu ms, segments]}
    inst.remove(); tl.release()

Every mark costs a Metal event record: PyTorch ends the current command encoder and commits the command buffer at each one,
so a mark perturbs the stream it measures. bench/timeline_check.py measures that cost and the instrument's accuracy.

The fence: before each event the timeline issues one tiny kernel (an in-place add on a 1-element tensor). A timing event whose
command buffer holds no GPU work (two marks back to back, or a mark right after an operation that committed its own command
buffer) is reported late by up to ~0.1 ms while the GPU runs the next buffer, so short segments came out negative and time
moved between neighbours. With the fence every event follows work in its own command buffer, and segments match the operations'
own times (bench/layer_logs/timeline_check.json).
"""
import time

import torch

_C = torch._C
_HAS_RAW = all(hasattr(_C, n) for n in ("_mps_acquireEvent", "_mps_recordEvent", "_mps_elapsedTimeOfEvents",
                                         "_mps_releaseEvent"))


def _acquire() -> int:
    return _C._mps_acquireEvent(True)


def _record(eid: int) -> None:
    _C._mps_recordEvent(eid)


def _elapsed(a: int, b: int) -> float:
    return _C._mps_elapsedTimeOfEvents(a, b)


def _release(eid: int) -> None:
    _C._mps_releaseEvent(eid)


class _Mark(torch.autograd.Function):
    """Identity in the forward; in the backward, a mark on the timeline (keyed by the forward mark that created it)."""

    @staticmethod
    def forward(ctx, tl, j, *xs):
        ctx.tl, ctx.j = tl, j
        ctx.set_materialize_grads(False)
        return xs[0] if len(xs) == 1 else xs

    @staticmethod
    def backward(ctx, *grads):
        ctx.tl.mark_backward(ctx.j)
        return (None, None) + grads


class Timeline:
    """Marks: P (program marks on the issuing thread: a component entered or left, or a point) and B (backward nodes, each
    carrying the index of the program mark that created it)."""

    def __init__(self, fence: bool = True):
        if not _HAS_RAW:
            raise RuntimeError("this PyTorch has no MPS timing events")
        self.on = False
        self.fence = torch.zeros(1, device="mps") if fence else None
        self.clear()

    def clear(self):
        self.eid, self.cpu, self.kind, self.comp, self.ref = [], [], [], [], []

    def start(self):
        self.on = True

    def stop(self):
        self.on = False

    def _add(self, kind, comp, ref) -> int:
        eid = _acquire()
        if self.fence is not None:
            self.fence.add_(1.0)        # needs no gradient: no autograd record
        _record(eid)
        self.cpu.append(time.perf_counter_ns())
        self.eid.append(eid)
        self.kind.append(kind)
        self.comp.append(comp)
        self.ref.append(ref)
        return len(self.eid) - 1

    def enter(self, comp) -> int:
        return self._add("enter", comp, -1) if self.on else -1

    def exit(self, comp) -> int:
        return self._add("exit", comp, -1) if self.on else -1

    def mark_backward(self, j: int) -> None:
        if self.on:
            self._add("B", None, j)

    def resolve(self) -> list:
        """Each mark's time in ms after the first mark. Call after the GPU has finished (torch.mps.synchronize()): each
        elapsed-time query waits for its events."""
        torch.mps.synchronize()
        if not self.eid:
            return []
        base = self.eid[0]
        out = [0.0]
        for e in self.eid[1:]:
            try:
                out.append(_elapsed(base, e))
            except RuntimeError:
                try:
                    out.append(-_elapsed(e, base))
                except RuntimeError:
                    out.append(float("nan"))
        return out

    def release(self):
        for e in self.eid:
            _release(e)
        self.clear()

    def attribute(self, times=None) -> dict:
        """{(component, direction): [gpu ms, cpu ms, segments]} by exclusive time, plus the timeline's checks under the key
        '_checks' (the span, the sum of the segments, segments with a negative GPU time, stack mismatches)."""
        t = self.resolve() if times is None else times
        n = len(t)
        owner_after = {}
        stack = []
        mismatches = 0
        prev_p = {}
        last_p = None
        rows = {}
        neg = 0
        for i in range(n):
            k = self.kind[i]
            if k in ("enter", "exit"):
                prev_p[i] = last_p
                last_p = i
                if k == "enter":
                    stack.append(self.comp[i])
                else:
                    if stack and stack[-1] == self.comp[i]:
                        stack.pop()
                    else:
                        mismatches += 1
                        for q in range(len(stack) - 1, -1, -1):
                            if stack[q] == self.comp[i]:
                                del stack[q]
                                break
                owner_after[i] = (stack[-1] if stack else "(outside)", "fwd")
        for i in range(n - 1):
            k = self.kind[i]
            if k == "B":
                p = prev_p.get(self.ref[i])
                owner = (owner_after[p][0] if p is not None else "(outside)", "bwd")
            else:
                owner = owner_after[i]
                # program marks issued during the backward (around it) keep their own component; a component whose name
                # says backward is reported in the backward direction
                if owner[0].endswith(":bwd"):
                    owner = (owner[0][:-4], "bwd")
            g = t[i + 1] - t[i]
            c = (self.cpu[i + 1] - self.cpu[i]) / 1e6
            if g < 0:
                neg += 1
            r = rows.setdefault(owner, [0.0, 0.0, 0])
            r[0] += g
            r[1] += c
            r[2] += 1
        rows["_checks"] = {"marks": n, "backward_marks": sum(1 for k in self.kind if k == "B"),
                           "span_ms": (t[-1] - t[0]) if n else 0.0,
                           "sum_ms": sum(v[0] for k, v in rows.items() if k != "_checks"),
                           "negative_segments": neg, "stack_mismatches": mismatches, "open_at_end": list(stack)}
        return rows


def _float_grad_tensors(xs):
    return [i for i, x in enumerate(xs) if isinstance(x, torch.Tensor) and x.requires_grad and x.is_floating_point()]


class Instrument:
    """Marks on modules (forward pre / post hooks) and on module-level functions (wrappers), each with its backward nodes."""

    def __init__(self, tl: Timeline, backward: bool = True):
        self.tl = tl
        self.backward = backward
        self.handles = []
        self.patched = []

    # the backward nodes: one node for all the tensors that need a gradient at a mark
    def _wrap_args(self, j, args, kwargs):
        if j < 0 or not self.backward or not torch.is_grad_enabled():
            return args, kwargs
        a_idx = _float_grad_tensors(args)
        k_names = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor) and v.requires_grad and v.is_floating_point()]
        if not a_idx and not k_names:
            return args, kwargs
        xs = [args[i] for i in a_idx] + [kwargs[k] for k in k_names]
        ys = _Mark.apply(self.tl, j, *xs)
        ys = (ys,) if isinstance(ys, torch.Tensor) else ys
        args = list(args)
        for i, y in zip(a_idx, ys[:len(a_idx)]):
            args[i] = y
        kwargs = dict(kwargs)
        for k, y in zip(k_names, ys[len(a_idx):]):
            kwargs[k] = y
        return tuple(args), kwargs

    def _wrap_out(self, j, out):
        if j < 0 or not self.backward or not torch.is_grad_enabled():
            return out
        if isinstance(out, torch.Tensor):
            return _Mark.apply(self.tl, j, out) if out.requires_grad and out.is_floating_point() else out
        if isinstance(out, (tuple, list)):
            idx = _float_grad_tensors(out)
            if not idx:
                return out
            ys = _Mark.apply(self.tl, j, *[out[i] for i in idx])
            ys = (ys,) if isinstance(ys, torch.Tensor) else ys
            new = list(out)
            for i, y in zip(idx, ys):
                new[i] = y
            return type(out)(new) if isinstance(out, list) else tuple(new)
        if hasattr(out, "keys") and hasattr(out, "__setitem__"):        # a ModelOutput (an ordered dict with attributes)
            names = [k for k in out.keys() if isinstance(out[k], torch.Tensor) and out[k].requires_grad
                     and out[k].is_floating_point()]
            if names:
                ys = _Mark.apply(self.tl, j, *[out[k] for k in names])
                ys = (ys,) if isinstance(ys, torch.Tensor) else ys
                for k, y in zip(names, ys):
                    out[k] = y
            return out
        return out

    def module(self, mod: torch.nn.Module, comp: str):
        tl = self.tl

        def pre(m, args, kwargs):
            j = tl.enter(comp)
            return self._wrap_args(j, args, kwargs)

        def post(m, args, kwargs, out):
            j = tl.exit(comp)
            return self._wrap_out(j, out)

        self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        self.handles.append(mod.register_forward_hook(post, with_kwargs=True))

    def function(self, owner, name: str, comp: str):
        orig = getattr(owner, name)
        tl = self.tl

        def wrapped(*args, **kwargs):
            j = tl.enter(comp)
            args, kwargs = self._wrap_args(j, args, kwargs)
            out = orig(*args, **kwargs)
            j = tl.exit(comp)
            return self._wrap_out(j, out)

        wrapped.__wrapped__ = orig
        setattr(owner, name, wrapped)
        self.patched.append((owner, name, orig))
        return orig

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for owner, name, orig in reversed(self.patched):
            setattr(owner, name, orig)
        self.patched = []
