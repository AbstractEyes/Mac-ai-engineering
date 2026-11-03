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
    rows = tl.attribute()        # {(component, "fwd" | "bwd"): [gpu ms, cpu ms, segments, net cpu ms, gpu ms waiting on the cpu]}
    inst.remove(); tl.release()

Every mark costs a Metal event record: PyTorch ends the current command encoder and commits the command buffer at each one,
so a mark perturbs the stream it measures. bench/timeline_check.py measures that cost and the instrument's accuracy.

The fence: before each event the timeline issues one tiny kernel (an in-place add on a 1-element tensor). A timing event whose
command buffer holds no GPU work (two marks back to back, or a mark right after an operation that committed its own command
buffer) is reported late by up to ~0.1 ms while the GPU runs the next buffer, so short segments came out negative and time
moved between neighbours. With the fence every event follows work in its own command buffer, and segments match the operations'
own times (bench/layer_logs/timeline_check.json).

The CPU's time of a segment is given twice: gross (from one mark to the next, the marks' own cost included) and net (from the
end of one mark's own work, its fence and event, to the start of the next's: the hooks' Python and the autograd nodes of the
instrument stay in it).
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


def warm_events(n: int) -> None:
    """Create n timing events once and return them to PyTorch's pool, so a profiled pass does not pay their creation."""
    fence = torch.zeros(1, device="mps")
    ids = []
    for _ in range(n):
        e = _acquire()
        fence.add_(1.0)
        _record(e)
        ids.append(e)
    torch.mps.synchronize()
    for e in ids:
        _release(e)


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
    """Marks: enter / exit (program marks on the issuing thread: a component entered or left) and B (backward nodes, each
    carrying the index of the program mark that created it)."""

    def __init__(self, fence: bool = True):
        if not _HAS_RAW:
            raise RuntimeError("this PyTorch has no MPS timing events")
        self.on = False
        self.fence = torch.zeros(1, device="mps") if fence else None
        self.clear()

    def clear(self):
        self.eid, self.cpu, self.cpu0, self.kind, self.comp, self.ref = [], [], [], [], [], []

    def start(self):
        self.on = True

    def stop(self):
        self.on = False

    def _add(self, kind, comp, ref) -> int:
        t0 = time.perf_counter_ns()
        eid = _acquire()
        if self.fence is not None:
            self.fence.add_(1.0)        # needs no gradient: no autograd record
        _record(eid)
        self.cpu0.append(t0)
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
        """Each mark's time in ms after the first mark. Each elapsed-time query waits for its events; the device is
        synchronized first."""
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

    def owners(self):
        """For each mark i, the (component, direction) its following segment belongs to; and the stack checks."""
        n = len(self.eid)
        owner_after, prev_p = {}, {}
        stack, mismatches, last_p = [], 0, None
        for i in range(n):
            k = self.kind[i]
            if k in ("enter", "exit"):
                prev_p[i] = last_p
                last_p = i
                if k == "enter":
                    stack.append(self.comp[i])
                elif stack and stack[-1] == self.comp[i]:
                    stack.pop()
                else:
                    mismatches += 1
                    for q in range(len(stack) - 1, -1, -1):
                        if stack[q] == self.comp[i]:
                            del stack[q]
                            break
                owner_after[i] = stack[-1] if stack else "(outside)"
        out = []
        for i in range(n):
            if self.kind[i] == "B":
                p = prev_p.get(self.ref[i])
                out.append((owner_after[p] if p is not None else "(outside)", "bwd"))
            else:
                c = owner_after[i]
                # a program component named "<name>:bwd" (the call that runs the backward) is reported as the backward's
                out.append((c[:-4], "bwd") if c.endswith(":bwd") else (c, "fwd"))
        return out, {"stack_mismatches": mismatches, "open_at_end": list(stack)}

    def lags(self, times) -> list:
        """How far the GPU trails the CPU at each mark, ms: the event's time minus the CPU's issue time, both from mark 0.
        Near the timeline's minimum the GPU had caught up with the CPU (it was waiting for work); well above it the CPU was
        ahead. The CPU's issue time alone cannot tell the two apart: when the GPU is behind, the CPU waits for a free slot in
        Metal's queue of command buffers, and that wait lands in the issue time of the next operation."""
        if not times:
            return []
        c0 = self.cpu[0]
        return [times[i] - (self.cpu[i] - c0) / 1e6 for i in range(len(times))]

    def attribute(self, times=None, caught_up_ms: float = 1.0) -> dict:
        """{(component, direction): [gpu ms, cpu ms gross, segments, cpu ms net, gpu ms waiting]} by exclusive time, plus the
        timeline's checks under the key '_checks' (the span, the sum of the segments, segments with a negative GPU time, the
        stack, the lag's minimum and median). 'gpu ms waiting': the segment's GPU time when the GPU had caught up with the CPU
        at the segment's end (its lag within caught_up_ms of the timeline's minimum): the GPU was waiting for the CPU."""
        t = self.resolve() if times is None else times
        n = len(t)
        own, checks = self.owners()
        lag = self.lags(t)
        lag_min = min(lag) if lag else 0.0
        rows, neg = {}, 0
        for i in range(n - 1):
            g = t[i + 1] - t[i]
            c = (self.cpu[i + 1] - self.cpu[i]) / 1e6
            cn = (self.cpu0[i + 1] - self.cpu[i]) / 1e6
            if g < 0:
                neg += 1
            r = rows.setdefault(own[i], [0.0, 0.0, 0, 0.0, 0.0])
            r[0] += g
            r[1] += c
            r[2] += 1
            r[3] += cn
            if lag[i + 1] < lag_min + caught_up_ms:
                r[4] += g
        srt = sorted(lag)
        rows["_checks"] = dict(checks, marks=n, backward_marks=sum(1 for k in self.kind if k == "B"),
                               span_ms=(t[-1] - t[0]) if n else 0.0,
                               sum_ms=sum(v[0] for k, v in rows.items() if k != "_checks"), negative_segments=neg,
                               lag_min_ms=round(lag_min, 4), lag_median_ms=round(srt[len(srt) // 2], 4) if srt else 0.0)
        return rows

    def raw(self, times) -> list:
        """The marks as rows (index, kind, component, forward mark it mirrors, GPU ms, CPU ms), for one timeline kept whole."""
        c0 = self.cpu[0] if self.cpu else 0
        return [[i, self.kind[i], self.comp[i], self.ref[i], round(times[i], 4), round((self.cpu[i] - c0) / 1e6, 4)]
                for i in range(len(times))]


def _float_grad_tensors(xs):
    return [i for i, x in enumerate(xs) if isinstance(x, torch.Tensor) and x.requires_grad and x.is_floating_point()]


class Instrument:
    """Marks on modules (forward pre / post hooks) and on functions or methods (wrappers), each with its backward nodes.
    While `quiet` is above zero (inside a component marked with quiet=True) nested hooks and wrappers mark nothing: a whole
    layer can be timed as one component."""

    def __init__(self, tl: Timeline, backward: bool = True):
        self.tl = tl
        self.backward = backward
        self.handles = []
        self.patched = []
        self.quiet = 0

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

    def wrap_out(self, j, out):
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
            return new if isinstance(out, list) else tuple(new)
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

    def module(self, mod: torch.nn.Module, comp: str, quiet: bool = False):
        tl = self.tl
        marked = []         # per call (calls of one module may nest): whether its enter was marked

        def pre(m, args, kwargs):
            if self.quiet:
                marked.append(False)
                return None
            marked.append(True)
            j = tl.enter(comp)
            if quiet:
                self.quiet += 1
            return self._wrap_args(j, args, kwargs)

        def post(m, args, kwargs, out):
            if not marked.pop():
                return None
            if quiet:
                self.quiet -= 1
            j = tl.exit(comp)
            return self.wrap_out(j, out)

        self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        self.handles.append(mod.register_forward_hook(post, with_kwargs=True))

    def wrap(self, fn, comp):
        """fn wrapped with an enter / exit mark; comp is a name, or a function of (args, kwargs) giving the name."""
        tl = self.tl

        def wrapped(*args, **kwargs):
            if self.quiet:
                return fn(*args, **kwargs)
            name = comp(args, kwargs) if callable(comp) else comp
            j = tl.enter(name)
            args, kwargs = self._wrap_args(j, args, kwargs)
            out = fn(*args, **kwargs)
            j = tl.exit(name)
            return self.wrap_out(j, out)

        wrapped.__wrapped__ = fn
        return wrapped

    def replace(self, owner, name: str, new):
        """Set owner.name = new until remove(); an attribute owner held only through its class is deleted again then."""
        had = name in getattr(owner, "__dict__", {})
        self.patched.append((owner, name, owner.__dict__[name] if had else None, had))
        setattr(owner, name, new)

    def function(self, owner, name: str, comp):
        """Wrap owner.name (a module-level function, or a bound method of an instance) until remove()."""
        orig = getattr(owner, name)
        self.replace(owner, name, self.wrap(orig, comp))
        return orig

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for owner, name, old, had in reversed(self.patched):
            if had:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
        self.patched = []
        self.quiet = 0
