"""Series D3-D5, step 0 (a): the readers' LoRA training step profiled in situ, by component (docs/series_layer.md).

The Phase 4 runner (bench/phase4_train.py: the model loaded through qwen_symbolic_thinking.train.run with the runner's patches,
the sample, the kernels with the default memory design, checkpointing off, the cache emptied each micro-batch) with a Trainer
that, on the optimizer steps named by --profile-steps, runs each micro-batch three times on the same inputs:

  A  as in training, timed by the wall clock only: its loss and gradient are the ones the step keeps;
  B  again, unprofiled (the input's shapes now warm in PyTorch's caches);
  C  again, profiled: Metal timing events at module and function boundaries, forward and backward, with the CPU's clock beside
     each (bench/timeline.py). B and C alternate in order from one micro-batch to the next.

After B and C the gradients are set back to A's, so the run trains as an unprofiled one does (only the dropout's random draws
move). C against B is the instrument's perturbation on the same inputs in the same process; A - B is what a new input shape
costs. The parts of a profiled step outside its micro-batches (collation, gradient clipping, the optimizer, the scheduler and
zero_grad, logging, the Trainer between micro-batches) are timed with events too. The other steps run unprofiled.

    RUNNER=bench/profile_step.py LOGS=bench/layer_logs HF_HOME=... GUARD=... PY=... sh bench/phase4_run.sh prof_0.8b_2x8 \
        --model 0.8b --arm kernels --memory recompute --epochs 4 --batch 2 --accum 8 --empty-cache \
        --profile-steps 3,5,7 --stop-after 8

The record (<logdir>/<name>.json) is the runner's, with the profile under "profile". bench/profile_report.py reads it.
"""
import os
import sys

print(f"[profile] runner pid {os.getpid()}", flush=True)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase4_train  # noqa: E402
from timeline import Instrument, Timeline, warm_events  # noqa: E402

MODULE = phase4_train.MODULE


def build_instrument(inst: Instrument, model, M, detail: set):
    """Hooks and wrappers for one profiled pass. detail: the decoder layers timed part by part; the others as one component."""
    import transformers.trainer as trainer_module   # noqa: F401  (the Trainer's own names are patched by the caller)
    inner = next(m for m in model.modules() if type(m).__name__ == "Qwen3_5ForConditionalGeneration")
    vlm, lm_head = inner.model, inner.lm_head
    vis, lm = vlm.visual, vlm.language_model

    # the model's outer wrappers: PEFT's model and its tuner, accelerate's autocast and the outputs converted to float32
    inst.module(model, "top.peft_accelerate")
    if model is not inner and hasattr(model, "base_model") and model.base_model is not inner:
        inst.module(model.base_model, "top.peft_accelerate")
    inst.module(inner, "top.causal_lm_rest")
    inst.module(vlm, "top.merge_rest")
    inst.function(vlm, "get_image_features", "top.image_features_rest")
    inst.function(vlm, "get_placeholder_mask", "top.placeholder_mask")
    inst.function(vlm, "compute_3d_position_ids", "top.position_ids")
    # the vision tower (frozen: forward only)
    inst.module(vis, "vis.rest")
    inst.module(vis.patch_embed, "vis.patch_embed")
    for b in vis.blocks:
        inst.module(b, "vis.blocks")
    inst.module(vis.merger, "vis.merger")
    # the language model's own parts
    inst.module(lm, "top.text_model_rest")
    inst.module(lm.embed_tokens, "top.embed_tokens")
    inst.module(lm.rotary_emb, "top.rotary_emb")
    inst.module(lm.norm, "top.final_norm")
    inst.module(lm_head, "top.lm_head")
    # the loss: transformers resolves model.loss_function through a property on the class
    orig_loss = inner.loss_function
    wrapped_loss = inst.wrap(orig_loss, "top.loss")
    owner = next((k for k in type(inner).__mro__ if "loss_function" in k.__dict__), None)
    if owner is not None and isinstance(owner.__dict__["loss_function"], property):
        inst.replace(type(inner), "loss_function", property(lambda self: wrapped_loss))
    else:
        inst.replace(inner, "loss_function", wrapped_loss)
    assert inner.loss_function is wrapped_loss, "the loss wrapper did not take"
    # module-level functions the layers call
    inst.function(M, "create_causal_mask", "top.masks")
    inst.function(M, "create_recurrent_attention_mask", "top.masks")
    inst.function(M, "causal_conv1d_fn", "lin.conv")
    inst.function(M, "torch_chunk_gated_delta_rule", "lin.rule")
    inst.function(M, "apply_rotary_pos_emb", "full.rotary")
    # the layer parts' functions (macai.layer, series design D4) that the replaced linear-attention forward calls: the fused
    # convolution path under step 0's name for the convolution, beta and g apart (the norms and the SwiGLU keep their modules)
    try:
        from macai.layer import _qwen3_5 as layer_q
    except ImportError:
        layer_q = None
    if layer_q is not None:
        inst.function(layer_q, "conv_path", "lin.conv")
        inst.function(layer_q, "gates_path", "lin.gates")
    # the training path (macai.training, series designs D8 and D9): the hidden states gathered at the labelled positions before
    # the head and the labels' shift and gather (in the LM head and the loss); the vision cache's own work (keys, serving,
    # stores; the tower's marked parts inside it excluded) in the vision tower
    inst.function(M, "slice_logits_to_keep", "top.lm_head_gather")
    try:
        from macai.training import _qwen3_5 as training_q, _vision_cache as vision_cache
    except ImportError:
        training_q = vision_cache = None
    if training_q is not None:
        inst.function(training_q, "head_positions", "top.loss_labels")
        inst.function(training_q, "head_targets", "top.loss_labels")
        inst.replace(vision_cache.VisionCache, "image_outputs",
                     inst.wrap(vision_cache.VisionCache.__dict__["image_outputs"], "vis.cache"))
    # the attention call, through transformers' attention registry (the vision tower uses it too)
    af = M.ALL_ATTENTION_FUNCTIONS
    orig_get = af.get_interface
    cache = {}

    def attention_name(args, kwargs):
        return "vis.attention" if type(args[0]).__name__ == "Qwen3_5VisionAttention" else "full.attention"

    def get_interface(*a, **k):
        fn = orig_get(*a, **k)
        w = cache.get(id(fn))
        if w is None:
            w = cache[id(fn)] = inst.wrap(fn, attention_name)
        return w

    inst.replace(af, "get_interface", get_interface)

    def lora(mod, name):
        if hasattr(mod, "base_layer") and hasattr(mod, "lora_A"):
            inst.module(mod, f"{name}.lora_rest")
            inst.module(mod.base_layer, f"{name}.base")
            inst.module(mod.lora_dropout["default"], f"{name}.lora_dropout")
            inst.module(mod.lora_A["default"], f"{name}.lora_A")
            inst.module(mod.lora_B["default"], f"{name}.lora_B")
        else:
            inst.module(mod, f"{name}.base")

    for idx, layer in enumerate(lm.layers):
        t = "lin" if layer.block_type == "linear_attention" else "full"
        if idx not in detail:
            inst.module(layer, f"{t}.layer_undetailed", quiet=True)
            continue
        inst.module(layer, f"{t}.residual_adds")
        inst.module(layer.input_layernorm, f"{t}.input_layernorm")
        inst.module(layer.post_attention_layernorm, f"{t}.post_attention_layernorm")
        inst.module(layer.mlp, f"{t}.swiglu")
        for n in ("gate_proj", "up_proj", "down_proj"):
            lora(getattr(layer.mlp, n), f"{t}.{n}")
        if t == "lin":
            a = layer.linear_attn
            inst.module(a, "lin.glue")
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                lora(getattr(a, n), f"lin.{n}")
            inst.module(a.norm, "lin.gated_norm")
        else:
            a = layer.self_attn
            inst.module(a, "full.attn_glue")
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lora(getattr(a, n), f"full.{n}")
            inst.module(a.q_norm, "full.q_norm")
            inst.module(a.k_norm, "full.k_norm")
    return {"layers": len(lm.layers), "detailed": sorted(detail),
            "layer_types": ["lin" if l.block_type == "linear_attention" else "full" for l in lm.layers]}


def shapes_of(inputs) -> dict:
    out = {}
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            out[k] = list(v.shape)
    am = inputs.get("attention_mask")
    if am is not None:
        out["real_tokens"] = int(am.sum())
    lab = inputs.get("labels")
    if lab is not None:
        out["loss_tokens"] = int((lab != -100).sum())
    g = inputs.get("image_grid_thw")
    if g is not None:
        out["pictures"] = int(g.shape[0])
        out["picture_patches"] = int(g.prod(-1).sum())
    return out


def rows_to_json(rows):
    """component|direction: [gpu ms, cpu ms gross, segments, cpu ms net, gpu ms while the GPU waited on the CPU]"""
    return {f"{k[0]}|{k[1]}": [round(v[0], 4), round(v[1], 4), v[2], round(v[3], 4), round(v[4], 4)] for k, v in rows.items()
            if k != "_checks"}


class Profiler:
    def __init__(self, opts, rec):
        self.o = opts
        self.rec = rec
        self.steps = set(opts.profile_steps)
        self.tl = Timeline()
        self.stl = Timeline()
        self.inst = None
        self.mb = 0                 # profiled micro-batches so far (B / C order, the detailed layers' rotation)
        self.kept = 0
        self.collator = []          # [global step being run, CPU ms] for each collator call
        self.state = None           # the Trainer's state (set by the Trainer)
        self.cur = None             # the step being profiled
        self.out = rec.setdefault("profile", {"options": vars(opts), "micro_batches": [], "steps": [], "raw_timelines": []})

    def profiling(self) -> bool:
        return self.state is not None and (self.state.global_step + 1) in self.steps

    def detail(self, n_layers):
        if self.o.detail == "all":
            return set(range(n_layers))
        k = int(self.o.detail.split(":")[1])
        return {i for i in range(n_layers) if i % k == self.mb % k}


def customize(Base, rec, args, opts, holder):
    import torch
    import transformers.trainer as trainer_module
    import qwen_symbolic_thinking.train as T
    import importlib
    M = importlib.import_module(MODULE)
    P = Profiler(opts, rec)
    rec["runner_patches"] = rec.get("runner_patches", []) + [
        f"profile: steps {sorted(P.steps)} run each micro-batch as A (trained), B (unprofiled replay), C (profiled replay); "
        "the gradients set back to A's", f"stop after step {opts.stop_after}"]

    # the collator's own CPU time, every call (a class-level wrapper: the Trainer's dataloader holds the instance)
    orig_call = T.Collator.__call__

    def timed_call(self, pairs):
        t0 = time.perf_counter()
        out = orig_call(self, pairs)
        P.collator.append([(P.state.global_step + 1) if P.state is not None else 0, round((time.perf_counter() - t0) * 1e3, 3)])
        return out

    T.Collator.__call__ = timed_call

    def sync():
        torch.mps.synchronize()

    class Profiled(Base):
        def __init__(self, *x, **kw):
            super().__init__(*x, **kw)
            P.state = self.state
            P.trainable = None

        # ---- the step's outer parts (events on the step timeline) ----
        def get_batch_samples(self, epoch_iterator, num_batches, device):
            P.state = self.state
            if not P.profiling():
                return super().get_batch_samples(epoch_iterator, num_batches, device)
            step = self.state.global_step + 1
            if P.trainable is None:         # once, before the first profiled step's timeline starts
                P.trainable = [p for p in self.model.parameters() if p.requires_grad]
                warm_events(opts.warm_events)
            sync()
            P.cur = {"step": step, "t0": time.perf_counter(), "mb_windows_ms": [], "a_ms": [], "b_ms": [], "c_ms": []}
            P.stl.start()
            P.stl.enter("step.trainer")
            P.stl.enter("step.collate_and_load")
            out = super().get_batch_samples(epoch_iterator, num_batches, device)
            P.stl.exit("step.collate_and_load")
            return out

        def _clip_grad_norm(self, model):
            if not P.stl.on:
                return super()._clip_grad_norm(model)
            P.stl.enter("step.clip_grad_norm")
            out = super()._clip_grad_norm(model)
            P.stl.exit("step.clip_grad_norm")
            return out

        # the Trainer's own calls after each micro-batch (the FLOP count walks the model's parameters each time)
        def floating_point_ops(self, inputs):
            if not P.stl.on:
                return super().floating_point_ops(inputs)
            P.stl.enter("step.trainer_flop_count")
            out = super().floating_point_ops(inputs)
            P.stl.exit("step.trainer_flop_count")
            return out

        def _track_num_input_tokens(self, inputs):
            if not P.stl.on:
                return super()._track_num_input_tokens(inputs)
            P.stl.enter("step.trainer_token_count")
            out = super()._track_num_input_tokens(inputs)
            P.stl.exit("step.trainer_token_count")
            return out

        def _maybe_log_save_evaluate(self, *x, **kw):
            if not P.stl.on:
                return super()._maybe_log_save_evaluate(*x, **kw)
            P.stl.enter("step.logging")
            out = super()._maybe_log_save_evaluate(*x, **kw)
            P.stl.exit("step.logging")
            P.stl.exit("step.trainer")
            P.stl.stop()
            sync()
            wall = (time.perf_counter() - P.cur["t0"]) * 1e3
            times = P.stl.resolve()
            rows = P.stl.attribute(times)
            checks = rows.pop("_checks")
            raw_step = P.stl.raw(times)
            P.stl.release()
            outside = {k: v for k, v in rows.items() if k[0] != "step.micro_batches"}
            windows = rows.get(("step.micro_batches", "fwd"), [0.0, 0.0, 0, 0.0])
            entry = {"step": P.cur["step"], "micro_batches": len(P.cur["a_ms"]),
                     "wall_ms_with_replays": round(wall, 2), "micro_batch_windows_ms": round(windows[0], 2),
                     "a_ms": P.cur["a_ms"], "b_ms": P.cur["b_ms"], "c_ms": P.cur["c_ms"],
                     "outside_ms": rows_to_json(outside), "checks": checks,
                     "collator_cpu_ms": [c[1] for c in P.collator if c[0] == P.cur["step"]], "raw_step_timeline": raw_step,
                     "trainer_args": {"include_num_input_tokens_seen": str(self.args.include_num_input_tokens_seen),
                                      "logging_nan_inf_filter": bool(self.args.logging_nan_inf_filter),
                                      "max_grad_norm": self.args.max_grad_norm,
                                      "dataloader_num_workers": self.args.dataloader_num_workers,
                                      "dataloader_pin_memory": bool(self.args.dataloader_pin_memory)}}
            entry["outside_sum_ms"] = round(sum(v[0] for v in outside.values()), 2)
            entry["step_ms_as_trained"] = round(sum(P.cur["a_ms"]) + entry["outside_sum_ms"], 2)
            P.out["steps"].append(entry)
            print(f"[profile] step {P.cur['step']}: as trained {entry['step_ms_as_trained'] / 1e3:.2f} s (micro-batches "
                  f"{sum(P.cur['a_ms']) / 1e3:.2f} s, outside {entry['outside_sum_ms'] / 1e3:.2f} s); C / B "
                  f"{sum(P.cur['c_ms']) / max(sum(P.cur['b_ms']), 1e-9):.4f}", flush=True)
            P.cur = None
            return out

        # ---- the micro-batch ----
        def _prepare_inputs(self, inputs):
            if not P.tl.on:
                return super()._prepare_inputs(inputs)
            P.tl.enter("mb.prepare_inputs")
            out = super()._prepare_inputs(inputs)
            P.tl.exit("mb.prepare_inputs")
            return out

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not P.tl.on:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            P.tl.enter("mb.compute_loss_rest")
            out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            j = P.tl.exit("mb.compute_loss_rest")
            return P.inst.wrap_out(j, out) if P.inst is not None else out

        def training_step(self, model, inputs, num_items_in_batch=None):
            if not P.profiling():
                return super().training_step(model, inputs, num_items_in_batch)
            k = P.mb
            P.stl.enter("step.micro_batches")
            sync()
            t0 = time.perf_counter()
            loss = super().training_step(model, inputs, num_items_in_batch)        # A: the micro-batch as trained
            sync()
            a_ms = (time.perf_counter() - t0) * 1e3
            saved = [(p, None if p.grad is None else p.grad.clone()) for p in P.trainable]
            b_ms = c_ms = None
            rows = None
            for run in ("B", "C") if k % 2 == 0 else ("C", "B"):
                if run == "B":
                    sync()
                    t0 = time.perf_counter()
                    super().training_step(model, inputs, num_items_in_batch)
                    sync()
                    b_ms = (time.perf_counter() - t0) * 1e3
                else:
                    c_ms, rows, info = self._profiled(model, inputs, num_items_in_batch, k)
            for p, g in saved:                                                      # back to A's gradients
                p.grad = g
            sync()
            entry = {"step": self.state.global_step + 1, "index": k, "shapes": shapes_of(inputs),
                     "a_ms": round(a_ms, 2), "b_ms": round(b_ms, 2), "c_ms": round(c_ms, 2),
                     "c_over_b": round(c_ms / b_ms, 4), "order": "BC" if k % 2 == 0 else "CB", **info,
                     "rows": rows_to_json(rows)}
            P.out["micro_batches"].append(entry)
            P.cur["a_ms"].append(round(a_ms, 2))
            P.cur["b_ms"].append(round(b_ms, 2))
            P.cur["c_ms"].append(round(c_ms, 2))
            print(f"[profile] step {entry['step']} mb {k}: tokens {entry['shapes'].get('input_ids')} A {a_ms:.0f} B {b_ms:.0f} "
                  f"C {c_ms:.0f} ms (C/B {c_ms / b_ms:.3f}); events {info['checks']['sum_ms']:.0f} ms over "
                  f"{info['checks']['marks']} marks", flush=True)
            P.mb += 1
            P.stl.exit("step.micro_batches")
            return loss

        def _profiled(self, model, inputs, num_items_in_batch, k):
            inst = P.inst = Instrument(P.tl)
            plan = build_instrument(inst, model, M, P.detail(len(next(
                m for m in model.modules() if type(m).__name__ == "Qwen3_5TextModel").layers)))
            # the Trainer's own calls inside the micro-batch: the cache emptied between forward and backward, the backward
            orig_clear = trainer_module.clear_device_cache

            def clear_device_cache(*x, **kw):
                P.tl.enter("mb.empty_cache")
                out = orig_clear(*x, **kw)
                P.tl.exit("mb.empty_cache")
                return out

            inst.replace(trainer_module, "clear_device_cache", clear_device_cache)
            orig_backward = self.accelerator.backward

            def backward(loss, **kw):
                P.tl.enter("mb.backward:bwd")
                out = orig_backward(loss, **kw)
                P.tl.exit("mb.backward:bwd")
                return out

            inst.replace(self.accelerator, "backward", backward)
            sync()
            P.tl.start()
            t0 = time.perf_counter()
            P.tl.enter("mb.training_step_rest")
            super().training_step(model, inputs, num_items_in_batch)
            P.tl.exit("mb.training_step_rest")
            sync()
            c_ms = (time.perf_counter() - t0) * 1e3
            P.tl.stop()
            times = P.tl.resolve()
            rows = P.tl.attribute(times)
            checks = rows.pop("_checks")
            if P.kept < opts.keep_timelines:
                P.out["raw_timelines"].append({"step": self.state.global_step + 1, "index": k, "marks": P.tl.raw(times)})
                P.kept += 1
            P.tl.release()
            inst.remove()
            P.inst = None
            return c_ms, rows, {"checks": checks, "plan": {"detailed": len(plan["detailed"]), "layers": plan["layers"]},
                                "events_span_over_c": round(checks["span_ms"] / c_ms, 4)}

    class Stopper(phase4_train_callback_base()):
        def on_pre_optimizer_step(self, args, state, control, **kw):
            if P.stl.on:
                P.stl.enter("step.optimizer")

        def on_optimizer_step(self, args, state, control, **kw):
            if P.stl.on:
                P.stl.exit("step.optimizer")
                P.stl.enter("step.scheduler_zero_grad")

        def on_step_end(self, args, state, control, **kw):
            if P.stl.on:
                P.stl.exit("step.scheduler_zero_grad")
            if opts.stop_after and state.global_step >= opts.stop_after:
                control.should_training_stop = True

    stopper = Stopper()

    class WithStopper(Profiled):
        def __init__(self, *x, **kw):
            kw["callbacks"] = list(kw.get("callbacks") or []) + [stopper]
            super().__init__(*x, **kw)

    def at_end():
        P.out["collator_calls"] = P.collator

    holder["at_end"] = at_end
    return WithStopper


def phase4_train_callback_base():
    import transformers
    return transformers.TrainerCallback


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--profile-steps", default="3,5,7", help="the optimizer steps profiled (1-based), comma-separated")
    ap.add_argument("--stop-after", type=int, default=8, help="stop training after this optimizer step (0: run the epochs)")
    ap.add_argument("--detail", default="all", help="'all': every decoder layer part by part; 'every:K': one layer in K "
                                                    "(rotating with each profiled micro-batch), the others as one component")
    ap.add_argument("--keep-timelines", type=int, default=2, help="profiled micro-batches whose marks are kept whole")
    ap.add_argument("--warm-events", type=int, default=12000, help="timing events created before the first profiled pass")
    opts, rest = ap.parse_known_args()
    opts.profile_steps = [int(s) for s in opts.profile_steps.split(",") if s]
    holder = {}

    def cust(Trainer, rec, args):
        holder["rec"] = rec
        return customize(Trainer, rec, args, opts, holder)

    rc = phase4_train.main(rest, customize=cust)
    # the runner wrote its record; add the profile's closing fields and write it again
    rec = holder.get("rec")
    if rec is not None:
        if "at_end" in holder:
            holder["at_end"]()
        a = phase4_train_args(rest)
        path = os.path.join(a.logdir, f"{a.name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[profile] wrote {path}", flush=True)
    return rc


def phase4_train_args(rest):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--name")
    ap.add_argument("--logdir", default=os.path.join(HERE, "phase4_logs"))
    a, _ = ap.parse_known_args(rest)
    return a


if __name__ == "__main__":
    sys.exit(main())
