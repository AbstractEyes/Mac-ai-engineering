"""Series designs D8 and D9 (macai.training; docs/series_layer.md): the probe in a real training run. The Phase 4 runner
(bench/phase4_train.py: the trainer's own run(), the runner's patches, the sample's pictures through the record's collator, LoRA
rank 32, bf16 autocast) with a Trainer that checks, on the first micro-batches that carry pictures, before training on them:

  head  (D8) one micro-batch run with the head at the labelled positions off and on, the same weights (the adapters' B drawn
        for the check, then put back) and the same dropout draws: the loss's relative difference, the logits at the labelled
        positions bit for bit, each LoRA weight's gradient and the final hidden states' gradient against each other, and the
        final hidden states' gradient of both against a float64 CPU run of the head and the loss on the same hidden states
        (the bar: at most twice the model's own path's error);
  tower (D9) the real tower in training mode: whether it draws random numbers (the CPU's and the GPU's generator states
        before and after); its output for the same micro-batch twice (bit for bit); each picture alone against its rows in
        the micro-batch (and, with one picture a micro-batch, paired with the next micro-batch's picture); a cache's served
        output against a recompute (bit for bit, or within the spread just measured); one pixel value moved to the next float
        and the same pixels at another grid, neither served; the key's and the serve's own time.

Then it trains as the runner does (the probe's own calls taken out of the counts), so the run's record shows the head's calls and
the cache's stores (epoch 1) and hits (epoch 2) in a real training step. The probe's draws move the dropout stream, so its
losses after step 1 are not comparable with other runs.

    RUNNER=bench/training_probe.py LOGS=bench/layer_logs HF_HOME=... GUARD=... PY=... sh bench/phase4_run.sh probe_d8d9_0.8b \\
        --model 0.8b --arm kernels --memory recompute --layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU --head-at-labels \\
        --vision-cache --epochs 2 --batch 2 --accum 2 --n-items 12 --empty-cache --probe-micro-batches 3
"""
import os
import sys

print(f"[probe] runner pid {os.getpid()}", flush=True)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase4_train  # noqa: E402


def bits_equal(a, b) -> bool:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


def max_rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def inner_of(model, name):
    return next(m for m in model.modules() if type(m).__name__ == name)


@contextlib.contextmanager
def float64_everywhere():
    """transformers' code casts to float32 in places (x.float(), .to(torch.float32)): for the float64 reference those casts
    keep float64 tensors as they are (CPU only, inside this block; as tests/test_layer.py)."""
    import torch
    orig_float, orig_to = torch.Tensor.float, torch.Tensor.to

    def f(self, *a, **k):
        return self if self.dtype == torch.float64 else orig_float(self, *a, **k)

    def to(self, *a, **k):
        if self.dtype == torch.float64:
            a = tuple(torch.float64 if x is torch.float32 else x for x in a)
            if k.get("dtype") is torch.float32:
                k["dtype"] = torch.float64
        return orig_to(self, *a, **k)
    torch.Tensor.float, torch.Tensor.to = f, to
    try:
        yield
    finally:
        torch.Tensor.float, torch.Tensor.to = orig_float, orig_to


def captured(cg, fn):
    """fn() (a forward returning (loss, logits)) then the backward of the loss: the loss, the logits, the final hidden states and
    their gradient, the input embeddings' gradient (the embedding's output made to require one), every trainable gradient."""
    import torch
    seen = {}

    def keep_final(_m, _i, out):
        out.retain_grad()
        seen["h"] = out

    def keep_embeds(_m, _i, out):
        if not out.requires_grad:
            out.requires_grad_(True)
        out.retain_grad()
        seen["x"] = out
    hooks = [cg.model.language_model.norm.register_forward_hook(keep_final),
             cg.model.language_model.embed_tokens.register_forward_hook(keep_embeds)]
    try:
        loss, logits = fn()
        loss.backward()
    finally:
        for h in hooks:
            h.remove()
    return {"loss": loss.detach().cpu().double(), "logits": logits.detach().cpu(), "h": seen["h"].detach().cpu(),
            "dh": seen["h"].grad.detach().cpu(), "dx": seen["x"].grad.detach().cpu(),
            "grads": {n: p.grad.detach().cpu().double() for n, p in cg.named_parameters() if p.requires_grad}}


def head_check(trainer, model, inputs, float64_model: bool) -> dict:
    """D8 on one micro-batch: the head at the labelled positions off and on, the same weights (the adapters' B drawn, so every
    adapter weight has a gradient) and no dropout (every LoRA dropout an identity for the check, so a float64 run can match
    it); with float64_model, the same model, weights and batch in float64 on the CPU as the reference of every gradient."""
    import torch
    from macai import layer, patch, training
    cg = inner_of(model, "Qwen3_5ForConditionalGeneration")
    lora_b = [p for n, p in cg.named_parameters() if "lora_B" in n]
    saved_b = [p.detach().clone() for p in lora_b]
    drops = [(m, m["default"]) for n, m in cg.named_modules() if n.endswith("lora_dropout") and "default" in m]
    gen = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for p in lora_b:
            p.copy_((torch.randn(p.shape, generator=gen) * 0.02).to(p.device, p.dtype))
    for m, _ in drops:
        m["default"] = torch.nn.Identity()
    trainable = [p for p in cg.parameters() if p.requires_grad]
    labels = inputs["labels"]
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    keep = (shift != -100).cpu()
    res = {}
    try:
        for head in (False, True):
            for p in trainable:
                p.grad = None

            def fwd():
                with training.configured(HEAD_AT_LABELS=head, VISION_CACHE=False), trainer.compute_loss_context_manager():
                    loss, out = trainer.compute_loss(model, dict(inputs), return_outputs=True)
                return loss, out.logits
            res[head] = captured(cg, fwd)
        ref = None
        if float64_model:
            # the reference: the same model (adapters included) in float64 on the CPU, transformers' own code and rule
            t0 = time.perf_counter()
            M = sys.modules[phase4_train.MODULE]
            ref_cg = copy.deepcopy(cg).to("cpu").to(torch.float64).train()
            cpu_inputs = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()}
            saved_rule = M.torch_chunk_gated_delta_rule
            M.torch_chunk_gated_delta_rule = patch._saved.get(M.__name__, saved_rule)
            torch.set_num_threads(8)
            try:
                for p in ref_cg.parameters():
                    p.grad = None

                def fwd64():
                    out = ref_cg(**cpu_inputs)
                    return out.loss, out.logits
                with layer.configured(**dict.fromkeys(layer.PARTS, False)), \
                        training.configured(HEAD_AT_LABELS=False, VISION_CACHE=False), float64_everywhere():
                    ref = captured(ref_cg, fwd64)
            finally:
                M.torch_chunk_gated_delta_rule = saved_rule
            del ref_cg
            ref_s = round(time.perf_counter() - t0, 1)
    finally:
        with torch.no_grad():
            for p, s in zip(lora_b, saved_b):
                p.copy_(s)
        for m, d in drops:
            m["default"] = d
        for p in trainable:
            p.grad = None
    off, on = res[False], res[True]
    # the float64 reference of the head and the loss alone, on the model's own path's final hidden states (the positions with
    # a label; every other position has no gradient on any path)
    W = cg.lm_head.weight.detach().cpu().double()
    h = off["h"].double()[keep].requires_grad_(True)
    tgt = shift.cpu()[keep]
    torch.set_num_threads(8)
    head_loss = torch.nn.functional.cross_entropy(h @ W.t(), tgt)
    head_loss.backward()
    head_dh = h.grad
    dh_off, dh_on = off["dh"].double()[keep], on["dh"].double()[keep]
    others_zero = bool((off["dh"][~keep] == 0).all()) and bool((on["dh"][~keep] == 0).all())
    out = {"positions": int(keep.numel()), "labelled": int(keep.sum()), "lora_dropout": "identity for the check",
           "loss_off": off["loss"].item(), "loss_on": on["loss"].item(),
           "loss_rel": abs(on["loss"].item() - off["loss"].item()) / abs(off["loss"].item()),
           "loss_float64_head_alone": head_loss.item(),
           "logits_shape_off": list(off["logits"].shape), "logits_shape_on": list(on["logits"].shape),
           "logits_labelled_same_bits": bits_equal(on["logits"], off["logits"][keep]),
           "final_hidden_same_bits": bits_equal(on["h"], off["h"]),
           "head_alone_dh_err_off": max_rel(dh_off, head_dh), "head_alone_dh_err_on": max_rel(dh_on, head_dh),
           "dh_on_vs_off_max_rel": max_rel(dh_on, dh_off), "dh_elements_differing": int((dh_on != dh_off).sum()),
           "dh_elements": int(dh_off.numel()), "unlabelled_positions_zero_gradient": others_zero}
    grads = {}
    for n in off["grads"]:
        key = n.replace(".default.weight", "")
        grads[key] = {"on_vs_off_max_rel": max_rel(on["grads"][n], off["grads"][n])}
        if ref is not None:
            grads[key].update(err_off=max_rel(off["grads"][n], ref["grads"][n]), err_on=max_rel(on["grads"][n], ref["grads"][n]))
    out["lora_grads_n"] = len(grads)
    out["lora_grads_on_vs_off_max_rel_worst"] = max(v["on_vs_off_max_rel"] for v in grads.values())
    bar = out["loss_rel"] <= 1e-5 and out["logits_labelled_same_bits"] and out["head_alone_dh_err_on"] <= 2 * out["head_alone_dh_err_off"]
    if ref is not None:
        rows = {"loss": (max_rel(off["loss"], ref["loss"]), max_rel(on["loss"], ref["loss"])),
                "d final hidden": (max_rel(off["dh"].double(), ref["dh"]), max_rel(on["dh"].double(), ref["dh"])),
                "d input embeddings": (max_rel(off["dx"].double(), ref["dx"]), max_rel(on["dx"].double(), ref["dx"]))}
        for k, v in grads.items():
            rows["d " + k] = (v["err_off"], v["err_on"])
        ratios = {k: (v[1] / v[0] if v[0] > 0 else float("inf")) for k, v in rows.items()}
        worst = max(ratios.items(), key=lambda kv: kv[1])
        out["float64_model"] = {"seconds": ref_s, "loss_float64": ref["loss"].item(),
                                "errors": {k: {"model_path": v[0], "head_at_labels": v[1]} for k, v in rows.items()},
                                "worst_ratio": {"name": worst[0], "ratio": worst[1]},
                                "ratio_median": sorted(ratios.values())[len(ratios) // 2],
                                "over_2": [k for k, r in ratios.items() if r > 2 and k != "loss"]}
        bar = bar and not out["float64_model"]["over_2"]
    out["bar_met"] = bar
    return out


def tower_check(trainer, model, inputs, previous) -> dict:
    """D9 on one micro-batch's pictures with the real tower."""
    import torch
    from macai.training import _vision_cache as VC
    vlm = inner_of(model, "Qwen3_5Model")
    pv, grid = inputs["pixel_values"], inputs["image_grid_thw"]
    rows = [int(t) * int(h) * int(w) for t, h, w in grid.tolist()]
    out = {"pictures": len(rows), "rows": rows, "grids": grid.tolist(), "pixel_dtype": str(pv.dtype),
           "pixel_shape": list(pv.shape), "tower_training": vlm.visual.training,
           "tower_frozen": not any(p.requires_grad for p in vlm.visual.parameters())}

    def features(p, g, grad=False):
        with trainer.accelerator.autocast(), torch.set_grad_enabled(grad):
            r = vlm.get_image_features(p, g, return_dict=True)
        return r

    cpu_state, gpu_state = torch.get_rng_state(), torch.mps.get_rng_state()
    a = features(pv, grid, grad=True)                 # as in training: gradients enabled, the tower frozen
    out["autocast_on"] = trainer.accelerator.mixed_precision
    out["rng_cpu_unchanged"] = bool(torch.equal(cpu_state, torch.get_rng_state()))
    out["rng_gpu_unchanged"] = bool(torch.equal(gpu_state, torch.mps.get_rng_state()))
    out["output_fields"] = {k: (list(v.shape) if hasattr(v, "shape") else [list(x.shape) for x in v])
                            for k, v in a.items() if v is not None}
    out["requires_grad"] = bool(a.pooler_output[0].requires_grad)
    a = type(a)(**{k: (v.detach() if hasattr(v, "detach") else tuple(x.detach() for x in v)) for k, v in a.items()})
    b = features(pv, grid)
    out["same_batch_twice_same_bits"] = bits_equal(a.last_hidden_state, b.last_hidden_state) and all(
        bits_equal(x, y) for x, y in zip(a.pooler_output, b.pooler_output))
    # each picture alone against its rows in the micro-batch
    alone, r0 = [], 0
    for i, n in enumerate(rows):
        s = features(pv[r0:r0 + n], grid[i:i + 1])
        alone.append(bits_equal(s.last_hidden_state, a.last_hidden_state[r0:r0 + n])
                     and bits_equal(s.pooler_output[0], a.pooler_output[i]))
        if not alone[-1]:
            out.setdefault("alone_max_rel", []).append(max_rel(s.pooler_output[0].float(), a.pooler_output[i].float()))
        r0 += n
    out["each_picture_alone_same_bits"] = alone
    if previous is not None and len(rows) == 1:
        # paired with the previous micro-batch's picture: this picture second in a two-picture batch
        ppv, pgrid = previous
        pair = features(torch.cat([ppv, pv]), torch.cat([pgrid, grid]))
        n0 = ppv.shape[0]
        out["paired_with_previous_same_bits"] = bits_equal(pair.last_hidden_state[n0:], a.last_hidden_state) and bits_equal(
            pair.pooler_output[-1], a.pooler_output[0])
    # a cache of the probe's own: computed, then served, against a recompute
    c = VC.VisionCache()
    with trainer.accelerator.autocast(), torch.no_grad():
        c.image_outputs(vlm, pv, grid, {})
        c.flush()
        t0 = time.perf_counter()
        served = c.image_outputs(vlm, pv, grid, {})
        torch.mps.synchronize()
        out["served_call_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    recompute = features(pv, grid)
    out["served"] = c.stats["served_calls"] == 1
    out["served_same_bits_as_recompute"] = bits_equal(served.last_hidden_state, recompute.last_hidden_state) and all(
        bits_equal(x, y) for x, y in zip(served.pooler_output, recompute.pooler_output))
    st = c.summary()
    out["key_ms_per_call"] = round(st["key_s"] / st["calls"] * 1e3, 2)
    out["serve_ms"] = round(st["serve_s"] * 1e3, 2)
    out["store_ms"] = round(st["store_s"] * 1e3, 2)
    out["bytes_a_picture"] = round(st["bytes_stored"] / max(st["entries"], 1))
    out["merged_bytes_a_picture"] = round(st["bytes_merged"] / max(st["entries"], 1))
    # one pixel value moved to the next float: not served
    moved = pv.clone()
    v = moved[min(7, rows[0] - 1), 11]
    moved[min(7, rows[0] - 1), 11] = torch.nextafter(v, v + 1)
    before = dict(c.stats)
    with trainer.accelerator.autocast(), torch.no_grad():
        c.image_outputs(vlm, moved, grid, {})
    out["one_pixel_moved_served"] = c.stats["served_calls"] != before["served_calls"]
    out["one_pixel_moved_computed"] = c.stats["computed_calls"] == before["computed_calls"] + 1
    # the same pixels at another grid with as many rows: not served
    t, h, w = (int(x) for x in grid[0].tolist())
    g2 = None
    if h != w:
        g2 = (t, w, h)
    elif w % 4 == 0:
        g2 = (t, 2 * h, w // 2)
    if g2 is not None:
        other = grid.clone()
        other[0] = torch.tensor(g2, dtype=grid.dtype)
        before = dict(c.stats)
        with trainer.accelerator.autocast(), torch.no_grad():
            c.image_outputs(vlm, pv, other, {})
        out["other_grid"] = [list((t, h, w)), list(g2)]
        out["other_grid_served"] = c.stats["served_calls"] != before["served_calls"]
    out["bar_met"] = (out["served"] and (out["served_same_bits_as_recompute"]) and not out["one_pixel_moved_served"]
                      and not out.get("other_grid_served", False))
    return out


def customize(Base, rec, args, opts):
    import torch
    from macai.training import _head, _qwen3_5
    P = rec.setdefault("probe", {"options": vars(opts), "head": [], "tower": []})
    state = {"done": 0, "previous": None}

    class Probed(Base):
        def training_step(self, model, inputs, num_items_in_batch=None):
            if state["done"] < opts.probe_micro_batches and inputs.get("pixel_values") is not None:
                model.train()
                prepared = self._prepare_inputs(inputs)
                t0 = time.perf_counter()
                counts = (dict(_head.STATS), dict(_qwen3_5.FALLBACKS))     # the probe's own calls are taken out below
                if opts.head and state["done"] < opts.head_micro_batches:
                    r = head_check(self, model, prepared, opts.float64_model)
                    r["step"], r["index"] = self.state.global_step + 1, state["done"]
                    P["head"].append(r)
                    print(f"[probe] head {json.dumps({k: r.get(k) for k in ('labelled', 'positions', 'loss_rel', 'logits_labelled_same_bits', 'head_alone_dh_err_off', 'head_alone_dh_err_on', 'lora_grads_on_vs_off_max_rel_worst', 'float64_model', 'bar_met')})[:3000]}", flush=True)
                r = tower_check(self, model, prepared, state["previous"])
                r["step"], r["index"] = self.state.global_step + 1, state["done"]
                P["tower"].append(r)
                print(f"[probe] tower {json.dumps({k: r.get(k) for k in ('pictures', 'rng_cpu_unchanged', 'rng_gpu_unchanged', 'same_batch_twice_same_bits', 'each_picture_alone_same_bits', 'paired_with_previous_same_bits', 'served_same_bits_as_recompute', 'one_pixel_moved_served', 'other_grid_served', 'key_ms_per_call', 'serve_ms', 'bytes_a_picture', 'bar_met')})}", flush=True)
                state["previous"] = (prepared["pixel_values"][-int(prepared["image_grid_thw"][-1].prod()):],
                                     prepared["image_grid_thw"][-1:])
                state["done"] += 1
                P.setdefault("seconds", []).append(round(time.perf_counter() - t0, 1))
                _head.STATS.update(counts[0])
                _qwen3_5.FALLBACKS.clear()
                _qwen3_5.FALLBACKS.update(counts[1])
                torch.mps.empty_cache()
            return super().training_step(model, inputs, num_items_in_batch)

    return Probed


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-micro-batches", type=int, default=3, help="micro-batches with pictures checked before training")
    ap.add_argument("--head-micro-batches", type=int, default=1, help="of those, the ones the head check runs on")
    ap.add_argument("--no-head", dest="head", action="store_false", help="skip the head check")
    ap.add_argument("--float64-model", action="store_true",
                    help="the head check's reference: the whole model in float64 on the CPU (0.8B: ~7 GB of weights)")
    opts, rest = ap.parse_known_args()
    holder = {}

    def cust(Trainer, rec, args):
        holder["rec"] = rec
        return customize(Trainer, rec, args, opts)

    rc = phase4_train.main(rest, customize=cust)
    return rc


if __name__ == "__main__":
    sys.exit(main())
