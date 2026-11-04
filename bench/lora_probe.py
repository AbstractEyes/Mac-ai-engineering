"""Series design D5 (macai.lora; docs/series_layer.md): the probe in a real training run. The Phase 4 runner (bench/phase4_train.py:
the trainer's own run(), the runner's patches, the sample's pictures through the record's collator, LoRA rank 32 with its
dropout, bf16 autocast) with --lora-fused and a Trainer that:

  counts  every call of the fused adapter path for the whole run, by adapted module (the module names from the model): every
          adapted projection must take it at every micro-batch; PEFT's own forward is counted by reason in the record's
          training_counts at each step;
  checks  on the first micro-batch with a picture, before training on it (the adapters' B drawn for the check, so every adapter
          gradient is nonzero; put back after):
            (a) dropout on: the micro-batch's forward and backward with the fused path off (PEFT's forward) and on, both from
                the same generator state: every mask the same (PEFT's dropout modules' masks against the fused path's), the
                loss's relative difference, each LoRA weight's gradient and the input embeddings' gradient against each other;
            (b) with --float64-model (0.8B): every LoRA dropout an identity, both paths against the same model in float64 on
                the CPU (transformers' code, the layer parts and the head at labels off, transformers' own rule): every
                gradient's error, the fused path's over PEFT's (the bar: at most twice).

The check's draws move the dropout stream, so the run's losses after step 1 are not comparable with other runs.

    RUNNER=bench/lora_probe.py LOGS=bench/layer_logs HF_HOME=... GUARD=... PY=... sh bench/phase4_run.sh probe_d5_0.8b \\
        --model 0.8b --arm kernels --memory recompute --layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU --head-at-labels \\
        --vision-cache --lora-fused --epochs 1 --batch 2 --accum 2 --n-items 12 --empty-cache --float64-model
"""
import faulthandler
import os
import sys

print(f"[probe] runner pid {os.getpid()}", flush=True)
# a first run of this probe stopped before the model's load (no CPU, no GPU memory, SIGINT not answered): if the Trainer is not
# built within 5 minutes, every thread's stack goes to the log (cancelled once it is built)
faulthandler.dump_traceback_later(300, repeat=False, file=sys.stdout)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase4_train  # noqa: E402
import training_probe as TP  # noqa: E402


def masks_of_both(cg, fn, on):
    """fn() run with the fused path on or off, capturing every mask drawn: PEFT's dropout modules' (kept, input zero) or the
    fused path's (torch's dropout call)."""
    import torch
    from macai import lora
    from macai.lora import _fused
    peft_masks, fused_masks = [], []
    hooks = [m.register_forward_hook(lambda _m, i, o: peft_masks.append(((o != 0).cpu(), (i[0] == 0).cpu())))
             for n, m in cg.named_modules() if n.endswith("lora_dropout.default") and isinstance(m, torch.nn.Dropout)]
    orig = _fused.dropout

    def rec(x, p):
        out = orig(x, p)
        fused_masks.append(out[1].cpu())
        return out
    _fused.dropout = rec
    try:
        with lora.configured(FUSED=on):
            out = fn()
    finally:
        for h in hooks:
            h.remove()
        _fused.dropout = orig
    return out, (fused_masks if on else peft_masks)


def lora_check(trainer, model, inputs, float64_model: bool) -> dict:
    import torch
    from macai import layer, lora, patch, training
    from macai.lora import _peft
    cg = TP.inner_of(model, "Qwen3_5ForConditionalGeneration")
    lora_b = [p for n, p in cg.named_parameters() if "lora_B" in n]
    saved_b = [p.detach().clone() for p in lora_b]
    gen = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for p in lora_b:
            p.copy_((torch.randn(p.shape, generator=gen) * 0.02).to(p.device, p.dtype))
    trainable = [p for p in cg.parameters() if p.requires_grad]
    counts = (dict(_peft.CALLS), dict(_peft.FALLBACKS))

    def fwd():
        with training.configured(VISION_CACHE=False), trainer.compute_loss_context_manager():
            loss, out = trainer.compute_loss(model, dict(inputs), return_outputs=True)
        return loss, out.logits
    out = {"positions": int(inputs["labels"].numel())}
    res, masks = {}, {}
    drops = [(m, m["default"]) for n, m in cg.named_modules() if n.endswith("lora_dropout") and "default" in m]
    try:
        # (a) dropout on, the same generator state
        for on in (False, True):
            for p in trainable:
                p.grad = None
            torch.mps.manual_seed(4321)
            res[on], masks[on] = masks_of_both(cg, lambda: TP.captured(cg, fwd), on)
        n_same = sum(bool(torch.equal(m.reshape(k.shape)[~z], k[~z])) for (k, z), m in zip(masks[False], masks[True]))
        out["dropout_on"] = {
            "masks_peft": len(masks[False]), "masks_fused": len(masks[True]), "masks_same": n_same,
            "loss_peft": res[False]["loss"].item(), "loss_fused": res[True]["loss"].item(),
            "loss_rel": abs(res[True]["loss"].item() - res[False]["loss"].item()) / abs(res[False]["loss"].item()),
            "d_input_embeddings_rel": TP.max_rel(res[True]["dx"].double(), res[False]["dx"].double()),
            "d_final_hidden_rel": TP.max_rel(res[True]["dh"].double(), res[False]["dh"].double())}
        g = {n: TP.max_rel(res[True]["grads"][n], res[False]["grads"][n]) for n in res[False]["grads"]}
        out["dropout_on"]["lora_grads"] = len(g)
        out["dropout_on"]["lora_grad_rel_worst"] = max(g.items(), key=lambda kv: kv[1])
        out["dropout_on"]["lora_grad_rel_median"] = sorted(g.values())[len(g) // 2]
        out["dropout_on"]["lora_grads_same_bits"] = sum(bool(torch.equal(res[True]["grads"][n], res[False]["grads"][n]))
                                                        for n in g)
        del masks, res
        # (b) every dropout an identity, both paths against float64 on the CPU
        if float64_model:
            for m, _ in drops:
                m["default"] = torch.nn.Identity()
            res = {}
            for on in (False, True):
                for p in trainable:
                    p.grad = None
                with lora.configured(FUSED=on):
                    res[on] = TP.captured(cg, fwd)
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
                    o = ref_cg(**cpu_inputs)
                    return o.loss, o.logits
                with layer.configured(**dict.fromkeys(layer.PARTS, False)), lora.configured(FUSED=False), \
                        training.configured(HEAD_AT_LABELS=False, VISION_CACHE=False), TP.float64_everywhere():
                    ref = TP.captured(ref_cg, fwd64)
            finally:
                M.torch_chunk_gated_delta_rule = saved_rule
            del ref_cg
            rows = {"loss": (TP.max_rel(res[False]["loss"], ref["loss"]), TP.max_rel(res[True]["loss"], ref["loss"])),
                    "d input embeddings": (TP.max_rel(res[False]["dx"].double(), ref["dx"]),
                                           TP.max_rel(res[True]["dx"].double(), ref["dx"]))}
            for n in ref["grads"]:
                rows["d " + n.replace(".default.weight", "")] = (TP.max_rel(res[False]["grads"][n], ref["grads"][n]),
                                                                  TP.max_rel(res[True]["grads"][n], ref["grads"][n]))
            ratios = {k: (v[1] / v[0] if v[0] > 0 else float("inf")) for k, v in rows.items()}
            worst = max(ratios.items(), key=lambda kv: kv[1])
            out["float64_model"] = {"seconds": round(time.perf_counter() - t0, 1), "gradients": len(rows) - 1,
                                    "errors": {k: {"peft": v[0], "fused": v[1]} for k, v in rows.items()},
                                    "worst_ratio": {"name": worst[0], "ratio": worst[1]},
                                    "ratio_range": [min(ratios.values()), max(ratios.values())],
                                    "ratio_median": sorted(ratios.values())[len(ratios) // 2],
                                    "over_2": [k for k, r in ratios.items() if r > 2 and k != "loss"],
                                    "peft_error_range": [min(v[0] for v in rows.values()), max(v[0] for v in rows.values())]}
    finally:
        with torch.no_grad():
            for p, s in zip(lora_b, saved_b):
                p.copy_(s)
        for m, d in drops:
            m["default"] = d
        for p in trainable:
            p.grad = None
        _peft.CALLS.clear()
        _peft.CALLS.update(counts[0])
        _peft.FALLBACKS.clear()
        _peft.FALLBACKS.update(counts[1])
    d = out["dropout_on"]
    out["bar_met"] = (d["masks_same"] == d["masks_peft"] == d["masks_fused"] > 0
                      and (not float64_model or not out["float64_model"]["over_2"]))
    return out


def customize(Base, rec, args, opts):
    import torch
    from macai.lora import _peft
    P = rec.setdefault("probe", {"options": vars(opts), "checks": [], "fused_by_module": {}, "adapted_modules": None})
    state = {"done": 0, "names": None, "wrapped": False}
    by_module = P["fused_by_module"]

    def wrap(model):
        names = {}
        for n, m in model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "base_layer") and "default" in getattr(m, "lora_A", {}):
                names[id(m.lora_A["default"].weight)] = n
        P["adapted_modules"] = len(names)
        orig = _peft.adapter_path

        def counted(x, result, a, b, p, s, cdt):
            n = names.get(id(a), "unnamed")
            by_module[n] = by_module.get(n, 0) + 1
            return orig(x, result, a, b, p, s, cdt)
        _peft.adapter_path = counted
        state["orig"] = orig

    class Probed(Base):
        def __init__(self, *x, **kw):
            super().__init__(*x, **kw)
            faulthandler.cancel_dump_traceback_later()

        def training_step(self, model, inputs, num_items_in_batch=None):
            if not state["wrapped"]:
                wrap(model)
                state["wrapped"] = True
            if state["done"] < opts.probe_micro_batches and inputs.get("pixel_values") is not None:
                model.train()
                prepared = self._prepare_inputs(inputs)
                t0 = time.perf_counter()
                saved_counts = dict(by_module)
                r = lora_check(self, model, prepared, opts.float64_model and state["done"] == 0)
                by_module.clear()
                by_module.update(saved_counts)
                r["step"], r["index"], r["seconds"] = self.state.global_step + 1, state["done"], round(time.perf_counter() - t0, 1)
                P["checks"].append(r)
                print(f"[probe] lora {json.dumps(r)[:3000]}", flush=True)
                state["done"] += 1
                torch.mps.empty_cache()
            return super().training_step(model, inputs, num_items_in_batch)

    return Probed


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-micro-batches", type=int, default=1, help="micro-batches with pictures checked before training")
    ap.add_argument("--float64-model", action="store_true",
                    help="(b) on the first checked micro-batch: both paths against the whole model in float64 on the CPU")
    opts, rest = ap.parse_known_args()

    def cust(Trainer, rec, args):
        return customize(Trainer, rec, args, opts)

    return phase4_train.main(rest, customize=cust)


if __name__ == "__main__":
    sys.exit(main())
