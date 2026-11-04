"""Series design D3 (the rule's chunking and tiling; docs/series_layer.md): the probe in a real training run. The Phase 4 runner
(bench/phase4_train.py: the trainer's own run(), the sample's pictures, LoRA rank 32, bf16 autocast, the kernels with the
recomputed states, the admitted entries as given) with D3's options and a Trainer that counts, for every micro-batch, the rule's
calls by the path they took (macai.gated_delta.op.CALLS: the forward's chunk length, the split, the double buffering; the
backward's split and double buffering) and checks that every linear-attention layer's forward and backward took the configured
path at every micro-batch (the model's layers counted from the model).

    RUNNER=bench/d3_probe.py LOGS=bench/layer_logs WALL=900 HF_HOME=... GUARD=... PY=... sh bench/phase4_run.sh probe_d3_0.8b \\
        --model 0.8b --arm kernels --memory recompute --layer RMSNORM,CONV,GATED_NORM,GATES,SWIGLU --head-at-labels \\
        --vision-cache --lora-fused --dbuf --epochs 1 --batch 2 --accum 2 --n-items 12 --empty-cache

The record (<logdir>/<name>.json) is the runner's, with the probe under "probe" (bar_met: every micro-batch's calls as configured).
"""
import faulthandler
import os
import sys

print(f"[probe] runner pid {os.getpid()}", flush=True)
faulthandler.dump_traceback_later(300, repeat=False, file=sys.stdout)

import json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase4_train  # noqa: E402


def customize(Base, rec, args):
    from macai.gated_delta import _chunked, op
    P = rec.setdefault("probe", {"micro_batches": [], "expected": None, "linear_attention_layers": None})

    def expected():
        return {"fwd_chunk": op.FWD_CHUNK, "value_split": op.VALUE_SPLIT, "dbuf": _chunked.dbuf_for(128, op.FWD_CHUNK),
                "dbuf_bwd": _chunked.dbuf_for(128, _chunked.C)}

    class Probed(Base):
        def __init__(self, *x, **kw):
            super().__init__(*x, **kw)
            faulthandler.cancel_dump_traceback_later()

        def training_step(self, model, inputs, num_items_in_batch=None):
            if P["linear_attention_layers"] is None:
                P["linear_attention_layers"] = sum(type(m).__name__ == "Qwen3_5GatedDeltaNet" for m in model.modules())
                P["expected"] = expected()
            before = dict(op.CALLS)
            out = super().training_step(model, inputs, num_items_in_batch)
            delta = {k: v - before.get(k, 0) for k, v in op.CALLS.items() if v - before.get(k, 0)}
            P["micro_batches"].append({"step": self.state.global_step + 1, "tokens": list(inputs["input_ids"].shape),
                                       "calls": delta})
            return out

    return Probed


def verdict(rec) -> dict:
    """Every micro-batch: one forward and one backward a linear-attention layer, each on the configured path."""
    P = rec.get("probe") or {}
    L, e = P.get("linear_attention_layers"), P.get("expected") or {}
    bad = []
    from macai.gated_delta import _chunked, op
    heads = 16 if rec["model_id"].endswith("0.8B") else 32
    for mb in P.get("micro_batches", []):
        B, T = mb["tokens"]
        sf = op.split_for(B * heads, T, 128, e["fwd_chunk"])
        sb = op.split_for(B * heads, T, 128, _chunked.C)
        want = {f"fwd c{e['fwd_chunk']} s{sf} d{e['dbuf']}": L, f"bwd s{sb} d{e['dbuf_bwd']}": L}
        if mb["calls"] != want:
            bad.append({"step": mb["step"], "tokens": mb["tokens"], "calls": mb["calls"], "want": want})
    return {"micro_batches": len(P.get("micro_batches", [])), "mismatches": bad[:10], "n_mismatches": len(bad),
            "bar_met": bool(P.get("micro_batches")) and not bad}


def main() -> int:
    holder = {}

    def cust(Trainer, rec, args):
        holder["rec"] = rec
        return customize(Trainer, rec, args)
    rc = phase4_train.main(sys.argv[1:], customize=cust)
    rec = holder.get("rec")
    if rec is not None and rec.get("probe"):
        v = verdict(rec)
        rec["probe"]["verdict"] = v
        print(f"[probe] verdict {json.dumps(v)[:2000]}", flush=True)
        # the runner has written its record; add the verdict to it
        logdir = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--logdir"), os.path.join(HERE, "phase4_logs"))
        path = os.path.join(logdir, rec["name"] + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[probe] wrote {path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
