"""Series design D4 (the layer's surroundings fused; docs/series_layer.md): every figure and verdict from the records, graded
against the registered bars and predictions (series journal, 2026-11-03 11:2x PST).

    python bench/layer_report.py      # prints the tables, writes bench/layer_logs/d4_results.json

Reads: the parity log of the test run (bench/layer_logs/d4_parity.jsonl, tests/test_layer.py with MACAI_PARITY_LOG); the parts
bench (bench/layer_logs/d4_parts.json, bench/bench_layer.py); the integration runs (bench/layer_logs/<run>.json with the guard's
<run>.log) against Phase 4's reference runs A, B, C (bench/phase4_logs) and D2m's confirmation runs (bench/memory_logs), read the
way Phase 4 and D2m read them (bench/memory_report.py); the step profile with the parts on (bench/profile_report.py's reading)
against step 0's.
"""
import json
import os
from collections import defaultdict

import memory_report as MR
import profile_report as PR

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "layer_logs")
RUN_08 = "d4_0.8b_2x8_100steps"
RUN_9 = "d4_9b_1x16_25steps"
PROFILE = "profile_d4_0.8b_2x8"
# the first integration chain: launched with every part on, but every training forward carries a fresh cache and the linear-
# attention forward then fell back to transformers' own (the convolution path and beta / g did not run): the norms and SwiGLU
FIRST = {"run_08": "d4_norms_swiglu_0.8b_2x8_100steps", "run_9": "d4_norms_swiglu_9b_1x16_25steps",
         "profile": "profile_d4_norms_swiglu_0.8b_2x8"}
D2M_08 = os.path.join(HERE, "memory_logs", "confirm_recompute_0.8b_2x8_100steps")
D2M_9 = os.path.join(HERE, "memory_logs", "confirm_recompute_9b_1x16_100steps")
BARS = {"speedup": 1.5, "parity_ratio": 2.0, "footprint_08_gb": 53.5, "footprint_9_gb": 74.6}
PRED = {"floor_ratio": 3.0, "layer_08_ms": (44, 50), "layer_9_ms": (158, 170), "step_08_s": (9.6, 10.4),
        "step_9_s": (45, 49)}
PARTS = ("input_layernorm", "q_norm", "k_norm", "conv", "gated_norm", "gates", "swiglu")
SETTING = {"input_layernorm": "RMSNORM", "q_norm": "RMSNORM", "k_norm": "RMSNORM", "conv": "CONV", "gated_norm": "GATED_NORM",
           "gates": "GATES", "swiglu": "SWIGLU"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parity():
    path = os.path.join(LOGS, "d4_parity.jsonl")
    if not os.path.exists(path):
        return None
    worst = {}
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n += 1
            words = r["case"].split()
            part = " ".join(words[:3]) if words[0] == "end" else words[0]
            dt = "bf16" if words[0] == "end" else next((w for w in words if w in ("f32", "bf16", "f16")), "?")
            for name, e in r["errors"].items():
                w = worst.setdefault(f"{part} {dt}", {"fused_worst": 0.0, "pytorch_worst": 0.0, "ratio_worst": 0.0,
                                                      "ratio_case": None})
                w["fused_worst"] = max(w["fused_worst"], e["fused"])
                w["pytorch_worst"] = max(w["pytorch_worst"], e["pytorch"])
                ratio = e["fused"] / e["pytorch"] if e["pytorch"] > 0 else float("inf")
                if ratio > w["ratio_worst"]:
                    w["ratio_worst"], w["ratio_case"] = ratio, f"{r['case']}: {name}"
    return {"cases": n, "by_part": worst, "met": all(w["ratio_worst"] <= BARS["parity_ratio"] for w in worst.values())}


def parts_table(rec):
    out = {}
    for model, res in rec["models"].items():
        rows = {}
        for part in PARTS:
            r = res.get(part)
            if r is None:
                continue
            row = {"pytorch_fwd_ms": r["pytorch"]["fwd_ms"], "pytorch_fwd_bwd_ms": r["pytorch"]["fwd_bwd_ms"],
                   "kernel_fwd_ms": r["kernel"]["fwd_ms"], "kernel_fwd_bwd_ms": r["kernel"]["fwd_bwd_ms"],
                   "speedup_fwd_bwd": r["speedup_fwd_bwd"], "floor_fwd_bwd_ms": r["floor_fwd_bwd_ms"],
                   "kernel_over_floor": r["kernel_over_floor"], "speed_bar_met": r["speedup_fwd_bwd"] >= BARS["speedup"],
                   "floor_prediction_met": (r["kernel_over_floor"] <= PRED["floor_ratio"]) if r["kernel_over_floor"] else None}
            if "pytorch_conv_only" in r:
                row["pytorch_conv_only_fwd_bwd_ms"] = r["pytorch_conv_only"]["fwd_bwd_ms"]
                row["speedup_fwd_bwd_against_conv_only"] = r["speedup_fwd_bwd_against_conv_only"]
            rows[part] = row
        for kind in ("linear", "full"):
            r = res.get(f"layer_{kind}")
            if r:
                rows[f"layer_{kind}"] = {"off_fwd_ms": r["off"]["fwd_ms"], "off_fwd_bwd_ms": r["off"]["fwd_bwd_ms"],
                                         "on_fwd_ms": r["on"]["fwd_ms"], "on_fwd_bwd_ms": r["on"]["fwd_bwd_ms"],
                                         "speedup_fwd_bwd": r["speedup_fwd_bwd"], "saved_fwd_bwd_ms": r["saved_fwd_bwd_ms"]}
        lo, hi = PRED["layer_08_ms"] if model == "0.8b" else PRED["layer_9_ms"]
        lin = rows.get("layer_linear")
        if lin:
            v = lin["on_fwd_bwd_ms"]
            lin["predicted_ms"] = [lo, hi]
            lin["prediction"] = "met" if lo <= v <= hi else ("missed (faster)" if v < lo else "missed (slower)")
        out[model] = rows
    # a setting is admitted when every part it covers meets the speed bar at both shapes
    admitted = {}
    for part, setting in SETTING.items():
        ok = [out[m][part]["speed_bar_met"] for m in out if part in out[m]]
        admitted.setdefault(setting, True)
        admitted[setting] = admitted[setting] and bool(ok) and all(ok)
    return out, admitted


def integration_08(p4, name=RUN_08):
    r = MR.read(LOGS, name)
    if not MR.ok(r):
        return None
    la, lb, lc = (MR.by_step(p4[k], "loss") for k in ("ref_a", "ref_b", "ref_c"))
    d2m = MR.read(*os.path.split(D2M_08))
    ld = MR.by_step(r, "loss")
    last = max(ld)
    ref_last = (la[last] + lb[last]) / 2
    ref_mean = (p4["ref_a"]["trainer"]["train_loss"] + p4["ref_b"]["trainer"]["train_loss"]) / 2
    st = r["s_per_step_steady_trainer_clock"]
    lo, hi = PRED["step_08_s"]
    fp = r["guard"]["peak_footprint_gb"]
    return {"layer": (r.get("macai") or {}).get("layer"), "footprint_gb": fp, "footprint_met": fp <= BARS["footprint_08_gb"],
            "d2m_footprint_gb": d2m["guard"]["peak_footprint_gb"], "steady_s": st, "d2m_steady_s": d2m["s_per_step_steady_trainer_clock"],
            "steady_over_d2m": round(st / d2m["s_per_step_steady_trainer_clock"], 4), "predicted_s": [lo, hi],
            "step_prediction": "met" if lo <= st <= hi else ("missed (faster)" if st < lo else "missed (slower)"),
            "rms_loss_difference": {"D4-A": MR.rms_pair(ld, la), "D4-B": MR.rms_pair(ld, lb), "D4-C": MR.rms_pair(ld, lc),
                                    "D4-D2m": MR.rms_pair(ld, MR.by_step(d2m, "loss")), "A-B": MR.rms_pair(la, lb),
                                    "A-C": MR.rms_pair(la, lc), "B-C": MR.rms_pair(lb, lc)},
            "final_loss": ld[last], "final_loss_reference_ab": ref_last, "final_loss_rel": abs(ld[last] - ref_last) / ref_last,
            "mean_loss": r["trainer"]["train_loss"], "mean_loss_reference_ab": ref_mean,
            "mean_loss_rel": abs(r["trainer"]["train_loss"] - ref_mean) / ref_mean,
            "d2m_final_loss": d2m["loss_final_step"], "d2m_mean_loss": d2m["trainer"]["train_loss"],
            "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"], "d2m_live_peak_gib": d2m["live_at_forward_end_peak_gb"],
            "driver_peak_gb": r["mps_driver_peak_gb"], "d2m_driver_peak_gb": d2m["mps_driver_peak_gb"],
            "steps": len(r["steps"]), "wall_s": r["wall_s"]}


def integration_9(name=RUN_9):
    r = MR.read(LOGS, name)
    if not MR.ok(r):
        return None
    d2m = MR.read(*os.path.split(D2M_9))
    st = MR.steady(r, 11, 25)
    lo, hi = PRED["step_9_s"]
    fp = r["guard"]["peak_footprint_gb"]
    ld, l2 = MR.by_step(r, "loss"), MR.by_step(d2m, "loss")
    return {"layer": (r.get("macai") or {}).get("layer"), "footprint_gb": fp, "footprint_met": fp <= BARS["footprint_9_gb"],
            "d2m_footprint_through_25_gb": MR.guard_peak_through(d2m["guard"], 25), "steady_11_25_s": st,
            "d2m_steady_11_25_s": MR.steady(d2m, 11, 25), "steady_over_d2m": round(st / MR.steady(d2m, 11, 25), 4),
            "predicted_s": [lo, hi], "step_prediction": "met" if lo <= st <= hi else ("missed (faster)" if st < lo else "missed (slower)"),
            "rms_loss_difference_d2m_steps_1_25": MR.rms_pair(ld, {s: v for s, v in l2.items() if s <= 25}),
            "loss_step_25": ld.get(25), "d2m_loss_step_25": l2.get(25),
            "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"],
            "d2m_live_peak_by_25_gib": MR.window_max(d2m, "live_at_forward_end_gb", 1, 25),
            "steps": len(r["steps"]), "wall_s": r["wall_s"]}


def reprofile(name=PROFILE):
    path = os.path.join(LOGS, name + ".json")
    if not os.path.exists(path):
        return None
    new = PR.report("0.8b", path, D2M_08 + ".json")
    old = PR.report("0.8b", os.path.join(LOGS, "profile_0.8b_2x8.json"), D2M_08 + ".json")
    groups = {}
    for g in set(new["groups"]) | set(old["groups"]):
        a, b = old["groups"].get(g, {}), new["groups"].get(g, {})
        groups[g] = {"step0_ms_per_micro_batch": a.get("ms_per_micro_batch"), "d4_ms_per_micro_batch": b.get("ms_per_micro_batch"),
                     "step0_share_pct": a.get("share_pct"), "d4_share_pct": b.get("share_pct")}
    comps = defaultdict(dict)
    for tag, rep in (("step0", old), ("d4", new)):
        for c in rep["components"]:
            if c["group"] in ("surroundings", "lora_extra") or "in_proj_qkv" in c["component"]:
                comps[c["component"]][tag] = {"fwd_ms": c["fwd_ms"], "bwd_ms": c["bwd_ms"]}
    return {"validity": new["validity"]["c_over_b_per_micro_batch"], "per_micro_batch_ms": [old["per_micro_batch_ms_total"],
            new["per_micro_batch_ms_total"]], "groups": groups, "components": dict(comps)}


def main():
    p4 = {k: MR.read(MR.P4, v) for k, v in MR.PHASE4.items()}
    out = {"bars": BARS, "predictions": PRED, "parity": parity()}
    parts = load(os.path.join(LOGS, "d4_parts.json"))
    if parts:
        out["parts"], out["admitted"] = parts_table(parts)
        out["in_proj_qkv_lora_layout_0.8b_2x903"] = parts.get("in_proj_qkv_lora_layout_0.8b_2x903")
        out["bench_layer_on"] = parts.get("layer_on")
    out["integration_0.8b"] = integration_08(p4)
    out["integration_9b"] = integration_9()
    out["reprofile_0.8b"] = reprofile()
    out["first_chain_norms_and_swiglu_only"] = {"integration_0.8b": integration_08(p4, FIRST["run_08"]),
                                                "integration_9b": integration_9(FIRST["run_9"]),
                                                "reprofile_0.8b": reprofile(FIRST["profile"])}
    print(json.dumps(out, indent=1, default=str))
    with open(os.path.join(LOGS, "d4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", os.path.join(LOGS, "d4_results.json"))


if __name__ == "__main__":
    main()
