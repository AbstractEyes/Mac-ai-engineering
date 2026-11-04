"""Series design D5 (the products; the LoRA adapter path fused), docs/series_layer.md: every figure and verdict from the records,
graded against the registered bars and predictions (series journal, 2026-11-03 11:2x PST, row 2) and the design's own
predictions written before any code (the design log, "D5 (b): the design, and its predictions").

    python bench/d5_report.py          # writes bench/layer_logs/d5_results.json (offline by itself)

Reads: step 0's products (products_torch.json, products_mlx.json) and the LM head's products at the labelled row counts
(products_head_rows_torch.json, _mlx.json); the adapter path's bench (d5_adapter.json, bench/bench_lora.py); the test run's parity
log (d5_parity.jsonl, tests/test_lora.py with MACAI_PARITY_LOG); the probes in real training runs (probe_d5_0.8b / _9b.json,
bench/lora_probe.py); the runs (bench/layer_logs/<run>.json with the guard's <run>.log) against D8 and D9's runs of record (the
same batches: the data order is seeded) and Phase 4's reference runs (bench/phase4_logs); the step profiles of both models with D4
+ D8 and with D5 on as well (bench/profile_report.py's reading). Step times are on the trainer's clock (whole seconds), as D4's and D8 / D9's windows.
"""
import json
import os
import statistics

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the profile reading looks up each model's config in the Hub cache: never online
import layer_report as LR  # noqa: E402
import memory_report as MR  # noqa: E402
import profile_report as PR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "layer_logs")
BASE_08, BASE_9 = "d8d9_0.8b_2x8_100steps", "d8d9_9b_1x16_50steps"
RUN_08, RUN_9 = "d5_0.8b_2x8_100steps", "d5_9b_1x16_25steps"
# the step profiles, D4 + D8 (before) and D4 + D8 + D5, by model; with the D2m run whose steps they are compared with
PROFILES = {"0.8b": ("profile_d8_0.8b_2x8", "profile_d5_0.8b_2x8", "confirm_recompute_0.8b_2x8_100steps.json"),
            "9b": ("profile_d8_9b_1x16", "profile_d5_9b_1x16", "confirm_recompute_9b_1x16_100steps.json")}
PROBES = {"0.8b": "probe_d5_0.8b_run2", "9b": "probe_d5_9b"}   # probe_d5_0.8b: the first run, stopped before the model's load
BENCH, BENCH_RUN1 = "d5_adapter", "d5_adapter_run1"          # run 1: the layer less its base only (in the chain of record)
WINDOWS = ((1, 10), (11, 25), (26, 50), (51, 75), (76, 100), (11, 100))
BARS = {"grad_ratio": 2.0, "adapter_speedup": 1.2, "footprint_08_gb": 53.5, "footprint_9_gb": 74.6,
        "custom_product_share_pct": 5.0, "custom_product_of_mlx": 0.80}
REGISTERED = {"frozen_of_rate_bench_min": 0.85, "frozen_of_rate_micro_min": 0.70, "mlx_within": 0.10,
              "rank32_of_rate_max": 0.30, "adapter_speedup_min": 1.5}
DESIGN = {"adapter_ms": {"0.8b": (60, 85), "9b": (105, 150)}, "adapter_speedup_min": 1.8,
          "steady_08_s": (5.8, 6.0), "later_08_s": 5.6, "steps_11_25_9_s": (40.7, 41.4)}
HEAD_ROWS = {"0.8b": 312, "9b": 163}            # the labelled rows of the step profiles' average micro-batch
STEP0_LORA_9B_MS = 256.1                        # step 0's 9B profile: the adapters' extra work a micro-batch
SMALL = ("in_proj_a / in_proj_b",)              # 16-32 outputs wide: not among the big projections


def load(name):
    path = os.path.join(LOGS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def window(r, lo, hi):
    t = MR.trainer_clock(r)
    t[0] = 0
    if hi not in t or lo - 1 not in t:
        return None
    return round((t[hi] - t[lo - 1]) / (hi - lo + 1), 3)


def windows(r, top):
    return {f"{lo}-{hi}": window(r, lo, hi) for lo, hi in WINDOWS if hi <= top}


def grade(value, lo, hi):
    return "met" if lo <= value <= hi else ("missed (faster)" if value < lo else "missed (slower)")


# ---- (a) the products ---------------------------------------------------------------------------------------------------

def products_a():
    t, m = load("products_torch"), load("products_mlx")
    out = {"large_square_tflops": {"pytorch": t["large_square_tflops"], "mlx": m["large_square_tflops"]}, "shapes": {}}
    ref = t["large_square_tflops"]
    for model in t["models"]:
        for i, shape in enumerate(t["models"][model]):
            rows = [(a, b) for a, b in zip(t["models"][model][shape], m["models"][model][shape])
                    if a["product"].startswith("base") and a["projection"] not in SMALL + ("lm_head",) and a["layout"] == "plain"]
            of_rate = [a["tflops_events"] / ref for a, _ in rows]
            vs_mlx = [b["wall_ms"] / a["wall_ms"] for a, b in rows]           # PyTorch's rate over MLX's (wall clocks)
            kind = "bench" if i == 0 else "micro_batch"
            bar = REGISTERED["frozen_of_rate_bench_min"] if kind == "bench" else REGISTERED["frozen_of_rate_micro_min"]
            low = min(zip(of_rate, rows), key=lambda z: z[0])
            out["shapes"][f"{model} {shape}"] = {
                "kind": kind, "products": len(rows), "of_large_square": [round(min(of_rate), 3), round(max(of_rate), 3)],
                "of_large_square_median": round(statistics.median(of_rate), 3),
                "lowest": f"{low[1][0]['projection']} {low[1][0]['product']}",
                "under_registered": [f"{a['projection']} {a['product']} {x:.2f}" for x, (a, _) in zip(of_rate, rows) if x < bar],
                "registered_min": bar, "met": min(of_rate) >= bar,
                "pytorch_over_mlx": [round(min(vs_mlx), 3), round(max(vs_mlx), 3)],
                "pytorch_over_mlx_median": round(statistics.median(vs_mlx), 3),
                "mlx_within_10pct": all(abs(1 / v - 1) <= REGISTERED["mlx_within"] for v in vs_mlx),
                "outside_10pct": [f"{a['projection']} {a['product']} {v:.2f}" for v, (a, _) in zip(vs_mlx, rows)
                                  if abs(1 / v - 1) > REGISTERED["mlx_within"]]}
    th, mh = load("products_head_rows_torch"), load("products_head_rows_mlx")
    head = {}
    for model in th["models"]:
        for a, b in zip(th["models"][model], mh["models"][model]):
            head.setdefault(model, {})[f"{a['rows']} {a['product']}"] = {
                "rows": a["rows"], "pytorch_ms": a["events_ms"], "pytorch_tflops": a["tflops_events"], "mlx_ms": b["wall_ms"],
                "mlx_tflops": b["tflops_wall"], "pytorch_over_mlx": round(b["wall_ms"] / a["wall_ms"], 3)}
    out["head_rows"] = head
    out["head_rows_squares"] = {"pytorch": [th["squares"], th["squares_end"]], "mlx": [mh["squares"], mh["squares_end"]]}
    return out


def head_trigger(prof_before, prof_after, a, model="0.8b"):
    """(c)'s rule at the head's products: their share of the micro-batch (in situ, the step profiles) and PyTorch over MLX at
    the labelled rows."""
    out = {}
    for tag, rep in (("d4_d8", prof_before), ("d5", prof_after)):
        if rep is None:
            continue
        c = {x["component"]: x for x in rep["components"]}
        h = c.get("top.lm_head")
        mb = rep["per_micro_batch_ms_total"]
        out[tag] = {"micro_batch_ms": mb, "head_fwd_ms": h["fwd_ms"], "head_bwd_ms": h["bwd_ms"],
                    "head_fwd_share_pct": round(100 * h["fwd_ms"] / mb, 2), "head_share_pct": round(100 * h["ms"] / mb, 2)}
    rows = a["head_rows"]
    out["rule"] = {}
    for m, n in HEAD_ROWS.items():
        f, b = rows[m][f"{n} forward Z = E W^T"], rows[m][f"{n} backward dE = dZ W"]
        both = round((f["mlx_ms"] + b["mlx_ms"]) / (f["pytorch_ms"] + b["pytorch_ms"]), 3)
        out["rule"][m] = {"rows": n, "forward_pytorch_over_mlx": f["pytorch_over_mlx"],
                              "backward_pytorch_over_mlx": b["pytorch_over_mlx"], "both_pytorch_over_mlx": both}
    share = out.get("d5", out.get("d4_d8", {})).get("head_fwd_share_pct")
    fwd_ratio = out["rule"][model]["forward_pytorch_over_mlx"]
    out["model"] = model
    out["triggered"] = bool(share is not None and share >= BARS["custom_product_share_pct"]
                            and fwd_ratio < BARS["custom_product_of_mlx"])
    return out


# ---- (b) the adapter path -----------------------------------------------------------------------------------------------

def bench_b(name=BENCH):
    r = load(name)
    if r is None:
        return None
    out = {"record": name, "started": r["started"], "rounds": r["rounds"], "squares": [r["squares_start"], r["squares_end"]],
           "models": {}}
    ref = max(list(r["squares_start"].values()) + list(r["squares_end"].values()))
    for model, res in r["models"].items():
        tot = res["per_micro_batch"]
        rates = []
        kern = []
        for shape, s in res["shapes"].items():
            for k, p in s["products"].items():
                rates.append((p["tflops"] / ref, f"{shape} {k}"))
            kern.append({"shape": shape, "up_ms": s["parts_ms"]["up kernel (R + s P B^T)"],
                         "up_floor_ms": s["floors_ms"]["up kernel"],
                         "down_ms": s["parts_ms"]["down kernel (M (G A) / (1 - p))"], "down_floor_ms": s["floors_ms"]["down kernel"],
                         "dropout_ms": s["parts_ms"]["dropout (torch.native_dropout)"],
                         "adapter_ms": [s["peft_adapter_fwd_bwd_ms"], s["fused_adapter_fwd_bwd_ms"]],
                         "speedup": s["adapter_speedup_fwd_bwd"]})
        lo, hi = DESIGN["adapter_ms"][model]
        sp = tot["adapter_speedup_fwd_bwd"]
        part = None
        if "part_speedup_fwd_bwd" in tot:
            psp = tot["part_speedup_fwd_bwd"]
            part = {"ms": [tot["part_peft_fwd_bwd_ms"], tot["part_fused_fwd_bwd_ms"]],
                    "fwd_ms": [tot["part_peft_fwd_ms"], tot["part_fused_fwd_ms"]], "speedup": psp,
                    "speedup_fwd": tot["part_speedup_fwd"], "bar_met": psp >= BARS["adapter_speedup"],
                    "registered_prediction_met": psp >= REGISTERED["adapter_speedup_min"],
                    "design_prediction_met": psp >= DESIGN["adapter_speedup_min"],
                    "by_shape": {sh: {"ms": [s["part_alone"]["peft"]["fwd_bwd_ms"], s["part_alone"]["fused"]["fwd_bwd_ms"]],
                                      "speedup": s["part_alone"]["speedup_fwd_bwd"]} for sh, s in res["shapes"].items()}}
        out["models"][model] = {
            "part_alone": part,
            "micro_batch": res["micro_batch"], "adapter_ms": [tot["peft_adapter_fwd_bwd_ms"], tot["fused_adapter_fwd_bwd_ms"]],
            "adapter_fwd_ms": [tot["peft_adapter_fwd_ms"], tot["fused_adapter_fwd_ms"]],
            "speedup": sp, "speedup_fwd": tot["adapter_speedup_fwd"], "bar": BARS["adapter_speedup"],
            "bar_met": sp >= BARS["adapter_speedup"],
            "registered_prediction": REGISTERED["adapter_speedup_min"], "registered_prediction_met": sp >= REGISTERED["adapter_speedup_min"],
            "design_prediction": DESIGN["adapter_speedup_min"], "design_prediction_met": sp >= DESIGN["adapter_speedup_min"],
            "design_adapter_ms": [lo, hi], "design_adapter_ms_grade_isolated": grade(tot["fused_adapter_fwd_bwd_ms"], lo, hi),
            "by_projection": [{k: p[k] for k in ("projection", "calls", "shape", "peft_adapter_fwd_bwd_ms", "fused_adapter_fwd_bwd_ms",
                                                 "adapter_speedup_fwd_bwd")} for p in res["projections"]],
            "speedup_range_by_shape": [min(k["speedup"] for k in kern), max(k["speedup"] for k in kern)],
            "kernels": kern,
            "rank32_products_of_large_square": {"min": round(min(rates)[0], 3), "max": round(max(rates)[0], 3),
                                                "median": round(statistics.median(x for x, _ in rates), 3),
                                                "lowest": min(rates)[1], "highest": max(rates)[1],
                                                "registered_max": REGISTERED["rank32_of_rate_max"],
                                                "grade": "met" if statistics.median(x for x, _ in rates) <= REGISTERED["rank32_of_rate_max"]
                                                else "missed (higher rate)"}}
    # step 0's own reading of the rank-32 products (products_torch.json), against its large-square rate
    t = load("products_torch")
    s0 = {}
    for model in t["models"]:
        for shape, rows in t["models"][model].items():
            f = [x["of_large_square"] for x in rows if x["product"].startswith("lora") and x["layout"] == "plain"]
            s0[f"{model} {shape}"] = {"min": min(f), "max": max(f), "median": round(statistics.median(f), 3)}
    out["rank32_step0"] = s0
    return out


def tests_parity():
    path = os.path.join(LOGS, "d5_parity.jsonl")
    if not os.path.exists(path):
        return None
    groups = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["case"].startswith("lora "):
                continue                        # the other parts' cases in the same run
            c = r["case"]
            g = "end to end" if "end to end" in c else ("no mask" if "no mask" in c else
                                                        next(t for t in c.split() if t.startswith("x-")) + (" base zero" if "base zero" in c else ""))
            for k, v in r["errors"].items():
                ratio = v["fused"] / v["pytorch"] if v["pytorch"] > 0 else (1.0 if v["fused"] == 0 else float("inf"))
                e = groups.setdefault(g, {"cases": set(), "tensors": 0, "worst_ratio": 0.0, "worst": None, "ratios": [],
                                          "peft_error_range": [float("inf"), 0.0]})
                e["cases"].add(c)
                e["tensors"] += 1
                e["ratios"].append(ratio)
                e["peft_error_range"] = [min(e["peft_error_range"][0], v["pytorch"]), max(e["peft_error_range"][1], v["pytorch"])]
                if ratio > e["worst_ratio"]:
                    e["worst_ratio"], e["worst"] = ratio, f"{c}: {k} ({v['pytorch']:.3g} / {v['fused']:.3g})"
    out = {}
    for g, e in groups.items():
        rs = sorted(e["ratios"])
        out[g] = {"cases": len(e["cases"]), "tensors": e["tensors"], "worst_ratio": round(e["worst_ratio"], 3), "worst": e["worst"],
                  "median_ratio": round(rs[len(rs) // 2], 3), "at_or_under_1": sum(x <= 1.0 for x in rs),
                  "peft_error_range": e["peft_error_range"], "met": e["worst_ratio"] <= BARS["grad_ratio"]}
    return out


def probe(model):
    r = load(PROBES[model])
    if r is None or r.get("error"):
        return None if r is None else {"error": r.get("error")}
    P = r["probe"]
    by = P["fused_by_module"]
    steps = r["steps"]
    n_mb = len(steps) * r["grad_accum"]
    last = steps[-1].get("training_counts", {})
    calls = sorted(set(by.values()))
    f64 = None
    for c in P["checks"]:
        f = c.get("float64_model")
        if f:
            ratios = sorted(v["fused"] / v["peft"] for k, v in f["errors"].items() if k != "loss" and v["peft"] > 0)
            pe = sorted(v["peft"] for k, v in f["errors"].items() if k != "loss")
            fe = sorted(v["fused"] for k, v in f["errors"].items() if k != "loss")
            f64 = {"gradients": len(ratios), "ratio_range": [round(ratios[0], 3), round(ratios[-1], 3)],
                   "ratio_median": round(statistics.median(ratios), 3),
                   "ratio_geomean": round(statistics.geometric_mean(ratios), 3),
                   "over_1": sum(x > 1 for x in ratios), "over_1_5": sum(x > 1.5 for x in ratios), "over_2": f["over_2"],
                   "worst": f["worst_ratio"], "worst_errors": f["errors"].get(f["worst_ratio"]["name"]),
                   "peft_error_median": round(statistics.median(pe), 4), "fused_error_median": round(statistics.median(fe), 4),
                   "peft_error_range": f["peft_error_range"], "loss": f["errors"]["loss"],
                   "input_embeddings": f["errors"]["d input embeddings"], "seconds": f["seconds"]}
    return {"micro_batches": n_mb, "adapted_modules": P["adapted_modules"], "modules_fused": len(by), "float64_model": f64,
            "calls_per_module": calls, "every_module_every_micro_batch": len(by) == P["adapted_modules"] and calls == [n_mb],
            "lora_fused_total": last.get("lora_fused"), "lora_fallbacks": last.get("lora_fallbacks"),
            "checks": P["checks"], "lora_dropout_modules": r.get("checks", {}).get("lora_dropout_modules"),
            "lora_forward_module": r.get("checks", {}).get("lora_linear_forward_at_train_begin"),
            "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty")}


# ---- the integration ----------------------------------------------------------------------------------------------------

def integration_08(p4):
    r, b = MR.read(LOGS, RUN_08), MR.read(LOGS, BASE_08)
    if not MR.ok(r):
        return None
    w, wb = windows(r, 100), windows(b, 100)
    lo, hi = DESIGN["steady_08_s"]
    lr, lb = MR.by_step(r, "loss"), MR.by_step(b, "loss")
    c = r["steps"][-1].get("training_counts", {})
    return {"windows_s": w, "baseline_windows_s": wb, "ratios": {k: round(w[k] / wb[k], 4) for k in w},
            "steady_s": r["s_per_step_steady_trainer_clock"], "baseline_steady_s": b["s_per_step_steady_trainer_clock"],
            "steady_ratio": round(r["s_per_step_steady_trainer_clock"] / b["s_per_step_steady_trainer_clock"], 4),
            "later_26_100_s": window(r, 26, 100), "baseline_later_26_100_s": window(b, 26, 100),
            "design_prediction_steady_s": [lo, hi], "design_prediction_grade": grade(r["s_per_step_steady_trainer_clock"], lo, hi),
            "footprint_gb": r["guard"]["peak_footprint_gb"], "footprint_met": r["guard"]["peak_footprint_gb"] <= BARS["footprint_08_gb"],
            "baseline_footprint_gb": b["guard"]["peak_footprint_gb"],
            "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"], "baseline_live_peak_gib": b["live_at_forward_end_peak_gb"],
            "driver_peak_gb": r["mps_driver_peak_gb"], "baseline_driver_peak_gb": b["mps_driver_peak_gb"],
            "lora_fused": c.get("lora_fused"), "lora_fallbacks": c.get("lora_fallbacks"), "micro_batches": 100 * r["grad_accum"],
            "head_taken": c.get("head_taken"), "vision_cache_served": c.get("vision_cache", {}).get("served_calls"),
            "loss": {"phase4_readings": LR.phase4_readings(p4, r), "baseline_phase4_readings": LR.phase4_readings(p4, b),
                     "rms_loss_difference": {"vs A": MR.rms_pair(lr, MR.by_step(p4["ref_a"], "loss")),
                                             "vs B": MR.rms_pair(lr, MR.by_step(p4["ref_b"], "loss")),
                                             "vs C": MR.rms_pair(lr, MR.by_step(p4["ref_c"], "loss")),
                                             "vs D8 + D9": MR.rms_pair(lr, lb),
                                             "A-B": MR.rms_pair(MR.by_step(p4["ref_a"], "loss"), MR.by_step(p4["ref_b"], "loss"))},
                     "final_loss": r["loss_final_step"], "mean_loss": r["trainer"]["train_loss"],
                     "baseline_final_loss": b["loss_final_step"], "baseline_mean_loss": b["trainer"]["train_loss"]},
            "steps_1_2": {s: {"run": lr[s], "baseline": lb[s], "same_bits": lr[s] == lb[s]} for s in (1, 2)},
            "macai": r.get("macai", {}).get("lora"), "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"),
            "wall_s": r["wall_s"]}


def integration_9(p4):
    r, b = MR.read(LOGS, RUN_9), MR.read(LOGS, BASE_9)
    if not MR.ok(r):
        return None
    w, wb = windows(r, 25), windows(b, 25)
    lo, hi = DESIGN["steps_11_25_9_s"]
    lr, lb, lp = MR.by_step(r, "loss"), MR.by_step(b, "loss"), MR.by_step(p4["ref_9b"], "loss")
    c = r["steps"][-1].get("training_counts", {})
    cb = MR.by_step(b, "training_counts")
    served_b = cb[25].get("vision_cache", {}).get("served_calls") if 25 in cb else None
    return {"windows_s": w, "baseline_windows_s": wb, "ratio_11_25": round(w["11-25"] / wb["11-25"], 4),
            "ratio_1_10": round(w["1-10"] / wb["1-10"], 4), "steady_s": r["s_per_step_steady_trainer_clock"],
            "design_prediction_11_25_s": [lo, hi], "design_prediction_grade": grade(w["11-25"], lo, hi),
            "footprint_gb": r["guard"]["peak_footprint_gb"], "footprint_met": r["guard"]["peak_footprint_gb"] <= BARS["footprint_9_gb"],
            "baseline_footprint_through_25_gb": MR.guard_peak_through(b["guard"], 25), "baseline_footprint_50_gb": b["guard"]["peak_footprint_gb"],
            "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"],
            "baseline_live_peak_through_25_gib": MR.window_max(b, "live_at_forward_end_gb", 1, 25),
            "driver_peak_gb": r["mps_driver_peak_gb"], "baseline_driver_peak_through_25_gb": MR.window_max(b, "mps_driver_gb", 1, 25),
            "lora_fused": c.get("lora_fused"), "lora_fallbacks": c.get("lora_fallbacks"), "micro_batches": 25 * r["grad_accum"],
            "vision_cache_served": c.get("vision_cache", {}).get("served_calls"), "baseline_served_through_25": served_b,
            "loss_rms_vs_phase4_9b_reference_1_25": MR.rms_pair(lr, lp),
            "baseline_loss_rms_vs_phase4_9b_reference_1_2": MR.rms_pair({s: lb[s] for s in (1, 2)}, lp),
            "steps_1_2": {s: {"run": lr[s], "baseline": lb[s], "same_bits": lr[s] == lb[s], "phase4_9b_reference": lp.get(s)}
                          for s in (1, 2)},
            "final_loss": r["loss_final_step"], "phase4_9b_reference_final": lp.get(25),
            "macai": r.get("macai", {}).get("lora"), "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"),
            "wall_s": r["wall_s"]}


def profile(model):
    b_name, a_name, d2m_name = PROFILES[model]
    path, before = os.path.join(LOGS, a_name + ".json"), os.path.join(LOGS, b_name + ".json")
    if not os.path.exists(before):
        return None, None, None
    d2m = os.path.join(HERE, "memory_logs", d2m_name)
    old = PR.report(model, before, d2m)
    new = PR.report(model, path, d2m) if os.path.exists(path) else None

    def lora_rows(rep):
        kinds, projs = {}, {}
        for c in rep["components"]:
            if c["group"] != "lora_extra":
                continue
            proj, kind = c["component"].rsplit(".", 1)
            kinds[kind] = round(kinds.get(kind, 0) + c["ms"], 2)
            projs[proj] = round(projs.get(proj, 0) + c["ms"], 2)
        return kinds, projs
    out = {"records": [b_name, a_name], "before": {"micro_batch_ms": old["per_micro_batch_ms_total"],
                      "lora_ms": old["groups"]["lora_extra"]["ms_per_micro_batch"],
                      "lora_share_pct": old["groups"]["lora_extra"]["share_pct"]}}
    out["before"]["lora_by_kind"], out["before"]["lora_by_projection"] = lora_rows(old)
    if new is not None:
        groups = {}
        for g in sorted(set(new["groups"]) | set(old["groups"]), key=lambda g: -(old["groups"].get(g, {}).get("ms_per_micro_batch") or 0)):
            a, b = old["groups"].get(g, {}), new["groups"].get(g, {})
            groups[g] = {"before_ms": a.get("ms_per_micro_batch"), "d5_ms": b.get("ms_per_micro_batch"),
                         "before_share_pct": a.get("share_pct"), "d5_share_pct": b.get("share_pct")}
        lo, hi = DESIGN["adapter_ms"][model]
        lora_ms = new["groups"]["lora_extra"]["ms_per_micro_batch"]
        out["d5"] = {"micro_batch_ms": new["per_micro_batch_ms_total"], "lora_ms": lora_ms,
                     "lora_share_pct": new["groups"]["lora_extra"]["share_pct"],
                     "validity": new["validity"]["c_over_b_per_micro_batch"], "events_span": new["validity"]["events_span_over_c_wall"],
                     "before_validity": old["validity"]["c_over_b_per_micro_batch"],
                     "groups": groups, "in_situ_speedup": round(out["before"]["lora_ms"] / lora_ms, 3),
                     "in_situ_bar_met": out["before"]["lora_ms"] / lora_ms >= BARS["adapter_speedup"],
                     "design_prediction_ms": [lo, hi], "design_prediction_grade": grade(lora_ms, lo, hi),
                     "repo_commit": new.get("repo_commit")}
        out["d5"]["lora_by_kind"], out["d5"]["lora_by_projection"] = lora_rows(new)
    return out, old, new


def main():
    p4 = {k: MR.read(MR.P4, v) for k, v in MR.PHASE4.items()}
    a = products_a()
    prof, old, new = profile("0.8b")
    prof9, old9, new9 = profile("9b")
    out = {"bars": BARS, "registered": REGISTERED, "design_predictions": DESIGN, "a_products": a,
           "c_head_trigger": head_trigger(old, new, a), "c_head_trigger_9b_in_situ": head_trigger(old9, new9, a, "9b"),
           "b_bench": bench_b(), "b_bench_run1": bench_b(BENCH_RUN1), "tests": tests_parity(),
           "probes": {m: probe(m) for m in PROBES}, "integration_0.8b": integration_08(p4), "integration_9b": integration_9(p4),
           "profile_0.8b": prof, "profile_9b": prof9}
    print(json.dumps(out, indent=1, default=str))
    with open(os.path.join(LOGS, "d5_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", os.path.join(LOGS, "d5_results.json"))


if __name__ == "__main__":
    main()
