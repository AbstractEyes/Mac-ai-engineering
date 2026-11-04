"""Series design D3 (the rule's chunking and tiling), docs/series_layer.md: every figure and verdict from the records, graded
against the registered bars and predictions (series journal, 2026-11-03 11:2x PST, row 3) and the design's own predictions written
before any code (the design log, "D3: where the rule's time is now, and the design's predictions").

    python bench/d3_report.py          # writes bench/layer_logs/d3_results.json (offline by itself)

Reads: the baselines before any build (d3_baseline.json) and the bench of record (d3_bench.json; bench/bench_d3.py), with the
exploration run (d3_bench_run2.json); the double buffering's forms (d3_dbuf_first_form.log, d3_dbuf_forms.log); the test run of
record (d3_tests.log) and its parity log (d3_parity.jsonl, tests/test_chunking.py with MACAI_PARITY_LOG); the probes in real
training runs (probe_d3_0.8b / _9b.json, bench/d3_probe.py); the integration runs (bench/layer_logs/<run>.json with the guard's
<run>.log) against D5's runs of record (the same batches: the data order is seeded) and Phase 4's reference runs; the step
profiles with D3 on against D5's (bench/profile_report.py's reading). Step times on the trainer's clock (whole seconds) and on
each step's own clock (the steps' t_s).
"""
import json
import os
import re
import statistics

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the profile reading looks up each model's config in the Hub cache: never online
import layer_report as LR  # noqa: E402
import memory_report as MR  # noqa: E402
import profile_report as PR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "layer_logs")
BASELINE, BENCH, RUN2 = "d3_baseline", "d3_bench", "d3_bench_run2"
BASE_08, BASE_9 = "d5_0.8b_2x8_100steps", "d5_9b_1x16_25steps"
RUN_08, RUN_9 = "d3_0.8b_2x8_100steps", "d3_9b_1x16_25steps"
PROFILES = {"0.8b": ("profile_d5_0.8b_2x8", "profile_d3_0.8b_2x8", "confirm_recompute_0.8b_2x8_100steps.json"),
            "9b": ("profile_d5_9b_1x16", "profile_d3_9b_1x16", "confirm_recompute_9b_1x16_100steps.json")}
PROBES = {"0.8b": "probe_d3_0.8b", "9b": "probe_d3_9b"}
MODES = ("fwd", "train_fwd", "fwd_bwd")
WINDOWS = ((1, 10), (11, 25), (26, 50), (51, 75), (76, 100), (11, 100), (26, 100))
BARS = {"part_speedup": 1.1, "slower_at_most": 0.02, "footprint_08_gb": 53.5, "footprint_9_gb": 74.6}
REGISTERED = {"a_together_ms_a_call_at_most": 1.0, "b_long_16_heads_forward_min": 1.5}
DESIGN = {"a1_train_fwd_bench": (1.05, 1.14), "a1_train_fwd_readers": (0.95, 1.05), "a2_state_passes": (1.05, 1.15),
          "a3_backward": (1.02, 1.08), "b_16_fwd": (1.6, 2.0), "b_16_fwd_bwd": (1.4, 1.8), "b_32": (1.0, 1.25),
          "b_readers": (0.95, 1.10), "a_ms_a_call_bench": (0.5, 1.5), "a_ms_a_call_readers": (0.2, 0.6),
          "long_16_fwd_ms_now": {4096: 6, 8192: 12, 16384: 24, 32768: 49}, "step_08_s": (5.91, 5.97), "step_9_s": (40.70, 40.85)}
# the harness's bars by case (tests/test_chunking.py CASES; the drop-in's by dtype)
HARNESS_BAR = (("shape", 2e-4), ("no_starting_state", 2e-4), ("stress", 5e-4), ("correlated", 5e-4), ("long_", 1e-3),
               ("dropin_float32", 2e-4), ("dropin_bfloat16", 2e-2))


def load(name):
    path = os.path.join(LOGS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def grade(value, lo, hi):
    return "met" if lo <= value <= hi else ("missed (faster)" if value > hi else "missed (slower)")


def ratio(a, b):
    return round(a / b, 4) if a and b else None


# ---- the bench ------------------------------------------------------------------------------------------------------------

def bench(name=BENCH):
    """Each variant against base: the rule a micro-batch, every shape and mode, the worst shape, the kernels alone."""
    r = load(name)
    if r is None:
        return None
    T, K = r["times"], r.get("kernels") or {}
    out = {"record": name, "repo_commit": r["repo_commit"], "repo_dirty": r["repo_dirty"], "rounds": r["rounds"], "reps": r["reps"],
           "started": r["started"], "wall_s": r["wall_s"], "variants": {}}
    for n in T:
        v = {"micro_batch_ms": r["micro_batch_ms"][n], "shapes_ms": {s: {m: T[n][s][m]["median_ms"] for m in MODES} for s in T[n]}}
        if n != "base":
            v["micro_batch_speedup"] = {m: {k: ratio(r["micro_batch_ms"]["base"][m][k], r["micro_batch_ms"][n][m][k]) for k in MODES}
                                        for m in r["micro_batch_ms"][n]}
            v["speedup"] = {s: {m: ratio(T["base"][s][m]["median_ms"], T[n][s][m]["median_ms"]) for m in MODES} for s in T[n]}
            worst = {}
            for m in MODES:
                w = min((v["speedup"][s][m], s) for s in T[n])
                worst[m] = {"speedup": w[0], "shape": w[1]}
            v["worst_shape"] = worst
            v["slower_than_2pct"] = sorted({s for s in T[n] for m in MODES if v["speedup"][s][m] < 1 - BARS["slower_at_most"]})
        if K:
            v["kernels_ms"] = {s: {p: x["median_ms"] for p, x in parts.items()} for s, parts in K[n].items()}
            if n != "base":
                v["kernels_speedup"] = {s: {p: ratio(K["base"][s][p]["median_ms"], K[n][s][p]["median_ms"]) for p in parts}
                                        for s, parts in K[n].items()}
        out["variants"][n] = v
    return out


def against(b, n, ref):
    """Variant n against variant ref of the same record (as each admitted item becomes the base of the next)."""
    if b is None or n not in b["variants"] or ref not in b["variants"]:
        return None
    a, c = b["variants"][n], b["variants"][ref]
    sp = {s: {m: ratio(c["shapes_ms"][s][m], a["shapes_ms"][s][m]) for m in MODES} for s in a["shapes_ms"]}
    mb = {mod: {m: ratio(c["micro_batch_ms"][mod][m], a["micro_batch_ms"][mod][m]) for m in MODES} for mod in a["micro_batch_ms"]}
    worst = {m: min((sp[s][m], s) for s in sp) for m in MODES}
    return {"micro_batch_speedup": mb, "speedup": sp, "worst_shape": {m: {"speedup": w[0], "shape": w[1]} for m, w in worst.items()},
            "slower_than_2pct": sorted({s for s in sp for m in MODES if sp[s][m] < 1 - BARS["slower_at_most"]})}


def long_rows(v, key="speedup"):
    return {s: v[key][s] for s in v[key] if s.startswith("long")}


def items(b):
    """Each item's bars: its part's speedup at its target shapes (>= 1.1x) and no shape slower by more than 2%."""
    if b is None:
        return None
    V = b["variants"]
    out = {}
    # (a1) chunk 8: its part is the training forward; its target shapes the readers' micro-batches
    a1 = V.get("fwd8")
    if a1:
        part = {mod: a1["micro_batch_speedup"][mod]["train_fwd"] for mod in a1["micro_batch_speedup"]}
        bench_part = {s: a1["speedup"][s]["train_fwd"] for s in ("0.8b_bench", "9b_bench")}
        out["a1_fwd8"] = {"part_readers": part, "part_bench_shapes": bench_part,
                          "part_bar_met": all(x >= BARS["part_speedup"] for x in part.values()),
                          "rule_micro_batch": a1["micro_batch_speedup"], "worst_shape": a1["worst_shape"],
                          "slower_than_2pct": a1["slower_than_2pct"], "no_shape_slower_met": not a1["slower_than_2pct"],
                          "design_grade_readers": {m: grade(x, *DESIGN["a1_train_fwd_readers"]) for m, x in part.items()},
                          "design_grade_bench": {s: grade(x, *DESIGN["a1_train_fwd_bench"]) for s, x in bench_part.items()}}
    # (a2) the reads ahead: its part is the state passes (the kernels alone)
    for n in ("dbuf", "dbuf2"):
        v = V.get(n)
        if v and v.get("kernels_speedup"):
            sp = {s: v["kernels_speedup"][s]["state_passes"] for s in v["kernels_speedup"]}
            out[f"a2_{n}"] = {"state_passes_speedup": sp, "part_bar_met": all(x >= BARS["part_speedup"] for x in sp.values()),
                              "rule_micro_batch": v["micro_batch_speedup"], "worst_shape": v["worst_shape"],
                              "no_shape_slower_met": not v["slower_than_2pct"],
                              "design_grade": {s: grade(x, *DESIGN["a2_state_passes"]) for s, x in sp.items()}}
    # (b) the split: its part is the forward at 1 row x 16 heads >= 4,096 tokens (and training there); read against base and,
    # with chunk 8 admitted before it, against fwd8
    for n, ref in (("split2", "base"), ("auto", "base"), ("fwd8+auto", "fwd8")):
        v = against(b, n, ref)
        if v is None:
            continue
        L = {s: x for s, x in v["speedup"].items() if s.startswith("long_h16") and int(s.split("_")[-1]) >= 4096}
        L32 = {s: x for s, x in v["speedup"].items() if s.startswith("long_h32")}
        fw = {s: x["fwd"] for s, x in L.items()}
        out[f"b_{n}_vs_{ref}"] = {"long_16_forward": fw, "long_16_fwd_bwd": {s: x["fwd_bwd"] for s, x in L.items()},
                                  "long_32": L32, "short_1_row": {s: x for s, x in v["speedup"].items() if s.endswith("_1024")},
                                  "readers_micro_batch": v["micro_batch_speedup"],
                                  "part_bar_met": all(x >= BARS["part_speedup"] for x in fw.values()),
                                  "registered_prediction_1_5x": all(x >= REGISTERED["b_long_16_heads_forward_min"] for x in fw.values()),
                                  "worst_shape": v["worst_shape"], "slower_than_2pct": v["slower_than_2pct"],
                                  "no_shape_slower_met": not v["slower_than_2pct"],
                                  "design_grade_16_fwd": {s: grade(x, *DESIGN["b_16_fwd"]) for s, x in fw.items()},
                                  "design_grade_16_fwd_bwd": {s: grade(x["fwd_bwd"], *DESIGN["b_16_fwd_bwd"]) for s, x in L.items()}}
    # (a) together: the ms a layer call the admitted (a) items remove at the bench shapes and a readers' micro-batch's call
    best = V.get("fwd8+auto") or V.get("fwd8")
    if best:
        base = V["base"]
        out["a_together_ms_a_call"] = {
            "variant": "fwd8+auto" if "fwd8+auto" in V else "fwd8",
            "bench_shapes": {s: round(base["shapes_ms"][s]["fwd_bwd"] - best["shapes_ms"][s]["fwd_bwd"], 3)
                             for s in ("0.8b_bench", "9b_bench")},
            "readers_a_call": {mod: round((base["micro_batch_ms"][mod]["fwd_bwd"] - best["micro_batch_ms"][mod]["fwd_bwd"])
                                          / (18 if mod == "0.8b" else 24), 3) for mod in base["micro_batch_ms"]}}
    return out


def dbuf_forms():
    """The double buffering's diagnostics: the state passes alone by form (the [diag] lines)."""
    out = {}
    for name in ("d3_dbuf_first_form", "d3_dbuf_forms"):
        path = os.path.join(LOGS, name + ".log")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"\[diag\] \((\d+), (\d+), (\d+)\) dbuf (\d) split (\d): (.*?)\s+\(", line)
                if m:
                    vals = dict(zip(m.group(6).split()[0::2], map(float, m.group(6).split()[1::2])))
                    rows.append({"shape": [int(m.group(i)) for i in (1, 2, 3)], "dbuf": int(m.group(4)), "split": int(m.group(5)),
                                 **vals})
        out[name] = rows
    return out


# ---- parity -------------------------------------------------------------------------------------------------------------------

def bar_of(case):
    return next(b for prefix, b in HARNESS_BAR if case.startswith(prefix))


def tests_parity():
    path = os.path.join(LOGS, "d3_parity.jsonl")
    log = os.path.join(LOGS, "d3_tests.log")
    out = {}
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        m = re.findall(r"(\d+) passed(?:, (\d+) \w+)*.*? in ([\d.]+)s", text)
        out["summary_line"] = next((ln for ln in text.splitlines() if " passed" in ln and " in " in ln), None)
        g = re.search(r"\[guard\] end \S+ rc (\S+) peak footprint ([\d.]+) GB wall (\d+) s", text)
        out["guard"] = {"rc": g.group(1), "peak_footprint_gb": float(g.group(2)), "wall_s": int(g.group(3))} if g else None
    if not os.path.exists(path):
        return out
    by = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("test") != "test_chunking":
                continue
            bar = bar_of(r["case"])
            e = by.setdefault(r["variant"], {"cases": 0, "tensors": 0, "worst_over_bar": 0.0, "worst": None,
                                             "forward_worst_over_bar": 0.0, "forward_worst": None, "base_worst_over_bar": 0.0,
                                             "over_base_max": 0.0, "over_base_max_where": None, "over_base_min": float("inf"),
                                             "over_base_min_where": None, "same_errors": 0})
            e["cases"] += 1
            for k, x in r["errors"].items():
                e["tensors"] += 1
                if x / bar > e["worst_over_bar"]:
                    e["worst_over_bar"], e["worst"] = x / bar, f"{r['case']}: {k} {x:.3g} (bar {bar:g})"
                if k in ("o", "sT") and x / bar > e["forward_worst_over_bar"]:
                    e["forward_worst_over_bar"], e["forward_worst"] = x / bar, f"{r['case']}: {k} {x:.3g} (bar {bar:g})"
                bx = r["base_errors"].get(k)
                e["base_worst_over_bar"] = max(e["base_worst_over_bar"], (bx or 0) / bar)
                e["same_errors"] += int(x == bx)
                if bx:
                    q = x / bx
                    if q > e["over_base_max"]:
                        e["over_base_max"], e["over_base_max_where"] = q, f"{r['case']}: {k} {x:.3g} / {bx:.3g}"
                    if q < e["over_base_min"]:
                        e["over_base_min"], e["over_base_min_where"] = q, f"{r['case']}: {k} {x:.3g} / {bx:.3g}"
    for e in by.values():
        for k in ("worst_over_bar", "forward_worst_over_bar", "base_worst_over_bar"):
            e[k] = round(e[k], 4)
        e["over_base_max"], e["over_base_min"] = round(e["over_base_max"], 3), round(e["over_base_min"], 3)
        e["met"] = e["worst_over_bar"] <= 1.0
    out["by_variant"] = by
    return out


def probe(model):
    r = load(PROBES[model])
    if r is None:
        return None
    P = r.get("probe") or {}
    last = (r.get("steps") or [{}])[-1].get("training_counts", {})
    return {"error": r.get("error"), "verdict": P.get("verdict"), "expected": P.get("expected"),
            "linear_attention_layers": P.get("linear_attention_layers"), "micro_batches": len(P.get("micro_batches", [])),
            "calls_by_path": last.get("rule_calls"), "lora_fused": last.get("lora_fused"), "head_taken": last.get("head_taken"),
            "macai": {k: (r.get("macai") or {}).get(k) for k in ("FWD_CHUNK", "VALUE_SPLIT", "DBUF", "RECOMPUTE_STATES")},
            "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"), "wall_s": r.get("wall_s")}


# ---- the integration -----------------------------------------------------------------------------------------------------

def own_clock(r):
    t = {s["step"]: s["t_s"] for s in r["steps"]}
    t[0] = 0.0
    return t


def windows(r, top, clock):
    t = clock(r)
    if clock is MR.trainer_clock:
        t[0] = 0
    return {f"{lo}-{hi}": round((t[hi] - t[lo - 1]) / (hi - lo + 1), 3) for lo, hi in WINDOWS
            if hi <= top and hi in t and lo - 1 in t}


def integration(model, p4):
    run, base = (RUN_08, BASE_08) if model == "0.8b" else (RUN_9, BASE_9)
    r, b = MR.read(LOGS, run), MR.read(LOGS, base)
    if not MR.ok(r):
        return None if r is None else {"error": r.get("error")}
    top = 100 if model == "0.8b" else 25
    wt, wbt = windows(r, top, MR.trainer_clock), windows(b, top, MR.trainer_clock)
    wo, wbo = windows(r, top, own_clock), windows(b, top, own_clock)
    lr, lb = MR.by_step(r, "loss"), MR.by_step(b, "loss")
    c = r["steps"][-1].get("training_counts", {})
    fp = r["guard"]["peak_footprint_gb"]
    out = {"windows_trainer_clock_s": wt, "baseline_windows_trainer_clock_s": wbt,
           "windows_own_clock_s": wo, "baseline_windows_own_clock_s": wbo,
           "ratios_own_clock": {k: round(wo[k] / wbo[k], 4) for k in wo if k in wbo},
           "steady_trainer_clock_s": r["s_per_step_steady_trainer_clock"], "baseline_steady_trainer_clock_s": b["s_per_step_steady_trainer_clock"],
           "footprint_gb": fp, "footprint_met": fp <= BARS[f"footprint_{'08' if model == '0.8b' else '9'}_gb"],
           "baseline_footprint_gb": b["guard"]["peak_footprint_gb"],
           "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"], "baseline_live_peak_gib": b["live_at_forward_end_peak_gb"],
           "driver_peak_gb": r["mps_driver_peak_gb"], "baseline_driver_peak_gb": b["mps_driver_peak_gb"],
           "rule_calls": c.get("rule_calls"), "lora_fused": c.get("lora_fused"), "head_taken": c.get("head_taken"),
           "vision_cache_served": (c.get("vision_cache") or {}).get("served_calls"),
           "steps_1_2": {s: {"run": lr.get(s), "baseline": lb.get(s), "same_bits": lr.get(s) == lb.get(s)} for s in (1, 2)},
           "macai": {k: (r.get("macai") or {}).get(k) for k in ("FWD_CHUNK", "VALUE_SPLIT", "DBUF", "RECOMPUTE_STATES")},
           "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"), "wall_s": r["wall_s"]}
    if model == "0.8b":
        lo, hi = DESIGN["step_08_s"]
        out["design_grade_steady_own_clock"] = grade(-wo["11-100"], -hi, -lo)
        out["loss"] = {"phase4_readings": LR.phase4_readings(p4, r), "baseline_phase4_readings": LR.phase4_readings(p4, b),
                       "rms_loss_difference": {"vs A": MR.rms_pair(lr, MR.by_step(p4["ref_a"], "loss")),
                                               "vs B": MR.rms_pair(lr, MR.by_step(p4["ref_b"], "loss")),
                                               "vs C": MR.rms_pair(lr, MR.by_step(p4["ref_c"], "loss")),
                                               "vs D5": MR.rms_pair(lr, lb),
                                               "A-B": MR.rms_pair(MR.by_step(p4["ref_a"], "loss"), MR.by_step(p4["ref_b"], "loss"))},
                       "final_loss": r["loss_final_step"], "mean_loss": r["trainer"]["train_loss"],
                       "baseline_final_loss": b["loss_final_step"], "baseline_mean_loss": b["trainer"]["train_loss"]}
    else:
        lo, hi = DESIGN["step_9_s"]
        out["design_grade_11_25_own_clock"] = grade(-wo["11-25"], -hi, -lo)
        lp = MR.by_step(p4["ref_9b"], "loss")
        out["loss"] = {"rms_vs_phase4_9b_reference_1_25": MR.rms_pair(lr, lp), "baseline_rms_vs_phase4_9b_reference_1_25": MR.rms_pair(lb, lp),
                       "final_loss": r["loss_final_step"], "baseline_final_loss": b["loss_final_step"], "phase4_9b_reference_final": lp.get(25)}
    return out


def profile(model):
    b_name, a_name, d2m_name = PROFILES[model]
    before, after = os.path.join(LOGS, b_name + ".json"), os.path.join(LOGS, a_name + ".json")
    if not os.path.exists(after):
        return None
    d2m = os.path.join(HERE, "memory_logs", d2m_name)
    old, new = PR.report(model, before, d2m), PR.report(model, after, d2m)
    groups = {g: {"d5_ms": old["groups"].get(g, {}).get("ms_per_micro_batch"), "d3_ms": new["groups"].get(g, {}).get("ms_per_micro_batch"),
                  "d5_share_pct": old["groups"].get(g, {}).get("share_pct"), "d3_share_pct": new["groups"].get(g, {}).get("share_pct")}
              for g in sorted(set(old["groups"]) | set(new["groups"]), key=lambda g: -(old["groups"].get(g, {}).get("ms_per_micro_batch") or 0))}

    def rule_rows(path):
        with open(path, encoding="utf-8") as fh:
            mbs = json.load(fh)["profile"]["micro_batches"]
        f = [m["rows"]["lin.rule|fwd"][0] for m in mbs]
        k = [m["rows"]["lin.rule|bwd"][0] for m in mbs]
        return {"forward_ms": round(statistics.mean(f), 2), "backward_ms": round(statistics.mean(k), 2), "micro_batches": len(mbs)}
    rb, ra = rule_rows(before), rule_rows(after)
    return {"records": [b_name, a_name], "micro_batch_ms": [old["per_micro_batch_ms_total"], new["per_micro_batch_ms_total"]],
            "validity": new["validity"]["c_over_b_per_micro_batch"], "events_span": new["validity"]["events_span_over_c_wall"],
            "rule_rows": {"d5": rb, "d3": ra, "forward_speedup": ratio(rb["forward_ms"], ra["forward_ms"]),
                          "rule_speedup": ratio(rb["forward_ms"] + rb["backward_ms"], ra["forward_ms"] + ra["backward_ms"])},
            "groups": groups}


def main():
    p4 = {k: MR.read(MR.P4, v) for k, v in MR.PHASE4.items()}
    b = bench()
    out = {"bars": BARS, "registered": REGISTERED, "design_predictions": DESIGN, "baseline": bench(BASELINE),
           "exploration": bench(RUN2), "bench": b, "items": items(b), "items_exploration": items(bench(RUN2)),
           "dbuf_forms": dbuf_forms(), "tests": tests_parity(), "probes": {m: probe(m) for m in PROBES},
           "integration_0.8b": integration("0.8b", p4), "integration_9b": integration("9b", p4),
           "profile_0.8b": profile("0.8b"), "profile_9b": profile("9b")}
    with open(os.path.join(LOGS, "d3_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    short = {k: out[k] for k in ("items", "tests", "probes", "integration_0.8b", "integration_9b")}
    print(json.dumps(short, indent=1, default=str)[:20000])
    print("wrote", os.path.join(LOGS, "d3_results.json"))


if __name__ == "__main__":
    main()
