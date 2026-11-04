"""Series designs D8 (the LM head and the loss at the labelled positions only) and D9 (the frozen vision tower's output cached
per picture), docs/series_layer.md: every figure and verdict from the records, graded against the registered bars and
predictions (series journal, 2026-11-03 13:1x PST).

    HF_HOME=<the Hub cache holding the models> python bench/d8d9_report.py      # writes bench/layer_logs/d8d9_results.json

Reads: the test run's parity log (bench/layer_logs/d8d9_parity.jsonl, tests/test_training.py with MACAI_PARITY_LOG); the probes
in real training runs (probe_d8d9_0.8b / _9b.json, bench/training_probe.py); the loss-head bench (d8_loss_head.json); the runs
(bench/layer_logs/<run>.json with the guard's <run>.log): D8 alone at 0.8B (1 epoch), the integration with both at 0.8B (100
steps) and 9B (50 steps), against D4's runs of record (the same batches: the data order is seeded) and, for the 9B epoch-2
baseline, D2m's 9B run's epoch-2 over epoch-1 ratio; Phase 4's reference runs A, B, C (bench/phase4_logs); the step profile with
D4 + D8 against D4's (bench/profile_report.py's reading). Step times are on the trainer's clock (whole seconds), as D4's windows.
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the profile reading looks up each model's config in the Hub cache: never online
import layer_report as LR  # noqa: E402
import memory_report as MR  # noqa: E402
import profile_report as PR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "layer_logs")
D4_08, D4_9 = "d4_0.8b_2x8_100steps", "d4_9b_1x16_25steps"
D2M_9 = os.path.join(HERE, "memory_logs", "confirm_recompute_9b_1x16_100steps")
D8_08, D8_9 = "d8_0.8b_2x8_25steps", "d8_9b_1x16_25steps"
D8_08_100 = "d8_0.8b_2x8_100steps"          # D8 alone over the integration's 4 epochs: D9's direct baseline at 0.8B
INT_08, INT_9 = "d8d9_0.8b_2x8_100steps", "d8d9_9b_1x16_50steps"
PROFILE, PROFILE_D4 = "profile_d8_0.8b_2x8", "profile_d4_0.8b_2x8"
PROBES = {"0.8b": "probe_d8d9_0.8b", "9b": "probe_d8d9_9b"}
WINDOWS = ((1, 10), (11, 25), (26, 50), (51, 75), (76, 100), (11, 100))
BARS = {"loss_rel": 1e-5, "grad_ratio": 2.0, "d8_step_08": 0.90, "d8_step_9": 0.95, "d9_later": 0.93,
        "footprint_08_gb": 53.5, "footprint_9_gb": 74.6}
PRED = {"head_loss_ms": {"0.8b": (50, 75), "9b": (70, 100)}, "d8_step": {"0.8b": (-0.18, -0.14), "9b": (-0.09, -0.06)},
        "d9_later": {"0.8b": (-0.11, -0.09), "9b": (-0.10, -0.08)}, "d9_first_epoch_max": 0.02,
        "cache_mb_a_picture": {"0.8b": 0.6, "9b": 2.3}, "lookup_ms_max": 15.0}
STEP0_HEAD_LOSS_MS = {"0.8b": 266.7, "9b": 347.6}        # step 0's profile (docs/series_layer.md)
STEP0_TOWER_MS = {"0.8b": 138.5, "9b": 317.5}


def load(name):
    path = os.path.join(LOGS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def window(r, lo, hi):
    """Seconds a step over steps lo..hi on the trainer's clock (D4's windows' clock)."""
    t = MR.trainer_clock(r)
    t[0] = 0
    if hi not in t or lo - 1 not in t:
        return None
    return round((t[hi] - t[lo - 1]) / (hi - lo + 1), 3)


def windows(r, top):
    return {f"{lo}-{hi}": window(r, lo, hi) for lo, hi in WINDOWS if hi <= top}


def grade(value, lo, hi):
    return "met" if lo <= value <= hi else ("missed (faster)" if value < lo else "missed (slower)")


def counts_at(r, step):
    for s in r["steps"]:
        if s["step"] == step:
            return s.get("training_counts", {})
    return {}


def cache_epochs(r, edges):
    """The vision cache's counts in each epoch (differences of the cumulative counts at the epochs' last steps)."""
    out, prev = [], {}
    for e, step in enumerate(edges, 1):
        v = counts_at(r, step).get("vision_cache", {})
        # the counters are cumulative: each epoch's own is the difference; the entries and the pending stores are states
        d = {k: (round(v[k] - prev.get(k, 0), 4) if k not in ("entries", "pending", "caches") else v[k]) for k in v}
        calls = d.get("calls") or 0
        d["epoch"] = e
        d["own_ms_per_call"] = round(1e3 * (d.get("key_s", 0) + d.get("serve_s", 0) + d.get("store_s", 0)) / calls, 2) if calls else None
        d["key_ms_per_call"] = round(1e3 * d.get("key_s", 0) / calls, 2) if calls else None
        d["serve_ms_per_served_call"] = (round(1e3 * d.get("serve_s", 0) / d["served_calls"], 2) if d.get("served_calls")
                                         else None)
        d["wait_ms_per_call"] = round(1e3 * d.get("wait_s", 0) / calls, 2) if calls else None
        out.append(d)
        prev = v
    last = counts_at(r, edges[-1]).get("vision_cache", {})
    size = {"entries": last.get("entries"), "pending": last.get("pending"), "bytes_stored": last.get("bytes_stored"),
            "mb_a_picture_in_full": round(last["bytes_stored"] / last["entries"] / 1e6, 3) if last.get("entries") else None,
            "mb_a_picture_merged": round(last["bytes_merged"] / last["entries"] / 1e6, 3) if last.get("entries") else None,
            "gb_total": round((last.get("bytes_stored") or 0) / 1e9, 3)}
    return out, size


def tests_parity():
    path = os.path.join(LOGS, "d8d9_parity.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "loss_rel_to_model_path" not in r:          # the layer parts' cases (tests/test_layer.py) in the same run
                continue
            rows = {k: v for k, v in r["errors"].items() if k != "loss"}
            ratios = {k: v["fused"] / v["pytorch"] for k, v in rows.items() if v["pytorch"] > 0}
            worst = max(ratios.items(), key=lambda kv: kv[1])
            out[r["case"]] = {"loss_rel_to_model_path": r["loss_rel_to_model_path"], "gradients": len(rows),
                              "worst_ratio": worst[1], "worst": worst[0],
                              "ratio_range": [min(ratios.values()), max(ratios.values())],
                              "model_path_error_range": [min(v["pytorch"] for v in rows.values()),
                                                         max(v["pytorch"] for v in rows.values())],
                              "loss_errors": r["errors"].get("loss"),
                              "met": r["loss_rel_to_model_path"] <= BARS["loss_rel"] and worst[1] <= BARS["grad_ratio"]}
    return out


def probe(model):
    r = load(PROBES[model])
    if r is None or r.get("error"):
        return None
    P = r["probe"]
    head = []
    for h in P["head"]:
        e = {k: h.get(k) for k in ("positions", "labelled", "loss_rel", "logits_labelled_same_bits", "final_hidden_same_bits",
                                    "head_alone_dh_err_off", "head_alone_dh_err_on", "dh_on_vs_off_max_rel",
                                    "dh_elements_differing", "dh_elements", "unlabelled_positions_zero_gradient",
                                    "lora_grads_n", "lora_grads_on_vs_off_max_rel_worst", "bar_met")}
        f = h.get("float64_model")
        if f:
            ratios = {k: v["head_at_labels"] / v["model_path"] for k, v in f["errors"].items() if k != "loss" and v["model_path"] > 0}
            e["float64_model"] = {"seconds": f["seconds"], "gradients": len(ratios), "worst": f["worst_ratio"],
                                  "ratio_range": [min(ratios.values()), max(ratios.values())],
                                  "model_path_error_range": [min(v["model_path"] for k, v in f["errors"].items() if k != "loss"),
                                                             max(v["model_path"] for k, v in f["errors"].items() if k != "loss")],
                                  "final_hidden": f["errors"]["d final hidden"], "input_embeddings": f["errors"]["d input embeddings"],
                                  "loss": f["errors"]["loss"], "over_2": f["over_2"]}
        head.append(e)
    tower = [{k: t.get(k) for k in ("pictures", "grids", "pixel_dtype", "requires_grad", "rng_cpu_unchanged", "rng_gpu_unchanged",
                                    "same_batch_twice_same_bits", "each_picture_alone_same_bits",
                                    "paired_with_previous_same_bits", "served_same_bits_as_recompute", "one_pixel_moved_served",
                                    "one_pixel_moved_computed", "other_grid", "other_grid_served", "served_call_ms",
                                    "key_ms_per_call", "serve_ms", "store_ms", "bytes_a_picture", "merged_bytes_a_picture",
                                    "bar_met")} for t in P["tower"]]
    steps = r["steps"]
    last = steps[-1].get("training_counts", {})
    n_mb = len(steps) * r["grad_accum"]
    return {"head": head, "tower": tower, "training": {"micro_batches": n_mb, "head_taken": last.get("head_taken"),
                                                       "head_fallback": last.get("head_fallback"),
                                                       "labelled_share": round(last["head_labelled"] / last["head_positions"], 4)
                                                       if last.get("head_positions") else None,
                                                       "fallbacks": last.get("fallbacks"),
                                                       "vision_cache_by_step": [
                                                           {"step": s["step"], **{k: s["training_counts"].get("vision_cache", {}).get(k)
                                                                                 for k in ("served_calls", "computed_calls",
                                                                                           "pictures_new", "pictures_served",
                                                                                           "entries")}} for s in steps]},
            "repo_commit": r.get("repo_commit")}


def loss_head():
    r = load("d8_loss_head")
    if r is None:
        return None
    out = {}
    for model, res in r["models"].items():
        a, b = res["every_position"]["ms"], res["labelled_only"]["ms"]
        out[model] = {"every_position_ms": a, "labelled_only_ms": b, "d8b_floor": res["d8b_floor"],
                      "whole_ratio": round(b["whole"] / a["whole"], 4)}
    return out


def labelled_share(name):
    """The share of the positions the head computes with D8 (the labelled ones), over a whole run: the fraction by which every
    vocabulary-wide tensor of the head and the loss shrinks."""
    r = load(name)
    if r is None or not r.get("steps"):
        return None
    c = r["steps"][-1].get("training_counts", {})
    return {"positions": c.get("head_positions"), "labelled": c.get("head_labelled"),
            "share": round(c["head_labelled"] / c["head_positions"], 4) if c.get("head_positions") else None}


def d8_alone_08(d4):
    r = MR.read(LOGS, D8_08)
    if not MR.ok(r):
        return None
    w, w4 = windows(r, 25), windows(d4, 25)
    l, l4 = MR.by_step(r, "loss"), MR.by_step(d4, "loss")
    ratio = round(w["11-25"] / w4["11-25"], 4)
    c = counts_at(r, 25)
    return {"windows_s": w, "d4_windows_s": w4, "ratio_11_25": ratio, "ratio_1_10": round(w["1-10"] / w4["1-10"], 4),
            "ratio_1_25": round(window(r, 1, 25) / window(d4, 1, 25), 4),
            "bar": BARS["d8_step_08"], "bar_met": ratio <= BARS["d8_step_08"],
            "prediction": [1 + PRED["d8_step"]["0.8b"][0], 1 + PRED["d8_step"]["0.8b"][1]],
            "prediction_grade": grade(ratio - 1, *PRED["d8_step"]["0.8b"]),
            "steps_1_2": {s: {"d8": l[s], "d4": l4[s], "rel": abs(l[s] - l4[s]) / l4[s]} for s in (1, 2)},
            "head_taken": c.get("head_taken"), "head_fallback": c.get("head_fallback"), "micro_batches": 25 * r["grad_accum"],
            "labelled_share": round(c["head_labelled"] / c["head_positions"], 4) if c.get("head_positions") else None,
            "footprint_gb": r["guard"]["peak_footprint_gb"], "d4_footprint_through_25_gb": MR.guard_peak_through(d4["guard"], 25),
            "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"],
            "d4_live_peak_through_25_gib": MR.window_max(d4, "live_at_forward_end_gb", 1, 25),
            "driver_peak_gb": r["mps_driver_peak_gb"], "d4_driver_peak_through_25_gb": MR.window_max(d4, "mps_driver_gb", 1, 25),
            "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"), "wall_s": r["wall_s"]}


def integration_08(d4, d8, p4):
    r = MR.read(LOGS, INT_08)
    if not MR.ok(r):
        return None
    w, w4 = windows(r, 100), windows(d4, 100)
    ratios = {k: round(w[k] / w4[k], 4) for k in w}
    later = round((window(r, 26, 100)) / window(d4, 26, 100), 4)
    first = ratios["11-25"]
    d9_gain = round(later / first, 4)
    epochs, size = cache_epochs(r, (25, 50, 75, 100))
    out = {"windows_s": w, "d4_windows_s": w4, "ratios": ratios, "ratio_later_26_100": later,
           "steady_s": r["s_per_step_steady_trainer_clock"], "d4_steady_s": d4["s_per_step_steady_trainer_clock"],
           "steady_ratio": round(r["s_per_step_steady_trainer_clock"] / d4["s_per_step_steady_trainer_clock"], 4),
           "d9_own_gain_later_over_first": d9_gain, "d9_bar": BARS["d9_later"], "d9_bar_met": d9_gain <= BARS["d9_later"],
           "d9_prediction": [1 + PRED["d9_later"]["0.8b"][0], 1 + PRED["d9_later"]["0.8b"][1]],
           "d9_prediction_grade": grade(d9_gain - 1, *PRED["d9_later"]["0.8b"]),
           "d9_by_epoch_over_first": {k: round(ratios[k] / first, 4) for k in ("26-50", "51-75", "76-100")},
           "cache_by_epoch": epochs, "cache_size": size,
           "footprint_gb": r["guard"]["peak_footprint_gb"], "footprint_met": r["guard"]["peak_footprint_gb"] <= BARS["footprint_08_gb"],
           "d4_footprint_gb": d4["guard"]["peak_footprint_gb"],
           "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"], "d4_live_peak_gib": d4["live_at_forward_end_peak_gb"],
           "driver_peak_gb": r["mps_driver_peak_gb"], "d4_driver_peak_gb": d4["mps_driver_peak_gb"],
           "loss": {"phase4_readings": LR.phase4_readings(p4, r), "d4_phase4_readings": LR.phase4_readings(p4, d4),
                    "rms_loss_difference": {"vs A": MR.rms_pair(MR.by_step(r, "loss"), MR.by_step(p4["ref_a"], "loss")),
                                            "vs B": MR.rms_pair(MR.by_step(r, "loss"), MR.by_step(p4["ref_b"], "loss")),
                                            "vs C": MR.rms_pair(MR.by_step(r, "loss"), MR.by_step(p4["ref_c"], "loss")),
                                            "vs D4": MR.rms_pair(MR.by_step(r, "loss"), MR.by_step(d4, "loss")),
                                            "A-B": MR.rms_pair(MR.by_step(p4["ref_a"], "loss"), MR.by_step(p4["ref_b"], "loss"))},
                    "final_loss": r["loss_final_step"], "mean_loss": r["trainer"]["train_loss"],
                    "d4_final_loss": d4["loss_final_step"], "d4_mean_loss": d4["trainer"]["train_loss"]},
           "steps_1_2": {s: {"run": MR.by_step(r, "loss")[s], "d4": MR.by_step(d4, "loss")[s]} for s in (1, 2)},
           "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"), "wall_s": r["wall_s"]}
    d8_100 = MR.read(LOGS, D8_08_100)
    if MR.ok(d8_100):
        w8 = windows(d8_100, 100)
        out["d8_alone_100_windows_s"] = w8
        out["d9_direct"] = {k: round(w[k] / w8[k], 4) for k in w}
        out["d9_direct"]["26-100"] = round(window(r, 26, 100) / window(d8_100, 26, 100), 4)
        out["d9_direct_bar_met"] = out["d9_direct"]["26-100"] <= BARS["d9_later"]
        out["d9_direct_prediction_grade"] = grade(out["d9_direct"]["26-100"] - 1, *PRED["d9_later"]["0.8b"])
        out["d8_alone_100_vs_d4"] = {k: round(w8[k] / w4[k], 4) for k in w8}
        out["d8_alone_repeat_1_25"] = {"d8_25steps": window(MR.read(LOGS, D8_08), 1, 25), "d8_100steps": window(d8_100, 1, 25),
                                       "d8_25steps_16_25": window(MR.read(LOGS, D8_08), 16, 25),
                                       "d8_100steps_16_25": window(d8_100, 16, 25), "integration_16_25": window(r, 16, 25)}
        out["d8_alone_100_footprint_gb"] = d8_100["guard"]["peak_footprint_gb"]
        out["d8_alone_100_live_peak_gib"] = d8_100["live_at_forward_end_peak_gb"]
        out["d8_alone_100_driver_peak_gb"] = d8_100["mps_driver_peak_gb"]
        out["d8_alone_100_steady_s"] = d8_100["s_per_step_steady_trainer_clock"]
        out["d8_alone_100_loss"] = {"phase4_readings": LR.phase4_readings(p4, d8_100), "final_loss": d8_100["loss_final_step"],
                                    "mean_loss": d8_100["trainer"]["train_loss"]}
    if d8 is not None:
        # the integration's first epoch against the first D8-alone run (25 steps; its steps 16-25 ran slow, see the repeat)
        out["first_epoch_vs_d8_alone_first_run"] = {k: round(w[k] / d8["windows_s"][k], 4) for k in ("1-10", "11-25")}
        out["first_epoch_vs_d8_alone_first_run"]["1-25"] = round(window(r, 1, 25) / window(MR.read(LOGS, D8_08), 1, 25), 4)
    if MR.ok(d8_100):
        # D9's first-epoch cost: the integration's first epoch against D8 alone over the same 100-step run's first epoch
        first = {k: round(w[k] / out["d8_alone_100_windows_s"][k], 4) for k in ("1-10", "11-25")}
        first["1-25"] = round(window(r, 1, 25) / window(d8_100, 1, 25), 4)
        out["d9_first_epoch_vs_d8_alone"] = first
        v = first["1-25"] - 1
        out["d9_first_epoch_prediction"] = {"max": PRED["d9_first_epoch_max"], "value": round(v, 4),
                                            "grade": "met" if v <= PRED["d9_first_epoch_max"] else "missed (slower)"}
    return out


def integration_9(d4, d2m, p4_9b):
    r = MR.read(LOGS, INT_9)
    if not MR.ok(r):
        return None
    w = windows(r, 50)
    w4 = windows(d4, 25)
    d2m_ratio = round(window(d2m, 26, 50) / window(d2m, 11, 25), 4)
    base_e2 = round(w4["11-25"] * d2m_ratio, 3)
    first = round(w["11-25"] / w4["11-25"], 4)
    later = round(w["26-50"] / base_e2, 4)
    epochs, size = cache_epochs(r, (25, 50))
    e1 = epochs[0]
    # D8 alone at 9B from the first epoch: the micro-batches served from the cache inside it (pictures that repeat in an epoch)
    # each saved about the tower's time (step 0's 317.5 ms a micro-batch) less the cache's own time, which every call pays
    served_11_25 = (counts_at(r, 25).get("vision_cache", {}).get("served_calls", 0)
                    - counts_at(r, 10).get("vision_cache", {}).get("served_calls", 0))
    calls_11_25 = (counts_at(r, 25).get("vision_cache", {}).get("calls", 0) - counts_at(r, 10).get("vision_cache", {}).get("calls", 0))
    own_11_25 = sum(counts_at(r, 25).get("vision_cache", {}).get(k, 0) - counts_at(r, 10).get("vision_cache", {}).get(k, 0)
                    for k in ("key_s", "serve_s", "store_s"))
    corrected = round((w["11-25"] * 15 + served_11_25 * STEP0_TOWER_MS["9b"] / 1e3 - own_11_25) / 15 / w4["11-25"], 4)
    d9_gain = round(later / first, 4)
    l, l4 = MR.by_step(r, "loss"), MR.by_step(d4, "loss")
    lp = MR.by_step(p4_9b, "loss")
    d8 = MR.read(LOGS, D8_9)
    d8_alone = None
    if MR.ok(d8):
        w8 = windows(d8, 25)
        ratio = round(w8["11-25"] / w4["11-25"], 4)
        l8 = MR.by_step(d8, "loss")
        d8_alone = {"windows_s": w8, "ratio_11_25": ratio, "ratio_1_10": round(w8["1-10"] / w4["1-10"], 4),
                    "bar_met": ratio <= BARS["d8_step_9"], "prediction_grade": grade(ratio - 1, *PRED["d8_step"]["9b"]),
                    "steps_1_2": {s: {"d8": l8[s], "d4": l4[s], "rel": abs(l8[s] - l4[s]) / l4[s]} for s in (1, 2)},
                    "loss_rms_vs_phase4_9b_reference_1_25": MR.rms_pair(l8, lp),
                    "d4_loss_rms_vs_phase4_9b_reference_1_25": MR.rms_pair(l4, lp),
                    "integration_epoch1_over_d8_alone": {k: round(w[k] / w8[k], 4) for k in ("1-10", "11-25")},
                    "footprint_gb": d8["guard"]["peak_footprint_gb"], "live_at_forward_end_peak_gib": d8["live_at_forward_end_peak_gb"],
                    "driver_peak_gb": d8["mps_driver_peak_gb"], "head_taken": counts_at(d8, 25).get("head_taken"),
                    "labelled_share": round(counts_at(d8, 25)["head_labelled"] / counts_at(d8, 25)["head_positions"], 4),
                    "repo_commit": d8.get("repo_commit"), "repo_dirty": d8.get("repo_dirty")}
    out = {"windows_s": w, "d4_windows_s": w4, "d2m_epoch2_over_epoch1": d2m_ratio, "baseline_epoch2_s": base_e2,
           "ratio_first_11_25": first, "ratio_later_26_50": later,
           "d8_first_epoch_served_micro_batches_11_25": served_11_25, "d8_first_epoch_calls_11_25": calls_11_25,
           "d8_first_epoch_cache_own_s_11_25": round(own_11_25, 3), "d8_ratio_corrected_for_served": corrected,
           "d8_bar": BARS["d8_step_9"], "d8_bar_met": first <= BARS["d8_step_9"] and corrected <= BARS["d8_step_9"],
           "d8_prediction": [1 + PRED["d8_step"]["9b"][0], 1 + PRED["d8_step"]["9b"][1]],
           "d8_prediction_grade": grade(corrected - 1, *PRED["d8_step"]["9b"]),
           "d9_own_gain_later_over_first": d9_gain, "d9_bar": BARS["d9_later"], "d9_bar_met": d9_gain <= BARS["d9_later"],
           "d9_prediction": [1 + PRED["d9_later"]["9b"][0], 1 + PRED["d9_later"]["9b"][1]],
           "d9_prediction_grade": grade(d9_gain - 1, *PRED["d9_later"]["9b"]),
           "steady_26_50_over_11_25": round(w["26-50"] / w["11-25"], 4),
           "cache_by_epoch": epochs, "cache_size": size,
           "footprint_gb": r["guard"]["peak_footprint_gb"], "footprint_met": r["guard"]["peak_footprint_gb"] <= BARS["footprint_9_gb"],
           "d4_footprint_gb": d4["guard"]["peak_footprint_gb"],
           "live_at_forward_end_peak_gib": r["live_at_forward_end_peak_gb"], "d4_live_peak_gib": d4["live_at_forward_end_peak_gb"],
           "driver_peak_gb": r["mps_driver_peak_gb"], "d4_driver_peak_gb": d4["mps_driver_peak_gb"],
           "steps_1_2": {s: {"run": l[s], "d4": l4[s], "rel": abs(l[s] - l4[s]) / l4[s], "phase4_9b_reference": lp.get(s)}
                         for s in (1, 2)},
           "learning_rates_step_2_3": {"run": [x.get("learning_rate") for x in r["steps"][1:3]],
                                       "phase4_9b_reference": [x.get("learning_rate") for x in p4_9b["steps"][1:3]]},
           "final_loss": r["loss_final_step"], "mean_loss": r["trainer"]["train_loss"], "d8_alone_9b": d8_alone,
           "repo_commit": r.get("repo_commit"), "repo_dirty": r.get("repo_dirty"), "wall_s": r["wall_s"]}
    return out


def profile():
    path = os.path.join(LOGS, PROFILE + ".json")
    if not os.path.exists(path):
        return None
    d2m = os.path.join(HERE, "memory_logs", "confirm_recompute_0.8b_2x8_100steps.json")
    new = PR.report("0.8b", path, d2m)
    old = PR.report("0.8b", os.path.join(LOGS, PROFILE_D4 + ".json"), d2m)
    groups = {}
    for g in sorted(set(new["groups"]) | set(old["groups"]), key=lambda g: -(old["groups"].get(g, {}).get("ms_per_micro_batch") or 0)):
        a, b = old["groups"].get(g, {}), new["groups"].get(g, {})
        groups[g] = {"d4_ms": a.get("ms_per_micro_batch"), "d8_ms": b.get("ms_per_micro_batch"),
                     "d4_share_pct": a.get("share_pct"), "d8_share_pct": b.get("share_pct")}
    comps = {}
    for tag, rep in (("d4", old), ("d8", new)):
        for c in rep["components"]:
            if c["group"] == "lm_head_and_loss":
                comps.setdefault(c["component"], {})[tag] = {"fwd_ms": c["fwd_ms"], "bwd_ms": c["bwd_ms"], "ms": c["ms"]}
    hl = new["groups"]["lm_head_and_loss"]
    head_ms = sum(comps.get(k, {}).get("d8", {}).get("ms", 0) for k in ("top.lm_head",))
    loss_ms = sum(comps.get(k, {}).get("d8", {}).get("ms", 0) for k in ("top.loss",))
    rest_ms = hl["ms_per_micro_batch"] - head_ms - loss_ms
    lo, hi = PRED["head_loss_ms"]["0.8b"]
    return {"validity": new["validity"]["c_over_b_per_micro_batch"], "events_span": new["validity"]["events_span_over_c_wall"],
            "micro_batch_ms": [old["per_micro_batch_ms_total"], new["per_micro_batch_ms_total"]],
            "shapes": new.get("micro_batch_shapes"), "groups": groups, "head_and_loss_components": comps,
            "head_and_loss_ms": hl["ms_per_micro_batch"], "head_and_loss_share_pct": hl["share_pct"],
            "head_product_ms": round(head_ms, 2), "loss_ms": round(loss_ms, 2), "rest_ms": round(rest_ms, 2),
            "prediction": [lo, hi], "prediction_grade": grade(hl["ms_per_micro_batch"], lo, hi),
            "d8b_trigger": {"threshold_pct": 3.0, "share_pct": hl["share_pct"], "triggered": hl["share_pct"] >= 3.0},
            "repo_commit": new.get("repo_commit")}


def main():
    p4 = {k: MR.read(MR.P4, v) for k, v in MR.PHASE4.items()}
    d4_08, d4_9 = MR.read(LOGS, D4_08), MR.read(LOGS, D4_9)
    d2m_9 = MR.read(*os.path.split(D2M_9))
    out = {"bars": BARS, "predictions": PRED, "tests": tests_parity(), "probes": {m: probe(m) for m in PROBES},
           "loss_head_bench": loss_head(),
           "labelled_share": {"0.8b": labelled_share(D8_08_100), "9b": labelled_share(D8_9)}}
    out["d8_alone_0.8b"] = d8_alone_08(d4_08)
    out["integration_0.8b"] = integration_08(d4_08, out["d8_alone_0.8b"], p4)
    out["integration_9b"] = integration_9(d4_9, d2m_9, p4["ref_9b"])
    out["profile_0.8b_d4_d8"] = profile()
    print(json.dumps(out, indent=1, default=str))
    with open(os.path.join(LOGS, "d8d9_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", os.path.join(LOGS, "d8d9_results.json"))


if __name__ == "__main__":
    main()
