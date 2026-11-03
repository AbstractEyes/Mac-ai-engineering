"""Series phase 4: the measurements and verdicts from the runs' records (bench/phase4_logs/<run>.json and the guard's
bench/phase4_logs/<run>.log). The readings are the ones fixed in docs/series_phase4.md before the registered runs.

    python bench/phase4_report.py      # prints the tables, writes bench/phase4_logs/phase4_results.json
"""
import json
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "phase4_logs")
RUNS = {"ref_a": "p41_ref_a", "ref_b": "p41_ref_b", "kernels": "p41_kernels", "ref_9b": "p42_9b_ref",
        "kernels_9b": "p42_9b_kernels", "control_ref_c": "control_p41_ref_c"}
# the registered predictions (series journal, 2026-11-02 20:xx PST)
PRED = {"0.8b": {"kernels_s": (10.0, 12.0), "reference_s": (14.0, 15.0), "speedup": 1.25},
        "9b": {"kernels_s": (55.0, 62.0), "reference_s": (70.0, 75.0), "speedup": 1.2},
        "memory_over_reference": 1.10, "gap_over_spread": 2.0, "final_loss_rel": 0.01}


def guard(name: str) -> dict:
    """The guard's end line (exit code, peak footprint, wall) and any stop it made."""
    out = {"rc": None, "peak_footprint_gb": None, "wall_s": None, "stops": []}
    path = os.path.join(LOGS, name + ".log")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"\[guard\] end \S+ rc (\S+) peak footprint ([\d.]+) GB wall (\d+) s", line)
            if m:
                out.update(rc=int(m.group(1)), peak_footprint_gb=float(m.group(2)), wall_s=int(m.group(3)))
            if "[guard] STOP" in line:
                out["stops"].append(line.strip())
    return out


def load(key: str):
    path = os.path.join(LOGS, RUNS[key] + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    r["guard"] = guard(RUNS[key])
    return r


def losses(r: dict) -> dict:
    return {s["step"]: s["loss"] for s in r["steps"]}


def rms(xs) -> float:
    xs = list(xs)
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def parity(a: dict, b: dict, k: dict) -> dict:
    """Run k against the reference pair (a, b), step by step, by the readings fixed before the runs."""
    la, lb, lk = losses(a), losses(b), losses(k)
    steps = sorted(set(la) & set(lb) & set(lk))
    rows = []
    for s in steps:
        ref = (la[s] + lb[s]) / 2
        rows.append({"step": s, "a": la[s], "b": lb[s], "k": lk[s], "ref": ref, "spread": abs(la[s] - lb[s]),
                     "gap": abs(lk[s] - ref)})
    if not rows:
        return {"steps": 0}
    max_spread = max(r["spread"] for r in rows)
    r1_fail = [r["step"] for r in rows if r["gap"] > PRED["gap_over_spread"] * r["spread"]]
    r2_fail = [r["step"] for r in rows if r["gap"] > PRED["gap_over_spread"] * max_spread]
    zero_spread = [r["step"] for r in rows if r["spread"] == 0.0]
    last = rows[-1]
    final_rel = abs(last["k"] - last["ref"]) / last["ref"]
    worst = max(rows, key=lambda r: r["gap"])
    ratios = [r["gap"] / r["spread"] for r in rows if r["spread"] > 0]
    mean_ref = (a["trainer"]["train_loss"] + b["trainer"]["train_loss"]) / 2
    return {
        "steps": len(rows), "rows": rows,
        "max_spread": max_spread, "max_spread_step": max(rows, key=lambda r: r["spread"])["step"],
        "worst_gap": worst["gap"], "worst_gap_step": worst["step"], "worst_gap_rel": worst["gap"] / worst["ref"],
        "zero_spread_steps": zero_spread,
        "reading1_failing_steps": r1_fail, "reading1_pass": not r1_fail,
        "reading1_failing_steps_where_spread_nonzero": [s for s in r1_fail if s not in zero_spread],
        "reading2_failing_steps": r2_fail, "reading2_pass": not r2_fail,
        "rms_gap": rms(r["gap"] for r in rows), "rms_spread": rms(r["spread"] for r in rows),
        "rms_gap_over_rms_spread": rms(r["gap"] for r in rows) / rms(r["spread"] for r in rows),
        "median_gap_over_spread": statistics.median(ratios) if ratios else None,
        "final_step": last["step"], "final_loss_k": last["k"], "final_loss_ref": last["ref"],
        "final_loss_a": last["a"], "final_loss_b": last["b"], "final_loss_rel_diff": final_rel,
        "final_loss_pass": final_rel <= PRED["final_loss_rel"],
        "mean_loss_k": k["trainer"]["train_loss"], "mean_loss_ref": mean_ref,
        "mean_loss_rel_diff": abs(k["trainer"]["train_loss"] - mean_ref) / mean_ref,
    }


def step_times(r: dict) -> dict:
    """{step: seconds of that step} from the runner's clock."""
    t = {s["step"]: s["t_s"] for s in r["steps"]}
    return {s: t[s] - t[s - 1] for s in t if s - 1 in t}


def timing(refs: list, k: dict, model: str) -> dict:
    """The steady seconds a step (the trainer's clock; steps after the warm-up), the reference as the mean of its runs, and the
    paired per-step ratio (the same batches in the same order in every run) from the runner's clock."""
    p = PRED[model]
    ref_s = statistics.mean(r["s_per_step_steady_trainer_clock"] for r in refs)
    k_s = k["s_per_step_steady_trainer_clock"]
    w = k["warmup_steps_excluded"]
    kt = step_times(k)
    paired = []
    for s, dt in kt.items():
        if s > w and all(s in step_times(r) for r in refs):
            paired.append(statistics.mean(step_times(r)[s] for r in refs) / dt)
    speed = ref_s / k_s
    return {
        "reference_runs_s": [r["s_per_step_steady_trainer_clock"] for r in refs],
        "reference_runs_s_runner_clock": [r["s_per_step_steady_runner_clock"] for r in refs],
        "reference_s": round(ref_s, 3), "kernels_s": k_s, "kernels_s_runner_clock": k["s_per_step_steady_runner_clock"],
        "speedup": round(speed, 3), "median_paired_step_ratio": round(statistics.median(paired), 3) if paired else None,
        "steps_in_window": f"{w + 1}-{max(kt) if kt else '?'}",
        "kernels_in_predicted_range": p["kernels_s"][0] <= k_s <= p["kernels_s"][1],
        "reference_in_expected_range": p["reference_s"][0] <= ref_s <= p["reference_s"][1],
        "speedup_met": speed >= p["speedup"], "predicted": p,
    }


def memory(refs: list, k: dict) -> dict:
    ref_peaks = [r["guard"]["peak_footprint_gb"] for r in refs]
    ref = statistics.mean(ref_peaks)
    kp = k["guard"]["peak_footprint_gb"]
    return {
        "reference_peak_footprint_gb": ref_peaks, "reference_mean_gb": round(ref, 2), "kernels_peak_footprint_gb": kp,
        "kernels_over_reference": round(kp / ref, 3), "met": kp <= PRED["memory_over_reference"] * ref,
        "reference_live_at_forward_end_peak_gb": [r.get("live_at_forward_end_peak_gb") for r in refs],
        "kernels_live_at_forward_end_peak_gb": k.get("live_at_forward_end_peak_gb"),
        "reference_mps_driver_peak_gb": [r.get("mps_driver_peak_gb") for r in refs],
        "kernels_mps_driver_peak_gb": k.get("mps_driver_peak_gb"),
    }


def main():
    runs = {key: load(key) for key in RUNS}
    out = {"runs": {}, "predictions": PRED}
    for key, r in runs.items():
        if r is None:
            out["runs"][key] = None
            continue
        out["runs"][key] = {k: r.get(k) for k in ("name", "model_id", "arm", "epochs", "batch", "grad_accum", "examples",
                                                    "repo_commit", "error", "s_per_step_steady_trainer_clock",
                                                    "s_per_step_steady_runner_clock", "loss_final_step", "wall_s",
                                                    "live_at_forward_end_peak_gb", "mps_driver_peak_gb", "gated_delta_rule")}
        out["runs"][key]["train_loss"] = (r.get("trainer") or {}).get("train_loss")
        out["runs"][key]["steps"] = (r.get("trainer") or {}).get("steps")
        out["runs"][key]["lora_A_abs_sum"] = (r.get("checks") or {}).get("lora_A_abs_sum")
        out["runs"][key]["guard"] = r["guard"]
    ok = lambda *ks: all(runs[k] is not None and not runs[k].get("error") for k in ks)  # noqa: E731
    if ok("ref_a", "ref_b", "kernels"):
        p = parity(runs["ref_a"], runs["ref_b"], runs["kernels"])
        out["p4_1"] = p
        out["p4_2_time_0.8b"] = timing([runs["ref_a"], runs["ref_b"]], runs["kernels"], "0.8b")
        out["p4_2_memory_0.8b"] = memory([runs["ref_a"], runs["ref_b"]], runs["kernels"])
        print(f"P4.1: max spread {p['max_spread']:.4g} (step {p['max_spread_step']}), worst gap {p['worst_gap']:.4g} "
              f"(step {p['worst_gap_step']}, {p['worst_gap_rel']:.2%}); reading 1 fails at {len(p['reading1_failing_steps'])} "
              f"steps {p['reading1_failing_steps'][:12]}; reading 2 fails at {p['reading2_failing_steps']}; rms gap / rms spread "
              f"{p['rms_gap_over_rms_spread']:.3f}; final {p['final_loss_k']:.4f} vs {p['final_loss_ref']:.4f} "
              f"({p['final_loss_rel_diff']:.3%}); mean loss {p['mean_loss_k']:.4f} vs {p['mean_loss_ref']:.4f}")
        print("P4.2 0.8B time:", json.dumps(out["p4_2_time_0.8b"]))
        print("P4.2 0.8B memory:", json.dumps(out["p4_2_memory_0.8b"]))
        print("\n| step | reference A | reference B | kernels | spread A-B | kernels' gap | gap / spread |")
        print("|---|---|---|---|---|---|---|")
        show = sorted({1, 2, 3, 4, 5, 10, 25, 50, 75, 100, p["worst_gap_step"], p["max_spread_step"]})
        for row in p["rows"]:
            if row["step"] in show:
                ratio = f"{row['gap'] / row['spread']:.2f}" if row["spread"] else "(spread 0)"
                print(f"| {row['step']} | {row['a']:.4f} | {row['b']:.4f} | {row['k']:.4f} | {row['spread']:.4f} | "
                      f"{row['gap']:.4f} | {ratio} |")
    if ok("ref_a", "ref_b", "control_ref_c"):
        c = parity(runs["ref_a"], runs["ref_b"], runs["control_ref_c"])
        out["control_ref_c"] = {k: v for k, v in c.items() if k != "rows"}
        print(f"\ncontrol C (a third reference run, judged like the kernels): reading 1 fails at "
              f"{len(c['reading1_failing_steps'])} steps; reading 2 fails at {c['reading2_failing_steps']}; rms gap / rms spread "
              f"{c['rms_gap_over_rms_spread']:.3f}; worst gap {c['worst_gap']:.4g} (step {c['worst_gap_step']}); final "
              f"{c['final_loss_rel_diff']:.3%}")
        if ok("kernels"):
            lc, lk, la, lb = (losses(runs[x]) for x in ("control_ref_c", "kernels", "ref_a", "ref_b"))
            pairs = {"A-B": (la, lb), "A-C": (la, lc), "B-C": (lb, lc), "K-A": (lk, la), "K-B": (lk, lb), "K-C": (lk, lc)}
            out["pairwise_rms_loss_difference"] = {n: rms(x[s] - y[s] for s in x if s in y) for n, (x, y) in pairs.items()}
            print("pairwise rms loss differences:", {n: round(v, 5) for n, v in out["pairwise_rms_loss_difference"].items()})
    if ok("ref_9b", "kernels_9b"):
        out["p4_2_time_9b"] = timing([runs["ref_9b"]], runs["kernels_9b"], "9b")
        out["p4_2_memory_9b"] = memory([runs["ref_9b"]], runs["kernels_9b"])
        l9r, l9k = losses(runs["ref_9b"]), losses(runs["kernels_9b"])
        out["loss_9b"] = {"reference_final": l9r[max(l9r)], "kernels_final": l9k[max(l9k)],
                          "reference_mean": runs["ref_9b"]["trainer"]["train_loss"],
                          "kernels_mean": runs["kernels_9b"]["trainer"]["train_loss"],
                          "steps_1_2_rel_diff": [abs(l9k[s] - l9r[s]) / l9r[s] for s in (1, 2) if s in l9k and s in l9r]}
        print("\nP4.2 9B time:", json.dumps(out["p4_2_time_9b"]))
        print("P4.2 9B memory:", json.dumps(out["p4_2_memory_9b"]))
        print("9B loss:", json.dumps(out["loss_9b"]))
    with open(os.path.join(LOGS, "phase4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(LOGS, "phase4_results.json"))


if __name__ == "__main__":
    main()
