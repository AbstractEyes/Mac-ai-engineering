"""Series memory (the kernels' memory, docs/series_memory.md): every figure and verdict from the runs' records, read the way
Phase 4 read them (bench/phase4_report.py). The memory runs are bench/memory_logs/<run>.json with the guard's
bench/memory_logs/<run>.log; the Phase 4 runs they are read against are bench/phase4_logs/<run>.json / .log.

    python bench/memory_report.py      # prints the tables, writes bench/memory_logs/memory_results.json

The bars and predictions are the registered ones (series journal, 2026-11-03 00:2x PST); the bars are also recomputed here from
the Phase 4 records as a check.
"""
import json
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
MEM, P4 = os.path.join(HERE, "memory_logs"), os.path.join(HERE, "phase4_logs")
ORDER = ["classes", "recompute", "classes+recompute"]          # the registered order of the designs
SLUG = {"classes": "classes", "recompute": "recompute", "classes+recompute": "classes_recompute"}
SCREEN = {d: f"screen_{SLUG[d]}_0.8b_2x8_25steps" for d in ORDER}
CONFIRM_08 = {d: f"confirm_{SLUG[d]}_0.8b_2x8_100steps" for d in ORDER}
CONFIRM_9 = {d: f"confirm_{SLUG[d]}_9b_1x16_100steps" for d in ORDER}
PHASE4 = {"ref_a": "p41_ref_a", "ref_b": "p41_ref_b", "ref_c": "control_p41_ref_c", "kernels": "p41_kernels",
          "ref_9b": "p42_9b_ref_1x16", "kernels_9b": "p42_9b_kernels_1x16"}
BARS = {"screen_driver_by_step20_gb": 43.3, "confirm_08_footprint_gb": 53.5, "confirm_08_steady_s": 11.39,
        "confirm_9_footprint_steps_1_25_gb": 74.6, "confirm_9_steady_11_25_s": 52.7, "long_run_growth_gb": 2.0}
PRED = {"classes": {"step_time_within": 0.01, "live_up_at_most": 0.05, "excess_cut_at_least": 0.5},
        "recompute": {"step_08_s": (11.1, 11.3), "step_9_s": (51.5, 52.0), "live_08_gib": 15.5, "live_9_gib": 21.0,
                      "live_cut_08_gib": 3.0, "live_cut_9_gib": 4.0}}


def guard(path: str) -> dict:
    """The guard's end line and its 30 s lines (seconds, footprint, running peak), in file order with the trainer's step lines."""
    out = {"rc": None, "peak_footprint_gb": None, "wall_s": None, "stops": [], "events": []}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"\[guard\] end \S+ rc (\S+) peak footprint ([\d.]+) GB wall (\d+) s", line)
            if m:
                out.update(rc=int(m.group(1)), peak_footprint_gb=float(m.group(2)), wall_s=int(m.group(3)))
            m = re.search(r"\[guard\]\s+(\d+) s footprint\s+([\d.]+) GB \(peak\s+([\d.]+)\)", line)
            if m:
                out["events"].append(("guard", int(m.group(1)), float(m.group(2)), float(m.group(3))))
            for m in re.finditer(r"step (\d+)/(\d+): loss [\d.]+", line):
                out["events"].append(("step", int(m.group(1))))
            if "[guard] STOP" in line:
                out["stops"].append(line.strip())
    return out


def guard_peak_through(g: dict, step: int):
    """The guard's running peak at its last line written before the trainer reported step + 1 (so it covers steps 1..step
    and part of step + 1, the model's load included)."""
    peak = None
    for e in g["events"]:
        if e[0] == "step" and e[1] == step + 1:
            break
        if e[0] == "guard":
            peak = e[3]
    return peak


def read(folder: str, name: str):
    path = os.path.join(folder, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    r["guard"] = guard(os.path.join(folder, name + ".log"))
    return r


def ok(r) -> bool:
    return r is not None and not r.get("error") and bool(r.get("steps"))


def by_step(r: dict, key: str) -> dict:
    return {s["step"]: s[key] for s in r["steps"] if key in s}


def window_max(r: dict, key: str, lo: int, hi: int):
    xs = [v for s, v in by_step(r, key).items() if lo <= s <= hi]
    return max(xs) if xs else None


def trainer_clock(r: dict) -> dict:
    return {x["step"]: x["spent_s"] for x in r.get("trainer_log", []) if "loss" in x and "spent_s" in x}


def steady(r: dict, lo: int, hi: int):
    """Seconds a step on the trainer's clock between the end of step lo - 1 and the end of step hi."""
    t = trainer_clock(r)
    if lo - 1 not in t or hi not in t:
        return None
    return round((t[hi] - t[lo - 1]) / (hi - lo + 1), 3)


def rms(xs) -> float:
    xs = list(xs)
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def rms_pair(x: dict, y: dict) -> float:
    return rms(x[s] - y[s] for s in x if s in y)


def steps_1_2_equal(r: dict, base: dict) -> dict:
    lr, lb = by_step(r, "loss"), by_step(base, "loss")
    return {"steps": {s: [lr.get(s), lb.get(s)] for s in (1, 2)}, "equal": all(lr.get(s) == lb.get(s) for s in (1, 2))}


def summary(r: dict) -> dict:
    return {k: r.get(k) for k in ("name", "memory", "model_id", "epochs", "batch", "grad_accum", "repo_commit", "repo_dirty",
                                  "error", "s_per_step_steady_trainer_clock", "loss_final_step", "live_at_forward_end_peak_gb",
                                  "mps_driver_peak_gb", "footprint_peak_in_steps_gb", "footprint_peak_before_training_gb",
                                  "wall_s")} | {"macai": r.get("macai"), "guard": {k: v for k, v in r["guard"].items()
                                                                                    if k != "events"},
                                                "train_loss": (r.get("trainer") or {}).get("train_loss")}


def main():
    p4 = {k: read(P4, v) for k, v in PHASE4.items()}
    refs = [p4["ref_a"], p4["ref_b"], p4["ref_c"]]
    out = {"bars": BARS, "predictions": PRED, "phase4": {}, "screen": {}, "confirm_0.8b": {}, "confirm_9b": {}}
    # the bars recomputed from the Phase 4 records
    ref_drv20 = [window_max(r, "mps_driver_gb", 1, 20) for r in refs]
    k_drv20 = window_max(p4["kernels"], "mps_driver_gb", 1, 20)
    ref_fp = statistics.mean(r["guard"]["peak_footprint_gb"] for r in refs[:2])
    out["phase4"] = {
        "reference_driver_max_by_step_20_gb": ref_drv20, "kernels_driver_max_by_step_20_gb": k_drv20,
        "reference_driver_mean_by_step_20_gb": round(statistics.mean(ref_drv20), 3),
        "bars_recomputed": {"screen_driver_by_step20_gb": round(max(ref_drv20) * 1.1, 2),
                            "confirm_08_footprint_gb": round(ref_fp * 1.1, 2),
                            "confirm_08_steady_s": round(p4["kernels"]["s_per_step_steady_trainer_clock"] * 1.03, 2),
                            "confirm_9_footprint_steps_1_25_gb": round(p4["ref_9b"]["guard"]["peak_footprint_gb"] * 1.1, 2),
                            "confirm_9_steady_11_25_s": round(p4["kernels_9b"]["s_per_step_steady_trainer_clock"] * 1.03, 2)},
        "reference_peak_footprint_gb": [r["guard"]["peak_footprint_gb"] for r in refs],
        "kernels_peak_footprint_gb": p4["kernels"]["guard"]["peak_footprint_gb"],
        "kernels_steady_11_25_s": steady(p4["kernels"], 11, 25),
        "kernels_live_peak_by_step_25_gib": window_max(p4["kernels"], "live_at_forward_end_gb", 1, 25),
        "reference_live_peak_by_step_25_gib": [window_max(r, "live_at_forward_end_gb", 1, 25) for r in refs],
        "kernels_guard_peak_through_step_25_gb": guard_peak_through(p4["kernels"]["guard"], 25),
        "reference_guard_peak_through_step_25_gb": [guard_peak_through(r["guard"], 25) for r in refs],
        "kernels_steady_11_25_runner_s": round((by_step(p4["kernels"], "t_s")[25] - by_step(p4["kernels"], "t_s")[10]) / 15, 3),
    }
    ref_mean20 = statistics.mean(ref_drv20)
    excess_k = k_drv20 - ref_mean20
    print(f"Phase 4, driver memory max by step 20: reference {ref_drv20} (mean {ref_mean20:.2f}), kernels {k_drv20}; "
          f"bars recomputed {out['phase4']['bars_recomputed']}")

    # the screens
    print("\n| design | driver max by step 20, GB (bar 43.3) | the kernels' excess over the reference's mean, GB | live at a forward's "
          "end, peak by step 25, GiB | per-step footprint peak by step 25, GB | guard's peak, GB | steps 11-25, s a step "
          "(Phase 4's kernels: same steps) | steps 1-2 losses = Phase 4's kernels' | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    k_live25 = out["phase4"]["kernels_live_peak_by_step_25_gib"]
    k_t = out["phase4"]["kernels_steady_11_25_s"]
    passed = []
    for d in ORDER:
        r = read(MEM, SCREEN[d])
        if r is None:
            out["screen"][d] = None
            continue
        s = summary(r)
        if ok(r):
            drv20 = window_max(r, "mps_driver_gb", 1, 20)
            live25 = window_max(r, "live_at_forward_end_gb", 1, 25)
            t = steady(r, 11, 25)
            eq = steps_1_2_equal(r, p4["kernels"])
            verdict = drv20 is not None and drv20 <= BARS["screen_driver_by_step20_gb"]
            tr = by_step(r, "t_s")
            t_run = round((tr[25] - tr[10]) / 15, 3) if 25 in tr and 10 in tr else None
            s.update(driver_max_by_step_20_gb=drv20, excess_over_reference_gb=round(drv20 - ref_mean20, 3),
                     excess_vs_phase4_kernels=round((drv20 - ref_mean20) / excess_k, 3), live_peak_by_step_25_gib=live25,
                     live_vs_phase4_kernels=round(live25 / k_live25, 4), steady_11_25_s=t,
                     steady_vs_phase4_kernels_same_steps=round(t / k_t, 4) if t else None, steady_11_25_runner_s=t_run,
                     steady_runner_vs_phase4_kernels_same_steps=(round(t_run / out["phase4"]["kernels_steady_11_25_runner_s"], 4)
                                                                 if t_run else None),
                     footprint_peak_in_steps_by_25_gb=window_max(r, "footprint_peak_in_step_gb", 1, 25),
                     steps_1_2_vs_phase4_kernels=eq, passes=verdict)
            if verdict:
                passed.append(d)
            print(f"| {d} | {drv20:.2f} | {drv20 - ref_mean20:.2f} (Phase 4's kernels {excess_k:.2f}) | {live25:.2f} "
                  f"(Phase 4's kernels {k_live25:.2f}) | {s['footprint_peak_in_steps_by_25_gb']:.2f} | "
                  f"{r['guard']['peak_footprint_gb']} | {t} ({k_t}) | {eq['equal']} | {'PASS' if verdict else 'MISS'} |")
        out["screen"][d] = s
    out["screen_passed_in_order"] = passed

    # the confirmations
    la, lb, lc, lk = (by_step(p4[k], "loss") for k in ("ref_a", "ref_b", "ref_c", "kernels"))
    for d in ORDER:
        r = read(MEM, CONFIRM_08[d])
        if r is None:
            continue
        s = summary(r)
        if ok(r):
            ld = by_step(r, "loss")
            last = max(ld)
            ref_last = (la[last] + lb[last]) / 2
            ref_mean = (p4["ref_a"]["trainer"]["train_loss"] + p4["ref_b"]["trainer"]["train_loss"]) / 2
            fp = r["guard"]["peak_footprint_gb"]
            st = r["s_per_step_steady_trainer_clock"]
            s.update(
                footprint_gb=fp, footprint_met=fp <= BARS["confirm_08_footprint_gb"], reference_mean_footprint_gb=ref_fp,
                footprint_over_reference=round(fp / ref_fp, 3), steady_s=st, steady_met=st <= BARS["confirm_08_steady_s"],
                steady_vs_phase4_kernels=round(st / p4["kernels"]["s_per_step_steady_trainer_clock"], 4),
                rms_loss_difference={"D-A": rms_pair(ld, la), "D-B": rms_pair(ld, lb), "D-C": rms_pair(ld, lc),
                                     "D-kernels": rms_pair(ld, lk), "A-B": rms_pair(la, lb), "A-C": rms_pair(la, lc),
                                     "B-C": rms_pair(lb, lc)},
                final_loss=ld[last], final_loss_reference=ref_last, final_loss_rel=abs(ld[last] - ref_last) / ref_last,
                mean_loss=r["trainer"]["train_loss"], mean_loss_reference=ref_mean,
                mean_loss_rel=abs(r["trainer"]["train_loss"] - ref_mean) / ref_mean,
                driver_peak_gb=r["mps_driver_peak_gb"], reference_driver_peak_gb=[x["mps_driver_peak_gb"] for x in refs],
                phase4_kernels_driver_peak_gb=p4["kernels"]["mps_driver_peak_gb"],
                live_peak_gib=r["live_at_forward_end_peak_gb"], phase4_kernels_live_peak_gib=p4["kernels"]["live_at_forward_end_peak_gb"],
                reference_live_peak_gib=[x["live_at_forward_end_peak_gb"] for x in refs],
                steps_1_2_vs_phase4_kernels=steps_1_2_equal(r, p4["kernels"]))
            print(f"\n0.8B confirmation, {d}: footprint {fp} GB (bar {BARS['confirm_08_footprint_gb']}; reference mean "
                  f"{ref_fp:.2f}): {'MET' if s['footprint_met'] else 'MISSED'}; steady {st} s (bar {BARS['confirm_08_steady_s']}): "
                  f"{'MET' if s['steady_met'] else 'MISSED'}; final loss {ld[last]:.4f} vs {ref_last:.4f} "
                  f"({s['final_loss_rel']:.2%}); mean {s['mean_loss']:.4f} vs {ref_mean:.4f} ({s['mean_loss_rel']:.2%}); rms "
                  + json.dumps({k: round(v, 4) for k, v in s["rms_loss_difference"].items()}))
        out["confirm_0.8b"][d] = s
    for d in ORDER:
        r = read(MEM, CONFIRM_9[d])
        if r is None:
            continue
        s = summary(r)
        if ok(r):
            g25 = guard_peak_through(r["guard"], 25)
            run25 = max(window_max(r, "footprint_peak_in_step_gb", 1, 25), r.get("footprint_peak_before_training_gb") or 0)
            st = steady(r, 11, 25)
            d1, d2 = window_max(r, "mps_driver_gb", 1, 50), window_max(r, "mps_driver_gb", 51, 100)
            s.update(guard_peak_through_step_25_gb=g25, runner_peak_through_step_25_gb=run25,
                     footprint_met=g25 is not None and g25 <= BARS["confirm_9_footprint_steps_1_25_gb"],
                     footprint_met_runner_poll=run25 <= BARS["confirm_9_footprint_steps_1_25_gb"],
                     phase4_reference_footprint_gb=p4["ref_9b"]["guard"]["peak_footprint_gb"],
                     phase4_kernels_footprint_gb=p4["kernels_9b"]["guard"]["peak_footprint_gb"],
                     steady_11_25_s=st, steady_met=st is not None and st <= BARS["confirm_9_steady_11_25_s"],
                     phase4_kernels_steady_11_25_s=p4["kernels_9b"]["s_per_step_steady_trainer_clock"],
                     phase4_reference_steady_11_25_s=p4["ref_9b"]["s_per_step_steady_trainer_clock"],
                     steady_11_100_s=r["s_per_step_steady_trainer_clock"],
                     driver_max_steps_1_50_gb=d1, driver_max_steps_51_100_gb=d2,
                     long_run_growth_gb=round(d2 - d1, 3) if d1 is not None and d2 is not None else None,
                     long_run_met=d1 is not None and d2 is not None and d2 <= d1 + BARS["long_run_growth_gb"],
                     driver_max_steps_1_25_gb=window_max(r, "mps_driver_gb", 1, 25),
                     phase4_kernels_driver_peak_gb=p4["kernels_9b"]["mps_driver_peak_gb"],
                     phase4_reference_driver_peak_gb=p4["ref_9b"]["mps_driver_peak_gb"],
                     live_peak_by_step_25_gib=window_max(r, "live_at_forward_end_gb", 1, 25),
                     phase4_kernels_live_peak_gib=p4["kernels_9b"]["live_at_forward_end_peak_gb"],
                     phase4_reference_live_peak_gib=p4["ref_9b"]["live_at_forward_end_peak_gb"],
                     steps_1_2_vs_phase4_kernels=steps_1_2_equal(r, p4["kernels_9b"]))
            print(f"\n9B confirmation, {d}: footprint through step 25: guard {g25} GB / runner {run25} GB (bar "
                  f"{BARS['confirm_9_footprint_steps_1_25_gb']}): {'MET' if s['footprint_met'] else 'MISSED'}; steady 11-25 {st} s "
                  f"(bar {BARS['confirm_9_steady_11_25_s']}): {'MET' if s['steady_met'] else 'MISSED'}; driver max steps 1-50 "
                  f"{d1}, 51-100 {d2} (growth {s['long_run_growth_gb']}, bar +2): {'MET' if s['long_run_met'] else 'MISSED'}")
        out["confirm_9b"][d] = s
    with open(os.path.join(MEM, "memory_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(MEM, "memory_results.json"))


if __name__ == "__main__":
    main()
