"""Series D3-D5, step 0: the step profile's tables from the in-situ records (bench/profile_step.py), graded against the
registered share predictions (docs/series_layer.md).

    python bench/profile_report.py     # reads bench/layer_logs/profile_{0.8b_2x8,9b_1x16}.json, writes profile_results.json

Per run: the instrument's validity (each profiled pass against the unprofiled replay of the same micro-batch; the components'
sum against the profiled pass's wall clock; the run's unprofiled steps and the profiled steps as trained against the D2m
confirmation runs' same steps, the same batches); the time by component per micro-batch and per optimizer step (forward,
backward, the CPU's net issue time, dispatch-bound or not); the registered groups and their shares; the parts outside D3-D5 at
5% of the step or more.
"""
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "layer_logs")
RUNS = {"0.8b": ("profile_0.8b_2x8.json", os.path.join(HERE, "memory_logs", "confirm_recompute_0.8b_2x8_100steps.json")),
        "9b": ("profile_9b_1x16.json", os.path.join(HERE, "memory_logs", "confirm_recompute_9b_1x16_100steps.json"))}

# the registered predictions: shares of a micro-batch (the step's own parts spread over its micro-batches)
PREDICTED = {"frozen_products": {"0.8b": (35, 45), "9b": (55, 65)},
             "lora_extra": {"0.8b": (8, 15), "9b": (5, 10)},
             "surroundings": {"0.8b": (10, 15), "9b": (6, 10)},
             "rule": {"0.8b": (6, 9), "9b": (3, 5)},
             "lm_head_and_loss": {"0.8b": (12, 20), "9b": (8, 14)},
             "vision_tower": {"0.8b": (8, 16), "9b": (6, 12)},
             "data_optimizer_trainer": {"0.8b": (5, 15), "9b": (3, 10)}}
GROUP_TEXT = {
    "frozen_products": "the layers' frozen products (every projection's base product, forward and the gradient to the input)",
    "lora_extra": "the LoRA adapters' extra work (dropout, the rank-32 products, the cast, scale and add, forward and backward)",
    "surroundings": "the surroundings (both RMSNorms, the convolution path, the gated norm, the SwiGLU, beta / g and the glue, "
                    "the residual adds; in the full-attention layers the q / k norms, the rotary embedding and the output gate)",
    "rule": "the gated delta rule",
    "full_attention_core": "the full-attention layers' attention (scaled dot-product attention)",
    "lm_head_and_loss": "the LM head and the loss (the logits converted to float32 by accelerate included)",
    "vision_tower": "the vision tower (forward only)",
    "other_model_parts": "the rest of the model (token embedding, picture merge, 3-D positions, masks, rotary tables, final norm)",
    "data_optimizer_trainer": "data, optimizer and the Trainer (collation, inputs to the GPU, the cache emptied each micro-batch, "
                              "gradient clipping, the optimizer, the scheduler and zero_grad, logging, the Trainer's own work)",
    "new_shapes": "a new input shape (the first pass of a micro-batch against its warm replay: A - B)",
}
DISPATCH = 0.8          # a component is dispatch-bound when its net CPU issue time >= 80% of its event time
MIN_MS = 0.5            # ... and its event time is at least this much a micro-batch (shorter ones are listed as noise)


def group_of(comp: str) -> str:
    scope, _, part = comp.partition(".")
    if scope in ("lin", "full"):
        if part.endswith(".base"):
            return "frozen_products"
        if any(part.endswith(s) for s in (".lora_dropout", ".lora_A", ".lora_B", ".lora_rest")):
            return "lora_extra"
        if part == "rule":
            return "rule"
        if part == "attention":
            return "full_attention_core"
        if part == "layer_undetailed":
            return "undetailed_layers"
        return "surroundings"
    if scope == "vis":
        return "vision_tower"
    if comp in ("top.lm_head", "top.loss", "top.peft_accelerate", "top.lm_head_gather", "top.loss_labels"):
        return "lm_head_and_loss"
    if scope == "top":
        return "other_model_parts"
    if scope in ("mb", "step"):
        return "data_optimizer_trainer"
    return "unassigned"


def d2m_step_ms(path):
    r = json.load(open(path))
    prev, out = 0.0, {}
    for s in r["steps"]:
        out[s["step"]] = (s["t_s"] - prev) * 1e3
        prev = s["t_s"]
    return out


def report(model, rec_path, d2m_path):
    rec = json.load(open(rec_path))
    P = rec["profile"]
    mbs, steps = P["micro_batches"], P["steps"]
    n_mb = len(mbs)
    out = {"record": os.path.basename(rec_path), "repo_commit": rec.get("repo_commit"), "profiled_steps": [s["step"] for s in steps],
           "micro_batches_profiled": n_mb}

    # ---- validity ----
    cb = [m["c_over_b"] for m in mbs]
    span = [m["events_span_over_c"] for m in mbs]
    sum_c = sum(m["c_ms"] for m in mbs)
    sum_b = sum(m["b_ms"] for m in mbs)
    sum_a = sum(m["a_ms"] for m in mbs)
    own = {r["step"]: r for r in rec["steps"]}
    run_step_ms, prev = {}, 0.0
    for r in rec["steps"]:
        run_step_ms[r["step"]] = (r["t_s"] - prev) * 1e3
        prev = r["t_s"]
    d2m = d2m_step_ms(d2m_path)
    prof = {s["step"]: s for s in steps}
    unprof = [s for s in run_step_ms if s not in prof and s > 1]
    out["validity"] = {
        "c_over_b_per_micro_batch": {"mean_of_sums": round(sum_c / sum_b, 4), "median": round(statistics.median(cb), 4),
                                     "min": round(min(cb), 4), "max": round(max(cb), 4),
                                     "over_1.10": sum(1 for x in cb if x > 1.10), "n": n_mb},
        "events_span_over_c_wall": {"median": round(statistics.median(span), 4), "min": round(min(span), 4)},
        "stack_mismatches": sum(m["checks"]["stack_mismatches"] for m in mbs),
        "negative_segments": sum(m["checks"]["negative_segments"] for m in mbs),
        "marks_per_micro_batch": round(statistics.mean(m["checks"]["marks"] for m in mbs)),
        "unprofiled_steps_against_d2m": {str(s): {"this_run_ms": round(run_step_ms[s], 1), "d2m_ms": round(d2m[s], 1),
                                                  "ratio": round(run_step_ms[s] / d2m[s], 4)} for s in sorted(unprof)},
        "profiled_steps_as_trained_against_d2m": {str(s): {"as_trained_ms": prof[s]["step_ms_as_trained"],
                                                           "d2m_ms": round(d2m[s], 1),
                                                           "ratio": round(prof[s]["step_ms_as_trained"] / d2m[s], 4)}
                                                  for s in sorted(prof)},
        "d2m_steady_s": json.load(open(d2m_path)).get("s_per_step_steady_trainer_clock"),
    }
    out["new_shape_cost"] = {"a_minus_b_ms_per_micro_batch": round((sum_a - sum_b) / n_mb, 2),
                             "of_a": round((sum_a - sum_b) / sum_a, 4)}

    # ---- the micro-batch's components (the profiled passes), mean per micro-batch ----
    # gpu fwd, gpu bwd, cpu net fwd, cpu net bwd, segments, gpu waiting on the cpu fwd, bwd
    comp = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for m in mbs:
        for key, v in m["rows"].items():
            g, c, n, cn = v[:4]
            w = v[4] if len(v) > 4 else 0.0
            name, direction = key.split("|")
            i = 0 if direction == "fwd" else 1
            comp[name][i] += g
            comp[name][2 + i] += cn
            comp[name][4] += n
            comp[name][5 + i] += w
    # ---- the step's own parts (outside the micro-batches), mean per profiled step ----
    outside = defaultdict(lambda: [0.0, 0.0, 0.0])
    for s in steps:
        for key, v in s["outside_ms"].items():
            name = key.split("|")[0]
            outside[name][0] += v[0]
            outside[name][1] += v[3]
            outside[name][2] += v[4] if len(v) > 4 else 0.0
    n_steps = len(steps)
    mb_per_step = n_mb / n_steps
    rows = []
    for name, (gf, gb, cf, cbk, n, wf, wb) in comp.items():
        rows.append({"component": name, "group": group_of(name),
                     "fwd_ms": gf / n_mb, "bwd_ms": gb / n_mb, "ms": (gf + gb) / n_mb,
                     "cpu_net_fwd_ms": cf / n_mb, "cpu_net_bwd_ms": cbk / n_mb, "segments": n / n_mb,
                     "gpu_waiting_fwd_ms": wf / n_mb, "gpu_waiting_bwd_ms": wb / n_mb, "level": "micro-batch"})
    for name, (g, cn, w) in outside.items():
        k = n_steps * mb_per_step
        rows.append({"component": name, "group": group_of(name), "fwd_ms": g / k, "bwd_ms": 0.0,
                     "ms": g / k, "cpu_net_fwd_ms": cn / k, "cpu_net_bwd_ms": 0.0, "segments": None,
                     "gpu_waiting_fwd_ms": w / k, "gpu_waiting_bwd_ms": 0.0, "level": "step (spread over its micro-batches)"})
    cold = (sum_a - sum_b) / n_mb
    rows.append({"component": "new_shapes", "group": "new_shapes", "fwd_ms": cold, "bwd_ms": 0.0, "ms": cold,
                  "cpu_net_fwd_ms": None, "cpu_net_bwd_ms": None, "segments": None, "gpu_waiting_fwd_ms": None,
                  "gpu_waiting_bwd_ms": None, "level": "micro-batch (A - B)"})
    # the registered reading: shares of the profiled micro-batch (its components and its part of the step's own parts); the
    # new-shape cost (A - B) and the instrument's overhead (C - B) are reported beside it, against the same denominator
    total = sum(r["ms"] for r in rows if r["group"] != "new_shapes")
    for r in rows:
        r["share"] = r["ms"] / total
        for d in ("fwd", "bwd"):
            g, c, w = r[f"{d}_ms"], r[f"cpu_net_{d}_ms"], r[f"gpu_waiting_{d}_ms"]
            # the registered rule: the CPU's (net) issue time >= 80% of the event time
            r[f"dispatch_bound_{d}"] = bool(c is not None and g >= MIN_MS and c >= DISPATCH * g)
            # the GPU's own reading: it had caught up with the CPU (waited for work) for >= 80% of the component's time
            r[f"gpu_waits_{d}"] = bool(w is not None and g >= MIN_MS and w >= DISPATCH * g)
    rows.sort(key=lambda r: -r["ms"])
    out["per_micro_batch_ms_total"] = round(total, 2)
    out["per_step_ms_total"] = round(total * mb_per_step, 1)
    out["micro_batches_per_step"] = mb_per_step
    out["mean_c_ms"] = round(sum_c / n_mb, 2)
    out["mean_b_ms"] = round(sum_b / n_mb, 2)
    out["mean_a_ms"] = round(sum_a / n_mb, 2)
    out["instrument_overhead_ms_per_micro_batch"] = round((sum_c - sum_b) / n_mb, 2)
    out["beside_the_shares_pct"] = {"new_shapes (A - B)": round(100 * cold / total, 2),
                                    "the instrument (C - B)": round(100 * (sum_c - sum_b) / n_mb / total, 2)}
    # the step as trained, rebuilt: the profiled components less the instrument's overhead plus the new-shape cost
    out["step_ms_rebuilt_as_trained"] = round((total - (sum_c - sum_b) / n_mb + cold) * mb_per_step, 1)
    out["step_ms_as_trained_measured_mean"] = round(statistics.mean(s["step_ms_as_trained"] for s in steps), 1)
    out["components"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]

    groups = defaultdict(float)
    for r in rows:
        groups[r["group"]] += r["ms"]
    out["groups"] = {}
    for g, ms in sorted(groups.items(), key=lambda kv: -kv[1]):
        share = 100 * ms / total
        e = {"ms_per_micro_batch": round(ms, 2), "ms_per_step": round(ms * mb_per_step, 1), "share_pct": round(share, 2),
             "text": GROUP_TEXT.get(g, g)}
        if g in PREDICTED:
            lo, hi = PREDICTED[g][model]
            e["predicted_pct"] = [lo, hi]
            e["verdict"] = "met" if lo <= share <= hi else ("missed (lower)" if share < lo else "missed (higher)")
        out["groups"][g] = e

    # the frozen products' rate in situ: their FLOPs at the micro-batches' mean padded token count over their event time
    try:
        import sys
        sys.path.insert(0, HERE)
        from profile_products import MODELS, text_config
        tc = text_config(MODELS[model])
        h, i = tc["hidden_size"], tc["intermediate_size"]
        kd = tc["linear_key_head_dim"] * tc["linear_num_key_heads"]
        vd = tc["linear_value_head_dim"] * tc["linear_num_value_heads"]
        nq, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        n_lin_l = sum(t == "linear_attention" for t in tc["layer_types"])
        n_full_l = len(tc["layer_types"]) - n_lin_l
        dims = {"lin.in_proj_qkv.base": (h, 2 * kd + vd, n_lin_l), "lin.in_proj_z.base": (h, vd, n_lin_l),
                "lin.out_proj.base": (vd, h, n_lin_l), "lin.gate_proj.base": (h, i, n_lin_l), "lin.up_proj.base": (h, i, n_lin_l),
                "lin.down_proj.base": (i, h, n_lin_l), "full.q_proj.base": (h, 2 * nq * hd, n_full_l),
                "full.k_proj.base": (h, nkv * hd, n_full_l), "full.v_proj.base": (h, nkv * hd, n_full_l),
                "full.o_proj.base": (nq * hd, h, n_full_l), "full.gate_proj.base": (h, i, n_full_l),
                "full.up_proj.base": (h, i, n_full_l), "full.down_proj.base": (i, h, n_full_l),
                "top.lm_head": (h, tc["vocab_size"], 1)}
        m_mean = statistics.mean(m["shapes"]["input_ids"][0] * m["shapes"]["input_ids"][1] for m in mbs)
        rates = []
        for r in rows:
            if r["component"] in dims:
                k_in, n_out, layers = dims[r["component"]]
                fl = 2 * m_mean * k_in * n_out * layers
                rates.append({"component": r["component"], "fwd_tflops": round(fl / r["fwd_ms"] / 1e9, 2) if r["fwd_ms"] else None,
                              "bwd_tflops": round(fl / r["bwd_ms"] / 1e9, 2) if r["bwd_ms"] else None,
                              "ms": round(r["ms"], 2)})
        out["frozen_products_in_situ"] = {"mean_padded_tokens_per_micro_batch": round(m_mean, 1), "rates": rates}
    except Exception as ex:  # noqa: BLE001 - the config is read from the offline Hub cache; without it the rates are skipped
        out["frozen_products_in_situ"] = {"error": f"{type(ex).__name__}: {ex}"}

    # per layer class: the in-situ layer's forward + backward per layer (the micro-batch's mean shape)
    n_lin = rec.get("checks", {}).get("linear_attention_layers")
    n_layers = mbs[0].get("plan", {}).get("layers")
    if n_lin and n_layers:
        per = {}
        for scope, count in (("lin", n_lin), ("full", n_layers - n_lin)):
            f = sum(r["fwd_ms"] for r in rows if r["component"].startswith(scope + "."))
            b = sum(r["bwd_ms"] for r in rows if r["component"].startswith(scope + "."))
            per[scope] = {"layers": count, "fwd_ms_per_layer": round(f / count, 3), "bwd_ms_per_layer": round(b / count, 3),
                          "fwd_bwd_ms_per_layer": round((f + b) / count, 3)}
        out["per_layer_in_situ"] = per
    out["dispatch_bound"] = [{"component": r["component"], "direction": d, "ms": round(r[f"{d}_ms"], 3),
                              "cpu_net_ms": round(r[f"cpu_net_{d}_ms"], 3), "gpu_waiting_ms": round(r[f"gpu_waiting_{d}_ms"], 3),
                              "gpu_waits_on_cpu": r[f"gpu_waits_{d}"]}
                             for r in rows for d in ("fwd", "bwd") if r.get(f"dispatch_bound_{d}") or r.get(f"gpu_waits_{d}")]
    # the GPU's waiting on the CPU, summed over the micro-batch's components
    out["gpu_waiting_ms_per_micro_batch"] = round(sum((r["gpu_waiting_fwd_ms"] or 0) + (r["gpu_waiting_bwd_ms"] or 0)
                                                      for r in rows if r["level"] == "micro-batch"), 2)
    # the parts outside D3-D5 (the layers' own parts) at >= 5% of the step
    layer_groups = {"frozen_products", "lora_extra", "surroundings", "rule", "full_attention_core", "undetailed_layers"}
    out["outside_d3_d5_at_5pct"] = [{"component": r["component"], "group": r["group"], "share_pct": round(100 * r["share"], 2),
                                     "ms_per_step": round(r["ms"] * mb_per_step, 1)}
                                    for r in rows if r["group"] not in layer_groups and r["share"] >= 0.05]
    out["outside_d3_d5_groups_at_5pct"] = {g: v["share_pct"] for g, v in out["groups"].items()
                                           if g not in layer_groups and v["share_pct"] >= 5}
    # the micro-batch shapes: the median padded length, for the isolated benches
    lens = sorted(m["shapes"]["input_ids"][1] for m in mbs)
    out["micro_batch_shapes"] = {"batch": mbs[0]["shapes"]["input_ids"][0], "padded_lengths": lens,
                                 "median_padded_length": lens[len(lens) // 2] if len(lens) % 2 else
                                 (lens[len(lens) // 2 - 1] + lens[len(lens) // 2]) // 2,
                                 "real_tokens_mean": round(statistics.mean(m["shapes"].get("real_tokens", 0) for m in mbs), 1),
                                 "loss_tokens_mean": round(statistics.mean(m["shapes"].get("loss_tokens", 0) for m in mbs), 1),
                                 "pictures_mean": round(statistics.mean(m["shapes"].get("pictures", 0) for m in mbs), 2),
                                 "picture_patches_mean": round(statistics.mean(m["shapes"].get("picture_patches", 0)
                                                                               for m in mbs), 1)}
    out["collator_cpu_ms_per_call"] = round(statistics.mean(c for s in steps for c in s["collator_cpu_ms"]), 2)
    return out


PRODUCT_OF = {  # in-situ component -> (projection, product, layout) in bench/profile_products.py
    "top.lm_head": ("lm_head", None), "lin.in_proj_qkv.base": ("in_proj_qkv", None), "lin.in_proj_z.base": ("in_proj_z", None),
    "lin.out_proj.base": ("out_proj", None), "lin.gate_proj.base": ("gate_proj / up_proj", None),
    "lin.up_proj.base": ("gate_proj / up_proj", None), "lin.down_proj.base": ("down_proj", None),
    "full.gate_proj.base": ("gate_proj / up_proj", None), "full.up_proj.base": ("gate_proj / up_proj", None),
    "full.down_proj.base": ("down_proj", None), "full.q_proj.base": ("q_proj", None), "full.o_proj.base": ("o_proj", None),
    "full.k_proj.base": ("k_proj / v_proj", None), "full.v_proj.base": ("k_proj / v_proj", None),
    "lin.in_proj_a.base": ("in_proj_a / in_proj_b", None), "lin.in_proj_b.base": ("in_proj_a / in_proj_b", None),
}


def products_join(res):
    """The products carrying the most in-situ time, with PyTorch's and MLX's rates at the micro-batch's median shape."""
    pt_path, mx_path = os.path.join(LOGS, "products_torch.json"), os.path.join(LOGS, "products_mlx.json")
    if not os.path.exists(pt_path):
        return None
    pt = json.load(open(pt_path))
    mx = json.load(open(mx_path)) if os.path.exists(mx_path) else None
    out = {"large_square_tflops_torch": pt["large_square_tflops"],
           "large_square_tflops_mlx": mx["large_square_tflops"] if mx else None, "models": {}}
    for model, r in res.items():
        shapes = pt["shapes"].get(model, [])
        if len(shapes) < 2:
            continue
        mb_shape = shapes[1]
        rows_pt = {(c["projection"], c["product"], c["layout"]): c for c in pt["models"][model][mb_shape]}
        rows_mx = ({(c["projection"], c["product"], c["layout"]): c for c in mx["models"][model][mb_shape]}
                   if mx and model in mx["models"] else {})
        lst = []
        for c in r["components"]:
            name = c["component"]
            for prefix, (proj, _) in PRODUCT_OF.items():
                if name != prefix:
                    continue
                for d, prod in (("fwd", "base fwd"), ("bwd", "base bwd")):
                    layout = "layer" if (proj == "in_proj_qkv" and d == "bwd") else "plain"
                    p = rows_pt.get((proj, prod, layout))
                    m = rows_mx.get((proj, prod, layout))
                    if p is None:
                        continue
                    lst.append({"component": name, "direction": d, "in_situ_ms_per_micro_batch": round(c[f"{d}_ms"], 3),
                                "share_pct": round(100 * c[f"{d}_ms"] / r["per_micro_batch_ms_total"], 2),
                                "product": f"{proj} {prod}" + (" (the layer's layout)" if layout == "layer" else ""),
                                "torch_ms": p["events_ms"], "torch_tflops": p["tflops_events"],
                                "torch_of_large_square": p["of_large_square"],
                                "mlx_ms": m["wall_ms"] if m else None, "mlx_tflops": m["tflops_wall"] if m else None,
                                "torch_over_mlx_rate": round(p["tflops_events"] / m["tflops_wall"], 3) if m else None})
        lst.sort(key=lambda x: -x["in_situ_ms_per_micro_batch"])
        out["models"][model] = {"micro_batch_shape": mb_shape, "products": lst}
    return out


def main():
    res = {}
    for model, (name, d2m) in RUNS.items():
        p = os.path.join(LOGS, name)
        if os.path.exists(p):
            res[model] = report(model, p, d2m)
    pj = products_join(res)
    if pj:
        res["products_join"] = pj
    path = os.path.join(LOGS, "profile_results.json")
    json.dump(res, open(path, "w"), indent=1)
    for model, r in res.items():
        if model == "products_join":
            for m, v in r["models"].items():
                print(f"== products at {m} {v['micro_batch_shape']}")
                for p in v["products"][:12]:
                    print("  ", json.dumps(p))
            continue
        print(f"== {model}: {r['micro_batches_profiled']} micro-batches, steps {r['profiled_steps']}")
        print(" validity", json.dumps(r["validity"]))
        print(" new shapes", r["new_shape_cost"], "instrument", r["instrument_overhead_ms_per_micro_batch"], "ms/mb;",
              "mb A/B/C", r["mean_a_ms"], r["mean_b_ms"], r["mean_c_ms"], "total/mb", r["per_micro_batch_ms_total"],
              "per step", r["per_step_ms_total"])
        for g, v in r["groups"].items():
            print(f"  {g:24s} {v['ms_per_micro_batch']:9.2f} ms/mb {v['ms_per_step']:9.1f} ms/step {v['share_pct']:6.2f}%"
                  + (f"  predicted {v['predicted_pct']} {v['verdict']}" if "predicted_pct" in v else ""))
        for c in r["components"][:40]:
            print(f"   {c['component']:40s} f {c['fwd_ms']:8.2f} b {c['bwd_ms']:8.2f} = {c['ms']:8.2f} ms {100 * c['share']:5.2f}%"
                  f"  cpu {c['cpu_net_fwd_ms'] if c['cpu_net_fwd_ms'] is not None else float('nan'):7.2f}"
                  f"/{c['cpu_net_bwd_ms'] if c['cpu_net_bwd_ms'] is not None else float('nan'):7.2f}"
                  f"{'  DISPATCH-f' if c['dispatch_bound_fwd'] else ''}{'  DISPATCH-b' if c['dispatch_bound_bwd'] else ''}"
                  f"{'  WAITS-f' if c['gpu_waits_fwd'] else ''}{'  WAITS-b' if c['gpu_waits_bwd'] else ''}")
        print(" dispatch-bound", json.dumps(r["dispatch_bound"]))
        print(" outside D3-D5 >= 5%", json.dumps(r["outside_d3_d5_at_5pct"]), json.dumps(r["outside_d3_d5_groups_at_5pct"]))
        print(" shapes", json.dumps(r["micro_batch_shapes"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
