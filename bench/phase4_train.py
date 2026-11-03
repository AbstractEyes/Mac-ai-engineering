"""Series phase 4: LoRA training of the Qwen3.5 picture readers on the Mac's GPU through qwen-symbolic-thinking's own trainer
(qwen_symbolic_thinking.train.run), transformers' reference gated delta rule against macai's Metal kernels
(macai.patch.install() before the model loads). The tests, their predictions and their verdicts: docs/series_phase4.md.

    python bench/phase4_train.py --name p41_ref_a --model 0.8b --arm reference --epochs 4 --batch 4 --accum 4

Each run is its own process under a memory guard, one GPU job at a time, with the Metal allocator's watermarks set
(bench/phase4_run.sh). Writes bench/phase4_logs/<name>.json (also when the run fails: the error and its traceback).

Around the trainer the runner changes these things, the same way for both arms, and nothing else:
1. the model loads on the Apple GPU: the trainer's load_vlm(device_map="cuda") gets device_map="mps";
2. gradient checkpointing is off: the trainer calls model.gradient_checkpointing_enable() itself and passes
   gradient_checkpointing=True to TrainingArguments; the call becomes a no-op on the loaded model and the argument is forced
   to False (checked when training begins: no module checkpoints);
3. the loss is logged at every optimizer step: the trainer's logging_steps=10 is forced to 1;
4. the pictures load from the folder they were fetched into before (zip members from the offline Hub cache): the run refuses
   to start when one is missing, so nothing is downloaded; the run's own outputs (the adapter) go to <data>/phase4;
5. a callback records each step's time (after the Trainer's logging synchronizes the GPU), the Trainer's whole log (loss,
   gradient norm, learning rate) and the memory: the Metal allocator's figures and the process's physical footprint at each
   step, and (a forward hook on the model) the allocator's live tensors when each micro-batch's forward returns.
"""
import argparse
import ctypes
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

MODELS = {"0.8b": "Qwen/Qwen3.5-0.8B", "9b": "Qwen/Qwen3.5-9B"}
# the readers' recipe: LoRA rank 32, alpha 64, dropout 0.05; plain Adam at 2e-4 under the trainer's cosine schedule; bf16;
# sequences up to 2,048 tokens; the seed of the published runs; nothing shipped
RECIPE = dict(rank=32, alpha=64, dropout=0.05, lr=2e-4, max_len=2048, seed=20261030, do_ship=False)
EXAMPLES = 400         # the sample's examples (382 items; 18 carry two forms): checked before training
WARMUP_STEPS = 10      # the steady time a step is measured after this many optimizer steps, the same window in every run
MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
SAMPLE = "sample_every55.json"


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
        "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
        "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint_gb() -> float:
    """This process's physical footprint (the figure Activity Monitor shows; on Apple silicon it includes the GPU buffers the
    process owns): the guard's measure, read here at each step."""
    buf = _RusageInfoV2()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
        return -1.0
    return round(buf.ri_phys_footprint / 2 ** 30, 3)


def git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_pictures(sample: str, images: str, n_items) -> dict:
    """Every picture the run uses must already be on disk: url pictures in the fetched folder (<sample_id>.img), Hub files in
    the offline cache. Raises when one is missing, so the trainer downloads nothing."""
    from huggingface_hub import try_to_load_from_cache
    from qwen_symbolic_thinking.extract import locator_row
    with open(sample, encoding="utf-8") as fh:
        items = json.load(fh)["items"][:n_items]
    kinds, missing = {}, []
    for it in items:
        if it["kind"] != "image":
            continue
        r = locator_row(it)
        loc = r.get("image") or {}
        kind = loc.get("kind", "pending")
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "url":
            ok = os.path.exists(os.path.join(images, r["sample_id"] + ".img"))
        elif kind in ("zip_member", "hf_file", "hf_parquet"):
            ok = isinstance(try_to_load_from_cache(loc["repo"], loc["path"], repo_type=loc.get("repo_type", "dataset")), str)
        else:
            ok = False
        if not ok:
            missing.append(f"{r['sample_id']} ({kind})")
    if missing:
        raise RuntimeError(f"{len(missing)} pictures are not on disk (first {missing[:5]}); the run would download them")
    return {"items": len(items), "pictures_by_locator": kinds}


def steady(by_step: dict, warmup: int = WARMUP_STEPS):
    """Seconds a step from a {step: elapsed seconds} clock, between the end of step `warmup` and the last step."""
    last = max(by_step) if by_step else 0
    if warmup not in by_step or last <= warmup:
        return None
    return round((by_step[last] - by_step[warmup]) / (last - warmup), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--arm", choices=("reference", "kernels"), required=True)
    ap.add_argument("--epochs", type=float, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--n-items", type=int, default=None, help="only the sample's first n items (smoke runs)")
    ap.add_argument("--count-calls", action="store_true",
                    help="count the calls of the model's gated delta rule (smoke runs; adds one Python call a layer)")
    ap.add_argument("--data", default=os.path.expanduser("~/mirel/data/mac_bench/reader_train"),
                    help="the folder holding the sample file, its fetched pictures (out/images) and the runs' outputs (phase4)")
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "phase4_logs"), exist_ok=True)
    out_json = os.path.join(HERE, "phase4_logs", f"{a.name}.json")
    rec = {"name": a.name, "model_id": MODELS[a.model], "arm": a.arm, "epochs": a.epochs, "batch": a.batch, "grad_accum": a.accum,
           "sequences_a_step": a.batch * a.accum, "n_items": a.n_items, "recipe": RECIPE, "warmup_steps_excluded": WARMUP_STEPS,
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "repo_commit": git("rev-parse", "--short", "HEAD"),
           "repo_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "env": {k: os.environ.get(k) for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                                                  "PYTORCH_ENABLE_MPS_FALLBACK", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")},
           "runner_patches": ["load_vlm device_map cuda -> mps", "model.gradient_checkpointing_enable -> no-op",
                              "TrainingArguments gradient_checkpointing -> False", "TrainingArguments logging_steps 10 -> 1",
                              "fetch_images -> the fetched pictures folder"],
           "steps": [], "final_logs": []}
    t_start = time.time()
    rc = 0
    calls = [0]
    try:
        import torch
        import transformers
        import peft
        import qwen_symbolic_thinking
        if a.arm == "kernels":
            import macai
            import macai.patch
            from macai.gated_delta import _chunked, op
            rec["patched_modules"] = macai.patch.install()          # before the model loads
            assert MODULE in rec["patched_modules"], rec["patched_modules"]
            rec["macai"] = {"version": macai.__version__, "KERNEL": op.KERNEL, "KERNEL_BWD": op.KERNEL_BWD,
                            "FUSED_NORM": op.FUSED_NORM, "chunk": _chunked.C}
        M = importlib.import_module(MODULE)
        fn = M.torch_chunk_gated_delta_rule
        rec["gated_delta_rule"] = f"{fn.__module__}.{fn.__qualname__}"
        if a.arm == "kernels":
            from macai.gated_delta import chunk_gated_delta_rule
            assert fn is chunk_gated_delta_rule
        else:
            assert fn.__module__ == MODULE, rec["gated_delta_rule"]
        if a.count_calls:
            def counted(*x, **kw):
                calls[0] += 1
                return fn(*x, **kw)
            M.torch_chunk_gated_delta_rule = counted
        rec["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                           "qwen_symbolic_thinking": qwen_symbolic_thinking.__version__, "python": sys.version.split()[0]}
        rec["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 2 ** 30, 2)

        import qwen_symbolic_thinking.train as T
        sample = os.path.join(a.data, SAMPLE)
        images = os.path.join(a.data, "out", "images")
        rec["sample"] = sample
        rec["pictures"] = check_pictures(sample, images, a.n_items)
        orig_load, orig_fetch = T.load_vlm, T.fetch_images

        def load_on_mps(src, **kw):
            model = orig_load(src, **{**kw, "device_map": "mps"})          # 1. the trainer's one CUDA literal
            model.gradient_checkpointing_enable = lambda *x, **y: None      # 2. checkpointing off (the trainer's own call)
            return model

        T.load_vlm = load_on_mps
        T.fetch_images = lambda rows, cache_dir, **kw: orig_fetch(rows, images, **kw)   # 4. the fetched pictures

        class Args(transformers.TrainingArguments):
            def __post_init__(self):
                self.logging_steps = 1               # 3. the loss at every optimizer step
                self.gradient_checkpointing = False  # 2. checkpointing off (the Trainer's call)
                super().__post_init__()

        class Recorder(transformers.TrainerCallback):
            """5. each step: time since training began (taken in on_log, after the Trainer's logging has read the loss off the
            GPU), the Trainer's log, memory."""

            def __init__(self):
                self.t0 = None
                self.live = self.driver = 0

            def at_forward_end(self, module, inputs, output):
                """The allocator's live tensors when a micro-batch's forward returns (every activation saved for the backward, the
                logits and the loss alive): the step's memory need, which the footprint hides under the allocator's cache."""
                self.live = max(self.live, torch.mps.current_allocated_memory())
                self.driver = max(self.driver, torch.mps.driver_allocated_memory())

            def on_train_begin(self, args, state, control, model=None, **kw):
                self.t0 = time.time()
                model.register_forward_hook(self.at_forward_end)
                ckpt = [n for n, m in model.named_modules() if getattr(m, "gradient_checkpointing", False) is True]
                rec["checks"] = {
                    "modules_checkpointing": ckpt[:5], "logging_steps": args.logging_steps,
                    "args_gradient_checkpointing": args.gradient_checkpointing,
                    "linear_attention_layers": sum(type(m).__name__ == "Qwen3_5GatedDeltaNet" for m in model.modules()),
                    "gated_delta_rule_at_train_begin": (lambda f: f"{f.__module__}.{f.__qualname__}")(M.torch_chunk_gated_delta_rule),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "model_dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device)}
                # the LoRA A matrices as initialized (B starts at zero): whether two runs of the same seed start alike
                lora_a = [p for n, p in model.named_parameters() if "lora_A" in n]
                rec["checks"]["lora_A_abs_sum"] = float(sum(p.detach().float().abs().sum() for p in lora_a))
                rec["checks"]["lora_A_first_values"] = [round(x, 6) for x in lora_a[0].detach().float().flatten()[:4].tolist()]
                rec["checks"]["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
                assert not ckpt, f"modules still checkpoint: {ckpt[:5]}"
                assert args.logging_steps == 1 and not args.gradient_checkpointing
                print(f"[phase4] checks {json.dumps(rec['checks'])}", flush=True)

            def on_log(self, args, state, control, logs=None, **kw):
                logs = dict(logs or {})
                if "loss" not in logs:
                    rec["final_logs"].append(logs)
                    return
                row = {"step": state.global_step, "t_s": round(time.time() - self.t0, 3)}
                row.update({k: logs[k] for k in ("loss", "grad_norm", "learning_rate", "epoch") if k in logs})
                row.update(mps_alloc_gb=round(torch.mps.current_allocated_memory() / 2 ** 30, 3),
                           mps_driver_gb=round(torch.mps.driver_allocated_memory() / 2 ** 30, 3), footprint_gb=footprint_gb(),
                           live_at_forward_end_gb=round(self.live / 2 ** 30, 3),
                           driver_at_forward_end_gb=round(self.driver / 2 ** 30, 3))
                self.live = self.driver = 0
                rec["steps"].append(row)

        recorder = Recorder()

        class Trainer(transformers.Trainer):
            def __init__(self, *x, **kw):
                kw["callbacks"] = list(kw.get("callbacks") or []) + [recorder]
                super().__init__(*x, **kw)
                n = len(kw["train_dataset"])
                rec["examples"] = n
                rec["accelerator"] = {"mixed_precision": str(self.accelerator.mixed_precision),
                                      "native_amp": bool(getattr(self.accelerator, "native_amp", False)),
                                      "device": str(self.args.device)}
                if a.n_items is None:
                    assert n == EXAMPLES, f"the sample gives {n} examples, not the registered {EXAMPLES}"

        # the trainer imports both from transformers inside run(). transformers' lazy module replaces itself in sys.modules at
        # its first name lookup (5.22.1), so the names are set on the module object that is current now, and checked
        tf = sys.modules["transformers"]
        tf.TrainingArguments, tf.Trainer = Args, Trainer
        from transformers import Trainer as _trainer, TrainingArguments as _args
        assert _args is Args and _trainer is Trainer, "the TrainingArguments / Trainer patch did not take"
        summ = T.run(model_id=MODELS[a.model], inputs=[sample], out_dir=os.path.join(a.data, "phase4"), epochs=a.epochs,
                     batch=a.batch, grad_accum=a.accum, n_items=a.n_items, tag=a.name, **RECIPE)
        assert "checks" in rec and rec["steps"], "the recorder did not run: the Trainer patch did not take inside the trainer"
        tlog = [r for r in summ.get("log", []) if "loss" in r]
        rec["trainer"] = {k: summ.get(k) for k in ("package", "steps", "train_loss", "s_per_step", "train_min", "items",
                                                     "items_with_picture", "examples", "examples_by_form", "collated_examples",
                                                     "examples_cut_at_max_len", "longest_example_tokens", "lora_modules",
                                                     "adapter_dir")}
        rec["trainer_log"] = summ.get("log", [])
        rec["s_per_step_steady_trainer_clock"] = steady({r["step"]: r["spent_s"] for r in tlog})
        rec["s_per_step_steady_runner_clock"] = steady({r["step"]: r["t_s"] for r in rec["steps"]})
        rec["loss_final_step"] = rec["steps"][-1]["loss"] if rec["steps"] else None
        rec["mps_driver_peak_gb"] = max((r["mps_driver_gb"] for r in rec["steps"]), default=None)
        rec["live_at_forward_end_peak_gb"] = max((r["live_at_forward_end_gb"] for r in rec["steps"]), default=None)
        rec["footprint_peak_at_steps_gb"] = max((r["footprint_gb"] for r in rec["steps"]), default=None)
        print(f"[phase4] {a.name}: {summ.get('steps')} steps, final loss {rec['loss_final_step']}, steady "
              f"{rec['s_per_step_steady_trainer_clock']} s a step (trainer clock, steps {WARMUP_STEPS + 1}-{summ.get('steps')})",
              flush=True)
    except BaseException as ex:  # noqa: BLE001 - the failure is the record
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:600]}"
        rec["traceback"] = traceback.format_exc()[-4000:]
        print(f"[phase4] {a.name} FAILED: {rec['error']}", flush=True)
        rc = 1
    finally:
        if a.count_calls:
            rec["gated_delta_rule_calls"] = calls[0]
        rec["wall_s"] = round(time.time() - t_start, 1)
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[phase4] wrote {out_json}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
