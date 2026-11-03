"""The layer parts installed into transformers' Qwen3.5 modeling code (transformers.models.qwen3_5.modeling_qwen3_5).

Four methods are replaced, each only while the source of what it replaces (and of the functions its replacement reproduces)
hashes as in transformers 5.22.1, the version they were written against; on a mismatch the original stays and a warning names
it. Each replacement reads its part's setting (macai.layer.RMSNORM, ...) at every call and runs the original method when the
part is off or the inputs are outside what its kernels take:

    Qwen3_5RMSNorm.forward          -> macai.layer.RMSNORM
    Qwen3_5RMSNormGated.forward     -> macai.layer.GATED_NORM
    Qwen3_5MLP.forward              -> macai.layer.SWIGLU (the projections stay module calls, in the original order)
    Qwen3_5GatedDeltaNet.forward    -> macai.layer.CONV and macai.layer.GATES, for the training / prefill path without a cache
                                       (with a cache, or with packed rows, the original runs). The projections stay module
                                       calls in the original order (PEFT's LoRA layers and their dropout draws are unchanged);
                                       the rule is called through the module's own name for it (macai.patch swaps that name).
"""
import hashlib
import importlib
import inspect
import warnings

import torch
import torch.nn.functional as F

from . import _conv, _gated_norm, _gates, _rmsnorm, _swiglu

MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
WRITTEN_AGAINST = "5.22.1"
# sha256 (first 16 hex digits) of inspect.getsource of each method a patch replaces and of each function it reproduces, in
# transformers 5.22.1
SOURCES = {
    "Qwen3_5RMSNorm.forward": "350041feaa400633",
    "Qwen3_5RMSNorm._norm": "a1c27ed5fac56db4",
    "Qwen3_5RMSNormGated.forward": "5a4bd64da7fd08bd",
    "Qwen3_5MLP.forward": "03df2d7685cf8446",
    "Qwen3_5GatedDeltaNet.forward": "43718d6c3e7255e6",
    "causal_conv1d_fn": "f932149bdf42900c",
    "apply_mask_to_padding_states": "ac70fe1177ec07ff",
}
# each patch: the class, the method and the sources it depends on
PATCHES = {
    "rmsnorm": ("Qwen3_5RMSNorm", "forward", ("Qwen3_5RMSNorm.forward", "Qwen3_5RMSNorm._norm")),
    "gated_norm": ("Qwen3_5RMSNormGated", "forward", ("Qwen3_5RMSNormGated.forward",)),
    "mlp": ("Qwen3_5MLP", "forward", ("Qwen3_5MLP.forward",)),
    "gated_delta_net": ("Qwen3_5GatedDeltaNet", "forward",
                        ("Qwen3_5GatedDeltaNet.forward", "causal_conv1d_fn", "apply_mask_to_padding_states")),
}

_orig = {}            # patch name -> the class attribute it replaced
_inner = {}           # patch name -> the undecorated original (what a replacement falls back to)

# the parts' functions, looked up at every call (an instrument can wrap them: bench/profile_step.py)
conv_path = _conv.conv_qkv
gates_path = _gates.gates


def _settings():
    from macai import layer
    return layer


def source_hash(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def _resolve(mod, dotted: str):
    obj = mod
    for part in dotted.split("."):
        obj = getattr(obj, part) if not isinstance(obj, type) else obj.__dict__[part]
    return obj


def check_sources(mod) -> dict:
    """{source name: True when it hashes as in the version written against}."""
    out = {}
    for name, want in SOURCES.items():
        try:
            out[name] = source_hash(_resolve(mod, name)) == want
        except (AttributeError, KeyError, OSError, TypeError):
            out[name] = False
    return out


# ---- the replacements -------------------------------------------------------------------------------------------------

def rmsnorm_forward(self, x):
    if _settings().RMSNORM and _rmsnorm.supported(x, self.weight):
        return _rmsnorm.rms_norm(x, self.weight, self.eps)
    return _inner["rmsnorm"](self, x)


def gated_norm_forward(self, hidden_states, gate):
    if (_settings().GATED_NORM and getattr(self, "activation", None) == "silu"
            and _gated_norm.supported(hidden_states, gate, self.weight)):
        return _gated_norm.gated_rms_norm(hidden_states, gate, self.weight, self.variance_epsilon)
    return _inner["gated_norm"](self, hidden_states, gate)


def mlp_forward(self, x):
    if _settings().SWIGLU and getattr(getattr(self, "config", None), "hidden_act", None) == "silu":
        e = self.gate_proj(x)
        u = self.up_proj(x)
        if _swiglu.supported(e, u):
            return self.down_proj(_swiglu.swiglu(e, u))
        return self.down_proj(self.act_fn(e) * u)
    return _inner["mlp"](self, x)


def gated_delta_net_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    s = _settings()
    use_conv = (s.CONV and self.activation == "silu" and self.conv1d.bias is None
                and self.conv_kernel_size == _conv.K)
    if (cache_params is not None or kwargs.get("cu_seq_lens_q") is not None or not (use_conv or s.GATES)
            or hidden_states.device.type != "mps"):
        return _inner["gated_delta_net"](self, hidden_states, cache_params=cache_params, attention_mask=attention_mask,
                                         **kwargs)
    M = importlib.import_module(MODULE)
    hidden_states = M.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    # the projections in the original order (PEFT's dropout draws stay in the same order)
    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    rep = self.num_v_heads // self.num_k_heads
    if use_conv and _conv.supported(mixed_qkv, self.conv1d.weight, self.key_dim, self.head_k_dim, rep, self.head_v_dim):
        # q, k [B, T, H_v, D_k] (repeated over the value heads), v [B, T, H_v, D_v] float32: the rule's inputs as it reads them
        query, key, value = conv_path(mixed_qkv, self.conv1d.weight, self.key_dim, self.head_k_dim, rep, self.head_v_dim)
    else:
        mixed_qkv = M.causal_conv1d_fn(mixed_qkv.transpose(1, 2), self.conv1d.weight.squeeze(1), self.conv1d.bias,
                                       activation=self.activation, **kwargs)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
    if s.GATES and _gates.supported(b, a, self.A_log, self.dt_bias):
        beta, g = gates_path(b, a, self.A_log, self.dt_bias)
    else:
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    core_attn_out, _ = M.torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


_NEW = {"rmsnorm": rmsnorm_forward, "gated_norm": gated_norm_forward, "mlp": mlp_forward,
        "gated_delta_net": gated_delta_net_forward}


def install() -> list:
    """Replace the methods whose sources match; returns the patches in force. A mismatch leaves the original, with a warning."""
    try:
        mod = importlib.import_module(MODULE)
    except ImportError:
        return []
    ok = check_sources(mod)
    done = []
    for name, (cls_name, meth, deps) in PATCHES.items():
        cls = getattr(mod, cls_name, None)
        if name in _orig:
            done.append(name)
            continue
        bad = [d for d in deps if not ok.get(d)]
        if cls is None or bad:
            import transformers
            warnings.warn(f"macai.layer: {cls_name}.{meth} left as transformers' own: the source of {bad or [cls_name]} differs "
                          f"from transformers {WRITTEN_AGAINST}, the version its replacement was written against (installed: "
                          f"{transformers.__version__})", stacklevel=2)
            continue
        orig = cls.__dict__[meth]
        inner = inspect.unwrap(orig)
        new = _NEW[name]
        if inner is not orig and name == "gated_delta_net":
            # the original is wrapped by transformers' force_accelerate_hooks("conv1d") (it reads the convolution's weight
            # directly); wrap the replacement the same way and let its fallback call the unwrapped original
            new = mod.force_accelerate_hooks("conv1d")(new)
        _orig[name] = orig
        _inner[name] = inner
        setattr(cls, meth, new)
        done.append(name)
    return done


def uninstall() -> None:
    mod = importlib.import_module(MODULE)
    for name, orig in list(_orig.items()):
        cls_name, meth, _ = PATCHES[name]
        setattr(getattr(mod, cls_name), meth, orig)
        del _orig[name]
        _inner.pop(name, None)


def installed() -> list:
    return list(_orig)
