"""The linear-attention layer's gates as one Metal pass each way:

    beta = sigmoid(b)                                        (float32)
    g    = -exp(A_log) * softplus(a + dt_bias)               (float32; softplus with PyTorch's threshold of 20)

from the [B, T, H_v] outputs of in_proj_b and in_proj_a, written in float32 (the rule's drop-in reads both as float32, so
neither is cast or copied again). transformers runs this as about eight PyTorch operations each way and rounds beta to the
input dtype first. The backward:

    db = dbeta beta (1 - beta)
    da = dg (-exp(A_log)) sigmoid(a + dt_bias)               (1 above the threshold, as PyTorch's softplus backward)

A_log's and dt_bias's gradients only when they need one (PyTorch's reductions over the tokens: no float atomics).
"""
import torch

from . import _lib

_SOURCE = r"""
// log(1 + y) for y >= 0, accurate where 1 + y rounds (the classic correction: y log(u) / (u - 1), u = 1 + y)
inline float log1p_acc(float y) {
  const float u = 1.0f + y;
  return (u == 1.0f) ? y : log(u) * (y / (u - 1.0f));
}

template <typename T, typename TW>
inline void gates_fwd_impl(const device T* b, const device T* a, const device TW* alog, const device TW* dtb,
                           device float* beta, device float* g, uint n, uint H, uint i) {
  if (i >= n) return;
  const uint h = i % H;
  beta[i] = 1.0f / (1.0f + exp(-float(b[i])));
  const float xa = float(a[i]) + float(dtb[h]);
  const float sp = (xa > 20.0f) ? xa : log1p_acc(exp(xa));
  g[i] = -exp(float(alog[h])) * sp;
}

template <typename T, typename TW>
inline void gates_bwd_impl(const device T* b, const device T* a, const device TW* alog, const device TW* dtb,
                           const device float* dbeta, const device float* dg, device T* db, device T* da,
                           uint n, uint H, uint i) {
  if (i >= n) return;
  const uint h = i % H;
  const float sb = 1.0f / (1.0f + exp(-float(b[i])));
  db[i] = T(dbeta[i] * sb * (1.0f - sb));
  const float xa = float(a[i]) + float(dtb[h]);
  const float sx = (xa > 20.0f) ? 1.0f : 1.0f / (1.0f + exp(-xa));
  da[i] = T(dg[i] * (-exp(float(alog[h]))) * sx);
}

#define GATES(T, TW, SUF)                                                                                              \
kernel void gates_fwd_##SUF(device const T* b [[buffer(0)]], device const T* a [[buffer(1)]],                          \
                            device const TW* alog [[buffer(2)]], device const TW* dtb [[buffer(3)]],                     \
                            device float* beta [[buffer(4)]], device float* g [[buffer(5)]], constant uint& n [[buffer(6)]], \
                            constant uint& H [[buffer(7)]], uint i [[thread_position_in_grid]]) {                       \
  gates_fwd_impl<T, TW>(b, a, alog, dtb, beta, g, n, H, i);                                                            \
}                                                                                                                      \
kernel void gates_bwd_##SUF(device const T* b [[buffer(0)]], device const T* a [[buffer(1)]],                          \
                            device const TW* alog [[buffer(2)]], device const TW* dtb [[buffer(3)]],                     \
                            device const float* dbeta [[buffer(4)]], device const float* dg [[buffer(5)]],               \
                            device T* db [[buffer(6)]], device T* da [[buffer(7)]], constant uint& n [[buffer(8)]],       \
                            constant uint& H [[buffer(9)]], uint i [[thread_position_in_grid]]) {                       \
  gates_bwd_impl<T, TW>(b, a, alog, dtb, dbeta, dg, db, da, n, H, i);                                                  \
}

__INSTANCES__
"""

GROUP = 256


def library():
    return _lib.library(_SOURCE.replace("__INSTANCES__", _lib.instances("GATES", True)), ())


def supported(b, a, A_log, dt_bias) -> bool:
    return (_lib.supported(b, a, A_log, dt_bias) and b.dtype == a.dtype and b.shape == a.shape and A_log.dtype == dt_bias.dtype
            and A_log.dim() == 1 and dt_bias.shape == A_log.shape and b.shape[-1] == A_log.shape[0] and b.numel() < 2 ** 31)


class Gates(torch.autograd.Function):
    """(beta, g) = (sigmoid(b), -exp(A_log) softplus(a + dt_bias)), both float32."""

    @staticmethod
    def forward(ctx, b, a, A_log, dt_bias):
        bc, ac = b.contiguous(), a.contiguous()
        al, dtb = A_log.contiguous(), dt_bias.contiguous()
        n, H = bc.numel(), A_log.shape[0]
        beta = torch.empty(bc.shape, dtype=torch.float32, device=b.device)
        g = torch.empty(bc.shape, dtype=torch.float32, device=b.device)
        if n:
            _lib.kernel(library(), "gates_fwd", b.dtype, A_log.dtype)(
                bc, ac, al, dtb, beta, g, n, H, threads=(n, 1, 1), group_size=(GROUP, 1, 1))
        ctx.save_for_backward(bc, ac, al, dtb)
        return beta, g

    @staticmethod
    def backward(ctx, dbeta, dg):
        bc, ac, al, dtb = ctx.saved_tensors
        n, H = bc.numel(), al.shape[0]
        dev = bc.device
        dbeta = torch.zeros(bc.shape, dtype=torch.float32, device=dev) if dbeta is None else dbeta.float().contiguous()
        dg = torch.zeros(bc.shape, dtype=torch.float32, device=dev) if dg is None else dg.float().contiguous()
        db, da = torch.empty_like(bc), torch.empty_like(ac)
        if n:
            _lib.kernel(library(), "gates_bwd", bc.dtype, al.dtype)(
                bc, ac, al, dtb, dbeta, dg, db, da, n, H, threads=(n, 1, 1), group_size=(GROUP, 1, 1))
        dA = ddt = None
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            # trained gates (not the LoRA recipe): float32 PyTorch reductions over the tokens
            x = ac.float() + dtb.float()
            dsp = dg * (-al.float().exp()) * torch.where(x > 20, torch.ones_like(x), torch.sigmoid(x))
            if ctx.needs_input_grad[2]:
                g = -al.float().exp() * torch.nn.functional.softplus(x)
                dA = (dg * g).reshape(-1, H).sum(0).to(al.dtype)
            if ctx.needs_input_grad[3]:
                ddt = dsp.reshape(-1, H).sum(0).to(dtb.dtype)
        return db, da, dA, ddt


def gates(b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    return Gates.apply(b, a, A_log, dt_bias)
