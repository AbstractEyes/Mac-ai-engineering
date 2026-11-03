"""Float64 CPU check of the chunked forward (F1-F5) and the hand-derived backward (B1-B6) in
docs/prior_art_gated_delta.md against autograd through the token recurrence. Tiny shapes, CPU only."""
import torch

torch.manual_seed(0)
torch.set_num_threads(1)
dt = torch.float64
T, C, DK, DV = 21, 8, 6, 5          # a tail chunk of 5 rows


def recurrence(q, k, v, g, b, S0):
    S = S0
    outs = []
    for t in range(T):
        St = torch.exp(g[t]) * S
        d = b[t] * (v[t] - St.T @ k[t])
        S = St + torch.outer(k[t], d)
        outs.append(S.T @ q[t])
    return torch.stack(outs), S


def chunked(q, k, v, g, b, S0, dO, dSfin):
    NT = (T + C - 1) // C
    S = S0.clone()
    O = torch.zeros(T, DV, dtype=dt)
    saved = []
    for c in range(NT):
        sl = slice(c * C, min((c + 1) * C, T))
        Q, K, V, gc, bc = q[sl], k[sl], v[sl], g[sl], b[sl]
        n = Q.shape[0]
        G = torch.cumsum(gc, 0)                                     # F1
        gam = torch.exp(G)
        e = torch.exp(G[-1] - G)
        diff = G[:, None] - G[None, :]
        low = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        lowi = torch.tril(torch.ones(n, n, dtype=torch.bool), 0)
        Gam = torch.where(lowi, torch.exp(torch.where(lowi, diff, torch.zeros_like(diff))), torch.zeros_like(diff))
        Kg = K @ K.T                                                # F2
        L = torch.where(low, bc[:, None] * Kg * Gam, torch.zeros_like(Kg))
        A = torch.linalg.inv(torch.eye(n, dtype=dt) + L)
        W = A @ ((bc * gam)[:, None] * K)                           # F3
        U = A @ (bc[:, None] * V)
        Vn = U - W @ S                                              # F4
        M = Gam * (Q @ K.T)                                         # F5
        O[sl] = gam[:, None] * (Q @ S) + M @ Vn
        saved.append((sl, Q, K, V, bc, G, gam, e, Gam, low, Kg, L, A, W, Vn, M, S.clone()))
        S = gam[-1] * S + K.T @ (e[:, None] * Vn)
    # backward B1-B6
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dg, db = torch.zeros_like(g), torch.zeros_like(b)
    dSn = dSfin.clone()
    for (sl, Q, K, V, bc, G, gam, e, Gam, low, Kg, L, A, W, Vn, M, Sc) in reversed(saved):
        dOc = dO[sl]
        dVn = M.T @ dOc + e[:, None] * (K @ dSn)                     # B1
        dSc = gam[-1] * dSn + (gam[:, None] * Q).T @ dOc - W.T @ dVn  # B2
        dM = Gam * (dOc @ Vn.T)                                     # B3
        dQ = gam[:, None] * (dOc @ Sc.T) + dM @ K
        dKs = e[:, None] * (Vn @ dSn.T)
        dK = dM.T @ Q + dKs
        dW = -dVn @ Sc.T
        dU = dVn
        dG = (Q * dQ).sum(1) - (K * dK).sum(1)                      # B4
        dG[-1] += gam[-1] * (Sc * dSn).sum() + (K * dKs).sum()
        X, Y = A.T @ dW, A.T @ dU                                   # B5
        dK = dK + (bc * gam)[:, None] * X
        dbc = gam * (K * X).sum(1) + (V * Y).sum(1)
        dG = dG + bc * gam * (K * X).sum(1)
        dV = bc[:, None] * Y
        dA = torch.where(low, dW @ ((bc * gam)[:, None] * K).T + dU @ (bc[:, None] * V).T, torch.zeros_like(Kg))
        dL = torch.where(low, -A.T @ dA @ A.T, torch.zeros_like(Kg))
        P = dL * Gam
        dK = dK + bc[:, None] * (P @ K) + P.T @ (bc[:, None] * K)
        dbc = dbc + (P * Kg).sum(1)
        R = dL * L
        dG = dG + R.sum(1) - R.sum(0)
        dgc = torch.flip(torch.cumsum(torch.flip(dG, [0]), 0), [0])  # B6
        dq[sl], dk[sl], dv[sl], dg[sl], db[sl] = dQ, dK, dV, dgc, dbc
        dSn = dSc
    return O, S, (dq, dk, dv, dg, db, dSn)


q = torch.nn.functional.normalize(torch.randn(T, DK, dtype=dt), dim=-1) * DK ** -0.5
k = torch.nn.functional.normalize(torch.randn(T, DK, dtype=dt), dim=-1)
v = torch.randn(T, DV, dtype=dt)
g = -torch.rand(T, dtype=dt) * 2.0
b = torch.rand(T, dtype=dt)
S0 = torch.randn(DK, DV, dtype=dt)
dO = torch.randn(T, DV, dtype=dt)
dSfin = torch.randn(DK, DV, dtype=dt)

leaves = [x.clone().requires_grad_(True) for x in (q, k, v, g, b, S0)]
Oref, Sref = recurrence(*leaves)
grads = torch.autograd.grad((Oref * dO).sum() + (Sref * dSfin).sum(), leaves)
O, S, mine = chunked(q, k, v, g, b, S0, dO, dSfin)
print("forward O max err", (O - Oref).abs().max().item(), " final S max err", (S - Sref).abs().max().item())
for name, a, r in zip(("dq", "dk", "dv", "dg", "dbeta", "dS0"), mine, grads):
    print(f"{name:6s} max err {(a - r).abs().max().item():.3e}   (scale {r.abs().max().item():.3e})")
