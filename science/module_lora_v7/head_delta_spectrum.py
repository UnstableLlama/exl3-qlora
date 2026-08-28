"""Is the trained head/embed delta real vocabulary learning, or logit calibration?

S64's amplification sweep says the module LoRA points in a useful direction and
stopped ~8x short of its own line-search optimum. Before spending a 2h v8 rerun
chasing that, distinguish two very different things it could be doing:

  (a) LOGIT CALIBRATION -- a low-rank, frequency-shaped prior correction. Would
      show up as a spectrum dominated by one or two singular directions, with
      per-token delta magnitude tracking unigram frequency. That win is mundane
      and reachable without retraining anything.

  (b) REAL VOCABULARY ADAPTATION -- a spread spectrum with no particular
      frequency alignment; different tokens moved for different reasons.

The trained delta is scale * (a @ b) with a: [d, 32], b: [32, V]. Rank <= 32, so
the exact SVD is cheap via QR -- never materialize the [6656, 202112] product:

    a = Qa Ra,  b^T = Qb Rb   ->   a@b = Qa (Ra Rb^T) Qb^T
    SVD the 32x32 middle; singular values are exact, singular vectors lift back
    through Qa / Qb.

Per-token column norms come from the same factorization: ||D[:,t]|| = ||M Qb[t]||
with M = Ra Rb^T, which is 202112 tiny matvecs done as one [V,32] @ [32,32].

Read-only, no model load. Unigram counts are tokenized off the v4 train set.
"""
import json, os, sys
import torch
from safetensors import safe_open

os.chdir("/home/unstable/exl3/private/exl3-qlora")
sys.path.insert(0, "/home/unstable/exl3/private/exl3-qlora")

MODEL = "/mnt/two/weights/Muse-Glimmer-30B-exl3-6.00bpw"
CKPT = "out/muse_glimmer_semancy_v7/checkpoint-00000160"
TRAIN = "mala-data/semancy_v4_train.jsonl"
SCALE = 5.65685424949238          # alpha/sqrt(r) = 32/sqrt(32), what v7 trained at
OUT = "out/muse_glimmer_v6_v7_comparison/head_delta_spectrum.json"

f = safe_open(os.path.join(CKPT, "lora_modules.safetensors"), "pt")
T = {k: f.get_tensor(k).float() for k in f.keys()}
report = {}


def factorize(a, b):
    """Exact rank-<=32 SVD of scale*(a@b) without forming the product.
    Returns (singular values, U in row-space of a, V in column-space of b)."""
    Qa, Ra = torch.linalg.qr(a)                 # a: [d, r]
    Qb, Rb = torch.linalg.qr(b.T)               # b^T: [V, r]
    M = SCALE * (Ra @ Rb.T)                     # [r, r]
    U, S, Vh = torch.linalg.svd(M)
    return S, Qa @ U, Qb @ Vh.T, M, Qb


def spectrum_report(name, S):
    tot = (S ** 2).sum()
    frac = [(S[:k] ** 2).sum().item() / tot.item() for k in (1, 2, 4, 8, 16, 32)]
    # participation ratio: effective number of directions carrying the energy
    p = (S ** 2) / tot
    pr = 1.0 / (p ** 2).sum().item()
    print(f"\n{name}: rank {len(S)}, Frobenius norm {S.norm():.4f}")
    print(f"  top singular values: {[round(v, 4) for v in S[:6].tolist()]}")
    print(f"  energy in top 1/2/4/8/16/32: " +
          " ".join(f"{x*100:.1f}%" for x in frac))
    print(f"  participation ratio (effective # directions): {pr:.2f} / {len(S)}")
    return {"frobenius": S.norm().item(), "singular_values": S.tolist(),
            "energy_top_1_2_4_8_16_32": frac, "participation_ratio": pr}


# ---- head delta: D = scale*(a@b), [6656, 202112]; columns are tokens ----
Sh, Uh, Vh_, Mh, Qbh = factorize(T["lm_head.lora_a"], T["lm_head.lora_b"])
report["head"] = spectrum_report("HEAD delta (logit-space)", Sh)

# per-token column norm, exact: ||D[:,t]|| = ||M Qb[t]||
head_tok_norm = (Qbh @ Mh.T).norm(dim=1)        # [V]

# ---- embed delta: D = scale*(a@b), [202048, 6656]; ROWS are tokens. Factor the
#      TRANSPOSE, D^T = scale*(b^T @ a^T) = [6656, 202048], so tokens are columns
#      again and the same per-column-norm path applies. Singular values are
#      identical either way. ----
Se, Ue, Ve, Me, Qbe = factorize(T["embed_tokens.lora_b"].T, T["embed_tokens.lora_a"].T)
report["embed"] = spectrum_report("EMBED delta", Se)
embed_tok_norm = (Qbe @ Me.T).norm(dim=1)       # [n_tok]

# ---- unigram frequency over the training data ----
from exllamav3 import Config, Tokenizer
tok = Tokenizer.from_config(Config.from_directory(MODEL))
V = head_tok_norm.numel()
counts = torch.zeros(V, dtype=torch.float)
n_rows = 0
with open(TRAIN) as fh:
    for line in fh:
        row = json.loads(line)
        text = "\n".join(m.get("content", "") for m in row.get("messages", []))
        ids = tok.encode(text, add_bos=False)[0]
        ids = ids[ids < V]
        counts.index_add_(0, ids.cpu().long(), torch.ones(ids.numel()))
        n_rows += 1
print(f"\n -- tokenized {n_rows} train rows, {int(counts.sum())} tokens, "
      f"{int((counts > 0).sum())} distinct of {V}")


def freq_analysis(name, norms, counts):
    n = norms[:counts.numel()].float()
    c = counts[:norms.numel()].float()
    seen = c > 0
    print(f"\n{name} vs unigram frequency:")
    print(f"  mean |delta| seen tokens {n[seen].mean():.5f}  "
          f"unseen {n[~seen].mean():.5f}  ratio {n[seen].mean()/n[~seen].mean():.2f}x")

    # Spearman over tokens actually present in the data
    def spearman(x, y):
        rx = x.argsort().argsort().float()
        ry = y.argsort().argsort().float()
        rx = rx - rx.mean(); ry = ry - ry.mean()
        return (rx @ ry / (rx.norm() * ry.norm())).item()

    rho = spearman(n[seen], c[seen])
    print(f"  Spearman(|delta|, count) over {int(seen.sum())} seen tokens: {rho:+.3f}")

    # decile table over seen tokens
    order = c[seen].argsort()
    ns, cs = n[seen][order], c[seen][order]
    k = len(ns) // 10
    print("  freq decile (low->high):  " +
          " ".join(f"{ns[i*k:(i+1)*k].mean():.4f}" for i in range(10)))
    return {"mean_seen": n[seen].mean().item(), "mean_unseen": n[~seen].mean().item(),
            "spearman_rho": rho,
            "deciles": [ns[i*k:(i+1)*k].mean().item() for i in range(10)]}


report["head_freq"] = freq_analysis("HEAD |delta| per token", head_tok_norm, counts)
report["embed_freq"] = freq_analysis("EMBED |delta| per token", embed_tok_norm, counts)
report["n_train_rows"] = n_rows
report["n_distinct_tokens"] = int((counts > 0).sum())

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nwrote {OUT}")
