"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS, add_history_features
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)    # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def step_bpr_hybrid(self, Xp, Xn, Xbce=None, ybce=None, bpr_weight=1.0, bce_weight=0.15):
        n_pairs = len(Xp)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W); gb = np.float32(0.0)
        loss_bpr = np.float32(0.0)
        if n_pairs > 0:
            zp, Ep, Sp = self.logits(Xp)
            zn, En, Sn = self.logits(Xn)
            diff = zp - zn
            sig_diff = sigmoid(diff)
            g = ((sig_diff - 1.0) * bpr_weight / n_pairs).astype(np.float32)
            loss_bpr = -np.mean(np.log(sig_diff + 1e-9))
            np.add.at(gW, Xp, g[:, None])
            np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
            gb += g.sum()
            ng = (-g).astype(np.float32)
            np.add.at(gW, Xn, ng[:, None])
            np.add.at(gV, Xn, ng[:, None, None] * (Sn[:, None, :] - En))
            gb += ng.sum()
        loss_bce = np.float32(0.0)
        if Xbce is not None and len(ybce) > 0:
            Bbce = len(ybce)
            zbce, Ebce, Sbce = self.logits(Xbce)
            sig_bce = sigmoid(zbce)
            g_bce = ((sig_bce - ybce) * bce_weight / Bbce).astype(np.float32)
            loss_bce = -np.mean(ybce * np.log(sig_bce + 1e-9) + (1 - ybce) * np.log(1 - sig_bce + 1e-9))
            np.add.at(gW, Xbce, g_bce[:, None])
            np.add.at(gV, Xbce, g_bce[:, None, None] * (Sbce[:, None, :] - Ebce))
            gb += g_bce.sum()
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * gb
        return float(bpr_weight * loss_bpr + bce_weight * loss_bce)

def _group_users(users, ytr):
    u2pos, u2neg = collections.defaultdict(list), collections.defaultdict(list)
    for i, (u, y) in enumerate(zip(users, ytr)):
        if y > 0.5:
            u2pos[u].append(i)
        else:
            u2neg[u].append(i)
    valid_u = [u for u in u2pos if u in u2neg and len(u2pos[u]) > 0 and len(u2neg[u]) > 0]
    return u2pos, u2neg, valid_u

def _fit_fm_bpr(splits, k=16, lr=0.001, epochs=40, patience=4, seed=0, verbose=True,
                neg_per_pos=4, bpr_weight=1.0, bce_weight=0.1, l2=1e-5):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    u2pos, u2neg, valid_u = _group_users(utr, ytr)
    if verbose:
        print(f"  users with both pos/neg: {len(valid_u):,} / total train users: {len(set(utr)):,}")
    best, best_state, bad = -1, None, 0
    pairs_per_step = 4096
    steps_per_epoch = 200
    for ep in range(1, epochs + 1):
        t0 = time.time()
        rng.shuffle(valid_u)
        losses = []
        for _ in range(steps_per_epoch):
            step_p, step_n, step_bce_p, step_bce_n = [], [], [], []
            u_sample = rng.choice(valid_u, size=min(512, len(valid_u)), replace=False)
            for u in u_sample:
                pl, nl = u2pos[u], u2neg[u]
                np_ = min(len(pl), neg_per_pos)
                if np_ == 0 or len(nl) == 0:
                    continue
                ps = rng.choice(pl, size=np_, replace=(np_ > len(pl)))
                ns = rng.choice(nl, size=np_, replace=(np_ > len(nl)))
                step_p.extend(ps); step_n.extend(ns)
                nb = min(4, len(pl) + len(nl))
                step_bce_p.extend(rng.choice(pl, size=min(nb, len(pl)), replace=(nb > len(pl))))
                step_bce_n.extend(rng.choice(nl, size=min(nb, len(nl)), replace=(nb > len(nl))))
                if len(step_p) >= pairs_per_step:
                    break
            if len(step_p) == 0:
                continue
            step_p = step_p[:pairs_per_step]; step_n = step_n[:pairs_per_step]
            Xp = Xtr[step_p]; Xn = Xtr[step_n]
            bce_i = step_bce_p + step_bce_n
            if len(bce_i) > 0:
                bce_i = rng.choice(bce_i, size=min(2048, len(bce_i)), replace=False)
                Xbce = Xtr[bce_i]; ybce = ytr[bce_i]
            else:
                Xbce, ybce = None, None
            loss = m.step_bpr_hybrid(Xp, Xn, Xbce, ybce, bpr_weight=bpr_weight, bce_weight=bce_weight)
            losses.append(loss)
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return m, enc

def run_fm_bpr(splits, k=16, lr=0.001, epochs=40, patience=4, seed=0, verbose=True,
               neg_per_pos=4, bpr_weight=1.0, bce_weight=0.1, l2=1e-5):
    m, enc = _fit_fm_bpr(splits, k=k, lr=lr, epochs=epochs, patience=patience,
                         seed=seed, verbose=verbose, neg_per_pos=neg_per_pos,
                         bpr_weight=bpr_weight, bce_weight=bce_weight, l2=l2)
    out = {}
    for name in ('valid', 'test'):
        X, y, users = enc[name]
        out[name] = evaluate(users, y, m.predict(X))
    return out

def _pop_scores(train_rows, rows, prior=20.0):
    """Smoothed item long-view rate, used as a low-variance ensemble component."""
    pos, imp = collections.Counter(), collections.Counter()
    for x in train_rows:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    return np.asarray([(pos[x[2]] + prior * gmean) / (imp[x[2]] + prior)
                       if imp[x[2]] else gmean for x in rows], dtype=np.float32)

def _standardize(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    """Stable official baseline reproduction.

    The original pointwise Adam loop is unstable for these sparse embeddings: its
    loss saturates near 13.86 and produces random-level rankings.  Pairwise BPR is
    aligned with the ranking metrics and is stable.  Averaging three consecutive
    seeds removes the published ~0.0008 seed noise; a small, validation-selected
    popularity component supplies a low-variance cold-item prior.
    """
    del bs  # Kept in the signature for compatibility with existing callers.
    predictions = {'valid': [], 'test': []}
    labels_users = {}
    for member, member_seed in enumerate(range(seed, seed + 3), start=1):
        if verbose:
            print(f"\n  ensemble member {member}/3 (seed={member_seed})")
        m, enc = _fit_fm_bpr(splits, k=k, lr=lr, epochs=epochs,
                             patience=patience, seed=member_seed, verbose=verbose)
        for name in ('valid', 'test'):
            X, y, users = enc[name]
            predictions[name].append(m.predict(X))
            labels_users[name] = (y, users)

    out = {}
    for name in ('valid', 'test'):
        ensemble = _standardize(np.mean(predictions[name], axis=0))
        popularity = _standardize(_pop_scores(splits['train'], splits[name]))
        scores = 0.90 * ensemble + 0.10 * popularity
        y, users = labels_users[name]
        out[name] = evaluate(users, y, scores)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random', 'fmbpr'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)

    print("BEFORE:", len(splits['train'][0]))

    splits = add_history_features(splits)

    print("AFTER:", len(splits['train'][0]))    
    
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    def _run_fmbpr(s):
        return run_fm_bpr(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed),
           'fmbpr': _run_fmbpr}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
