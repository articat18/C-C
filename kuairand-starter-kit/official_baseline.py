"""Organizer-provided KuaiRand-Pure pointwise FM reference.

The model and training configuration are preserved separately from experiment
candidates.  The logits implementation uses non-mutating dot products so the
reference remains reliable on supported Python and NumPy versions.  Automated
experiments must not edit this file.

The reproduction path intentionally evaluates validation only.  Hidden-test
evaluation remains behind the final approval gate.
"""

import argparse
import time

import numpy as np

from data import FIELDS, encode, load
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class OfficialFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(1)
        interaction = 0.5 * (
            np.einsum("ij,ij->i", summed, summed)
            - np.einsum("ijk,ijk->i", embeddings, embeddings)
        )
        return self.b + self.W[X].sum(1) + interaction, embeddings, summed

    def step(self, X, y):
        batch_size = len(y)
        logits, embeddings, summed = self.logits(X)
        gradient = ((sigmoid(logits) - y) / batch_size).astype(np.float32)
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X, gradient[:, None])
        np.add.at(
            grad_v,
            X,
            gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        self.t += 1
        beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
        parameters = (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        )
        for parameter, grad, first_moment, second_moment in parameters:
            first_moment *= beta_1
            first_moment += (1 - beta_1) * grad
            second_moment *= beta_2
            second_moment += (1 - beta_2) * (grad * grad)
            parameter -= self.lr * (
                first_moment / (1 - beta_1 ** self.t)
            ) / (
                np.sqrt(second_moment / (1 - beta_2 ** self.t)) + epsilon
            )
        self.b -= self.lr * gradient.sum()
        probabilities = sigmoid(logits)
        return float(
            -np.mean(
                y * np.log(probabilities + 1e-9)
                + (1 - y) * np.log(1 - probabilities + 1e-9)
            )
        )

    def predict(self, X, batch_size=200_000):
        return np.concatenate(
            [
                self.logits(X[i:i + batch_size])[0]
                for i in range(0, len(X), batch_size)
            ]
        )


def run_official_fm(splits, k=16, lr=0.001, epochs=40, batch_size=8192,
                    patience=4, seed=0, verbose=True):
    """Train the official FM and return validation metrics only."""

    development_splits = {
        'train': splits['train'],
        'valid': splits['valid'],
        'test': [],
    }
    encoded, dimension = encode(development_splits)
    X_train, y_train, _ = encoded['train']
    X_valid, y_valid, users_valid = encoded['valid']
    model = OfficialFM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best = -1
    best_state = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(y_train))
        started = time.time()
        losses = [
            model.step(
                X_train[indices[i:i + batch_size]],
                y_train[indices[i:i + batch_size]],
            )
            for i in range(0, len(indices), batch_size)
        ]
        valid = evaluate(users_valid, y_valid, model.predict(X_valid))
        if verbose:
            print(
                f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid GAUC {valid['GAUC']:.4f} nDCG@5 "
                f"{valid['nDCG@5']:.4f} primary {valid['primary']:.4f} | "
                f"{time.time() - started:.1f}s"
            )
        if valid['primary'] > best + 1e-5:
            best = valid['primary']
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break
    model.V, model.W, model.b = best_state
    return {'valid': evaluate(users_valid, y_valid, model.predict(X_valid))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(f"loading {args.data_dir} ...")
    data_splits = load(args.data_dir, split_names=('train', 'valid'))
    print({name: len(rows) for name, rows in data_splits.items()}, f"fields={FIELDS}")
    results = run_official_fm(
        data_splits,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"\n=== official_fm (seed={args.seed}) ===")
    metrics = results['valid']
    print(
        f"  valid  GAUC {metrics['GAUC']:.4f} | "
        f"nDCG@5 {metrics['nDCG@5']:.4f} | "
        f"primary {metrics['primary']:.4f}"
    )
