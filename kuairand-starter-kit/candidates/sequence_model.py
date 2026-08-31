"""Small deterministic sequence-aware MLP ranker implemented with NumPy."""

from __future__ import annotations

import numpy as np

from evaluate import evaluate


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class CausalSequenceMLP:
    """Field embeddings plus a nonlinear user-history-aware scoring head."""

    def __init__(self, dimension: int, *, embedding_dim: int, hidden_dim: int,
                 learning_rate: float, l2: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.H = rng.normal(0, 0.01, (embedding_dim, hidden_dim)).astype(np.float32)
        self.hb = np.zeros(hidden_dim, dtype=np.float32)
        self.o = rng.normal(0, 0.01, hidden_dim).astype(np.float32)
        self.ob = np.float32(0.0)
        self.learning_rate, self.l2 = learning_rate, l2
        self._m = {name: np.zeros_like(value) for name, value in self.checkpoint_state().items()}
        self._v = {name: np.zeros_like(value) for name, value in self.checkpoint_state().items()}
        self._step = 0

    def logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        summed = self.E[X].sum(axis=1)
        hidden = np.tanh(summed @ self.H + self.hb)
        return hidden @ self.o + self.ob, summed, hidden

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate([
            self.logits(X[offset:offset + batch_size])[0]
            for offset in range(0, len(X), batch_size)
        ])

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        logits, summed, hidden = self.logits(X)
        probability = _sigmoid(logits)
        batch = len(y)
        gradient = ((probability - y) / batch).astype(np.float32)
        gradients = {
            "o": hidden.T @ gradient + self.l2 * self.o,
            "ob": np.asarray(gradient.sum(), dtype=np.float32),
        }
        hidden_gradient = gradient[:, None] * self.o[None, :]
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        gradients["H"] = summed.T @ preactivation_gradient + self.l2 * self.H
        gradients["hb"] = preactivation_gradient.sum(axis=0)
        sum_gradient = preactivation_gradient @ self.H.T
        embedding_gradient = np.zeros_like(self.E)
        np.add.at(embedding_gradient, X, sum_gradient[:, None, :])
        gradients["E"] = embedding_gradient + self.l2 * self.E
        self._step += 1
        for name, gradient_value in gradients.items():
            self._m[name] = 0.9 * self._m[name] + 0.1 * gradient_value
            self._v[name] = 0.999 * self._v[name] + 0.001 * gradient_value * gradient_value
            update = (self._m[name] / (1 - 0.9 ** self._step)) / (
                np.sqrt(self._v[name] / (1 - 0.999 ** self._step)) + 1e-8
            )
            if name == "ob":
                self.ob = np.float32(self.ob - self.learning_rate * update)
            else:
                setattr(self, name, getattr(self, name) - self.learning_rate * update)
        return float(-np.mean(y * np.log(probability + 1e-9) + (1 - y) * np.log(1 - probability + 1e-9)))

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"E": self.E, "H": self.H, "hb": self.hb, "o": self.o, "ob": np.asarray(self.ob)}

    def restore(self, state: dict[str, np.ndarray]) -> None:
        for name in self.checkpoint_state():
            setattr(self, name, np.asarray(state[name], dtype=np.float32).copy())


def fit_sequence_mlp(splits, *, embedding_dim: int, hidden_dim: int, learning_rate: float,
                     l2: float, epochs: int, patience: int, batch_size: int, seed: int,
                     encode_fn, verbose: bool):
    encoded, dimension = encode_fn(splits)
    X_train, y_train, _ = encoded["train"]
    X_valid, y_valid, users_valid = encoded["valid"]
    model = CausalSequenceMLP(dimension, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                              learning_rate=learning_rate, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    best, bad, best_state = -1.0, 0, None
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(X_train))
        for offset in range(0, len(X_train), batch_size):
            indices = order[offset:offset + batch_size]
            losses.append(model.step(X_train[indices], y_train[indices]))
        primary = float(evaluate(users_valid, y_valid, model.predict(X_valid))["primary"])
        if verbose:
            print(f"  sequence epoch {epoch + 1:2d} | loss {np.mean(losses):.4f} | primary {primary:.4f}")
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = {name: value.copy() for name, value in model.checkpoint_state().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.restore(best_state)
    return model, encoded


class CausalAttentionRanker(CausalSequenceMLP):
    """Causal item-history attention ranker with a compact NumPy training loop."""

    def __init__(self, dimension: int, *, history_fields: int, embedding_dim: int,
                 hidden_dim: int, learning_rate: float, l2: float, seed: int) -> None:
        self.history_fields = history_fields
        self.base_fields = 5
        self.history_width = (dimension - self.base_fields) // history_fields
        if self.base_fields + self.history_width * history_fields != dimension:
            raise ValueError("attention history encoding has incompatible field ranges")
        super().__init__(dimension, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                         learning_rate=learning_rate, l2=l2, seed=seed)

    def logits(self, X: np.ndarray):
        base_X, history_X = X[:, :-self.history_fields], X[:, -self.history_fields:]
        base = self.E[base_X].sum(axis=1)
        history = self.E[history_X]
        scale = np.float32(np.sqrt(self.E.shape[1]))
        scores = (history * base[:, None, :]).sum(axis=2) / scale
        none_ids = self.base_fields + np.arange(self.history_fields) * self.history_width
        mask = history_X != none_ids[None, :]
        # The first ID in each slot range represents NONE.  All-NONE contexts
        # use zero attention context, avoiding an arbitrary padding embedding.
        scores = np.where(mask, scores, -1e9)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True)) * mask
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        context = (weights[:, :, None] * history).sum(axis=1)
        representation = base + context
        hidden = np.tanh(representation @ self.H + self.hb)
        cache = (base_X, history_X, base, history, weights, representation, hidden, mask)
        return hidden @ self.o + self.ob, cache

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate([
            self.logits(X[offset:offset + batch_size])[0]
            for offset in range(0, len(X), batch_size)
        ])

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        logits, cache = self.logits(X)
        base_X, history_X, base, history, weights, representation, hidden, mask = cache
        probability = _sigmoid(logits)
        batch = len(y)
        gradient = ((probability - y) / batch).astype(np.float32)
        gradients = {
            "o": hidden.T @ gradient + self.l2 * self.o,
            "ob": np.asarray(gradient.sum(), dtype=np.float32),
        }
        hidden_gradient = gradient[:, None] * self.o[None, :]
        preactivation_gradient = hidden_gradient * (1.0 - hidden * hidden)
        gradients["H"] = representation.T @ preactivation_gradient + self.l2 * self.H
        gradients["hb"] = preactivation_gradient.sum(axis=0)
        representation_gradient = preactivation_gradient @ self.H.T
        scale = np.float32(np.sqrt(self.E.shape[1]))
        history_gradient = weights[:, :, None] * representation_gradient[:, None, :]
        weight_gradient = (representation_gradient[:, None, :] * history).sum(axis=2)
        score_gradient = weights * (
            weight_gradient - (weight_gradient * weights).sum(axis=1, keepdims=True)
        ) * mask
        history_gradient += score_gradient[:, :, None] * base[:, None, :] / scale
        base_gradient = representation_gradient + (
            score_gradient[:, :, None] * history / scale
        ).sum(axis=1)
        embedding_gradient = np.zeros_like(self.E)
        np.add.at(embedding_gradient, base_X, base_gradient[:, None, :])
        np.add.at(embedding_gradient, history_X, history_gradient)
        gradients["E"] = embedding_gradient + self.l2 * self.E
        self._step += 1
        for name, gradient_value in gradients.items():
            self._m[name] = 0.9 * self._m[name] + 0.1 * gradient_value
            self._v[name] = 0.999 * self._v[name] + 0.001 * gradient_value * gradient_value
            update = (self._m[name] / (1 - 0.9 ** self._step)) / (
                np.sqrt(self._v[name] / (1 - 0.999 ** self._step)) + 1e-8
            )
            if name == "ob":
                self.ob = np.float32(self.ob - self.learning_rate * update)
            else:
                setattr(self, name, getattr(self, name) - self.learning_rate * update)
        return float(-np.mean(y * np.log(probability + 1e-9) + (1 - y) * np.log(1 - probability + 1e-9)))


def fit_causal_attention(splits, *, embedding_dim: int, hidden_dim: int, learning_rate: float,
                         l2: float, epochs: int, patience: int, batch_size: int, seed: int,
                         encode_fn, verbose: bool):
    encoded, dimension = encode_fn(splits)
    X_train, y_train, _ = encoded["train"]
    X_valid, y_valid, users_valid = encoded["valid"]
    from candidates.sequence_features import ATTENTION_HISTORY_FIELDS
    model = CausalAttentionRanker(
        dimension, history_fields=len(ATTENTION_HISTORY_FIELDS), embedding_dim=embedding_dim,
        hidden_dim=hidden_dim, learning_rate=learning_rate, l2=l2, seed=seed,
    )
    rng = np.random.default_rng(seed)
    best, bad, best_state = -1.0, 0, None
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(X_train))
        for offset in range(0, len(X_train), batch_size):
            indices = order[offset:offset + batch_size]
            losses.append(model.step(X_train[indices], y_train[indices]))
        primary = float(evaluate(users_valid, y_valid, model.predict(X_valid))["primary"])
        if verbose:
            print(f"  attention epoch {epoch + 1:2d} | loss {np.mean(losses):.4f} | primary {primary:.4f}")
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = {name: value.copy() for name, value in model.checkpoint_state().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    assert best_state is not None
    model.restore(best_state)
    return model, encoded
