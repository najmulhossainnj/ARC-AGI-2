"""
A tiny, dependency-free 2-layer MLP binary classifier.

torch is an optional dependency for this project (see pyproject.toml), so
the *default* trained ranker backend is this small NumPy net rather than
the torch `ProgramRanker`. It's trained with plain batch gradient descent
on a binary cross-entropy objective: given (task_features, program_features)
predict whether the program is an exact solution for that task.

This is intentionally simple (single hidden layer, sigmoid output) since the
input is a few dozen hand-engineered features and the training set is a few
thousand examples at most -- there's nothing here that needs a heavier
optimizer or architecture.
"""
from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyMLP:
    def __init__(self, input_dim, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, size=(input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, size=(hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.input_dim = input_dim
        self.hidden = hidden
        # feature normalization, fit during training
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.std = np.ones(input_dim, dtype=np.float32)

    def _normalize(self, X):
        return (X - self.mean) / self.std

    def forward(self, X):
        Xn = self._normalize(X)
        z1 = Xn @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def predict_proba(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        _, _, z2 = self.forward(X)
        return _sigmoid(z2).ravel()

    def fit(self, X, y, epochs=200, lr=0.05, l2=1e-4, verbose=False):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n = X.shape[0]

        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        self.mean = X.mean(axis=0)
        self.std = std

        for epoch in range(epochs):
            z1, a1, z2 = self.forward(X)
            p = _sigmoid(z2)

            # BCE gradient w.r.t. z2 is simply (p - y) / n
            dz2 = (p - y) / n
            dW2 = a1.T @ dz2 + l2 * self.W2
            db2 = dz2.sum(axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            Xn = self._normalize(X)
            dW1 = Xn.T @ dz1 + l2 * self.W1
            db1 = dz1.sum(axis=0)

            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            if verbose and (epoch % max(1, epochs // 10) == 0):
                eps = 1e-7
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                acc = float(np.mean((p > 0.5) == (y > 0.5)))
                print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.4f}")
        return self

    def to_dict(self):
        return {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "mean": self.mean, "std": self.std,
            "input_dim": np.asarray(self.input_dim),
            "hidden": np.asarray(self.hidden),
        }

    @classmethod
    def from_dict(cls, d):
        model = cls(int(d["input_dim"]), int(d["hidden"]))
        model.W1 = d["W1"]; model.b1 = d["b1"]
        model.W2 = d["W2"]; model.b2 = d["b2"]
        model.mean = d["mean"]; model.std = d["std"]
        return model

    def save(self, path):
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls.from_dict(d)
