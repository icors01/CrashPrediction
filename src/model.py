from dataclasses import dataclass
import numpy as np

# Poisson/Hawkes-inspired features

def compute_hawkes_features(Y, taus=(3, 7, 30)):
    """
    Compute Poisson/Hawkes-inspired exponential-decay features.

    Y: (T, C) array of daily crash counts per cell
       Y[t, c] = number of crashes in cell c on day index t

    taus: iterable of time scales (in days) for exponential decay

    Returns:
      S_dict: dict[tau] -> S_tau with shape (T, C)
              S_tau[t, c] = s_{t,c}^{(tau)} = rho * s_{t-1,c}^{(tau)} + Y[t-1,c]
              using only past days (no leakage)
    """
    T, C = Y.shape
    S_dict = {}

    for tau in taus:
        rho = float(np.exp(-1.0 / float(tau)))
        S = np.zeros((T, C))
        for t in range(1, T):
            # Only uses Y[t-1], so no look-ahead leakage
            S[t] = rho * S[t - 1] + Y[t - 1]
        S_dict[tau] = S

    return S_dict


def build_training_data_from_counts(Y, S_dict):
    """
    Build (X, y) from:
      - Y: (T, C) daily crash counts
      - S_dict: {tau: S_tau} Hawkes features from compute_hawkes_features

    For each day t (0..T-2) and cell c, we create one training row:
      - features: [S_tau[t, c] for tau in sorted(taus)]
      - label:    y = 1 if Y[t+1, c] >= 1 else 0

    Returns:
      X: ((T-1) * C, D) feature matrix
      y: ((T-1) * C,) labels (0/1)
    """
    T, C = Y.shape
    taus_sorted = sorted(S_dict.keys())

    rows_X = []
    rows_y = []

    for t in range(T - 1):  # last day only used as target, not feature
        for c in range(C):
            feats = [S_dict[tau][t, c] for tau in taus_sorted]
            rows_X.append(feats)
            rows_y.append(1 if Y[t + 1, c] >= 1 else 0)

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=int)
    return X, y


# Logistic regression model (L2)

def _sigmoid(z):
    """
    Elementwise sigmoid:
        sigma(z_i) = 1 / (1 + exp(-z_i))
    Uses a numerically stable split for positive/negative values.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos

    # For non-negative z
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    # For negative z (avoid overflow)
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out


def _compute_class_weights(y, pos_weight):
    """
    Per-sample weights for class imbalance:
        w_i = pos_weight   if y_i = 1 and pos_weight is given
            = 1            otherwise
    """
    y = np.asarray(y, dtype=int)
    if pos_weight is None:
        return np.ones_like(y, dtype=float)

    w = np.ones_like(y, dtype=float)
    w[y == 1] = float(pos_weight)
    return w


@dataclass
class LogisticConfig:
    lr: float = 0.05           # learning rate
    epochs: int = 50           # number of gradient steps
    l2: float = 1e-2           # L2 regularization strength
    pos_weight: float = None   # positive class weight (for imbalance)
    fit_intercept: bool = True
    seed: int = 42             # RNG seed for init


class LogisticHotspot:
    """
    L2-regularized logistic regression for next-day crash prediction.

    Model:
        z_i = x_i^T w + b
        p_i = sigma(z_i) = 1 / (1 + exp(-z_i))

    Loss (average over samples, with sample weights and L2):
        L(w, b) = (1/n) * sum_i w_i^(s) * [
                     -y_i log p_i - (1 - y_i) log(1 - p_i)
                  ] + (lambda / 2) * ||w||^2

    Gradients:
        r_i = (p_i - y_i) * w_i^(s)
        grad_w = (1/n) * sum_i r_i x_i + lambda * w
        grad_b = (1/n) * sum_i r_i

    Updates:
        w <- w - lr * grad_w
        b <- b - lr * grad_b
    """

    def __init__(self, cfg=LogisticConfig()):
        self.cfg = cfg
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape

        rng = np.random.default_rng(self.cfg.seed)
        self.w = rng.normal(0.0, 0.01, size=d)
        self.b = 0.0

        # Sample weights for imbalance
        w_samp = _compute_class_weights(y, self.cfg.pos_weight)

        for _ in range(self.cfg.epochs):
            # Forward pass
            if self.cfg.fit_intercept:
                z = X.dot(self.w) + self.b
            else:
                z = X.dot(self.w)

            p = _sigmoid(z)

            # r_i = (p_i - y_i) * w_i^(s)
            r = (p - y) * w_samp

            # Gradients
            grad_w = X.T.dot(r) / float(n) + self.cfg.l2 * self.w
            grad_b = r.mean() if self.cfg.fit_intercept else 0.0

            # Parameter update
            self.w -= self.cfg.lr * grad_w
            if self.cfg.fit_intercept:
                self.b -= self.cfg.lr * grad_b

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.w is None:
            raise ValueError("Model not fitted yet.")
        if self.cfg.fit_intercept:
            z = X.dot(self.w) + self.b
        else:
            z = X.dot(self.w)
        return _sigmoid(z)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
