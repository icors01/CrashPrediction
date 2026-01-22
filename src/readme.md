# `model.py`: Poisson/Hawkes Temporal Features + L2 Logistic Regression

This module implements two things:

1. Poisson/Hawkes-inspired temporal features for cell–day crash counts  
2. An L2-regularized logistic regression model for predicting: “Will this cell have at least one crash tomorrow?”

---

## 1. How it works

We need,

1. a daily counts matrix  
   $Y \in \mathbb{N}^{T \times C}$ where  
   - T = number of days  
   - C = number of grid cells  
   - $Y{t, c}$ = # of crashes in cell `c` on day `t`.

then,

2. the model:
   - Computes Poisson/Hawkes-style temporal features from `Y`.
   - Builds a training set `(X, y)` where each row corresponds to a `(day t, cell c)` pair.
   - Trains a L2-regularized logistic regression that outputs  
     $$\(\hat p_{t,c} = \Pr(\text{crash in cell } c \text{ on day } t+1)\).$$

---

## 2. Contents of `model.py`

`model.py` provides:

- `compute_hawkes_features(Y, taus=(3, 7, 30))`  
  Computes exponential-decay “recency” features (Hawkes-like) at multiple time scales.

- `build_training_data_from_counts(Y, S_dict)`  
  Builds the feature matrix `X` and label vector `y` from daily counts and the Hawkes features.

- `LogisticConfig`  
  A small configuration dataclass for the logistic regression hyperparameters.

- `LogisticHotspot`  
  L2-regularized logistic regression with class weighting to handle imbalance.

---

## 3. Why use Poisson/Hawkes temporal features?

### 3.1 Intuition

Crashes are often self-exciting:
- A crash in a cell increases the chance of another crash in the near future (e.g., due to persistent risk factors).
- This extra risk decays over time.

This is the intuition behind a Hawkes process. We approximate that behaviour using exponential-decay features.

### 3.2 Discrete-time decayed counts

Given daily counts `Y[t, c]`, for each time scale $\(\tau\)$ (in days) we define:

$$
s^{(\tau)}_{t,c} = \rho \, s^{(\tau)}_{t-1,c} + Y[t-1, c],
\quad \rho = e^{-1/\tau}, \quad s^{(\tau)}_{0,c} = 0
$$

- $s^{(τ)}_{t, c}$ summarizes all past crashes in that cell, with an exponential decay:
  - recent days have more weight,
  - older events contribute less.
- Different $\(\tau\)$ values capture short, medium, and long memory.

## 4. Implementation

### 4.1 Compute the Hawkes features:

```python
from model import compute_hawkes_features

S_dict = compute_hawkes_features(Y, taus=(3, 7, 30))
```
- `Y` has shape `(T, C)`.
- `S_dict[tau]` is an array of shape `(T, C)` containing $s^{(τ)}_{t,c}$.
- and `s^{(τ)}[t]` only depends on `Y[0..t-1]`.


### 4.2 Build the training data \( X, y \)

We want to predict for each cell–day pair: "Will there be at least one crash tomorrow?"

For each day $t = 0, 1, \dots, T-2$ and cell c:
- `Label:`
$$y_{t,c} = 1\{Y[t+1,c] \geq 1\}$$

- `Features:`
$$x_{t,c} = \left( s_{t,c}^{\tau_1}, s_{t,c}^{\tau_2}, \dots \right)$$

where $\tau_1, \tau_2, \dots$ are the time scales we chose (e.g. 3, 7, 30 days).

```python
from model import build_training_data_from_counts

X, y = build_training_data_from_counts(Y, S_dict)
```
- `X` has shape `((T-1) * C, D)` where `D = number of taus` (e.g., 3).
- `y` has shape `((T-1) * C)`, and entries in `{0, 1}`.

Each row of `X` corresponds to a specific `(day t, cell c)` pair predicting day t+1.


### 4.3 Logistic Model (L2-regularized):

We use a logistic regression model to map features to probability:

$$z_i = x_i^Tw + b,$$

$$\hat{p_i} = \Sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

for each training sample `i` (a `(t, c)` pair). 

#### 4.3.1 Loss function with L2 class weights

The loss is minimized by `LogisticHotspot` is:

$$ \mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n}w_i^{(s)}[-y_ilog\hat{p_i}-(1-y_i)log(1-\hat{p_i})] + \frac{\lambda}{2}\lVert w \rVert_2^2$$

- $w_i^{(s)}$ are sample weights: postives get larger weight to handle class imbalance
- $\lambda$ is the L2 regularization strength (`l2` in `LogisticConfig`).

The gradients are:

$$\tau_i = (\hat{p_i}-y_i)w_i^{(s)},$$

$$\nabla = \frac{1}{n}\sum_i r_ix_i + \lambda w,$$

$$\nabla_b = \frac{1}{n}\sum_ir_i$$

The model uses simple gradient descent to update w and b.


### 4.4 Using the model: Example

After getting the daily count matrix `Y` with shape `(T, C)`:

```python
import numpy as np
from model import (
    compute_hawkes_features,
    build_training_data_from_counts,
    LogisticConfig,
    LogisticHotspot,
)

# 1. Compute Hawkes-style features
taus = (3, 7, 30)
S_dict = compute_hawkes_features(Y, taus=taus)

# 2. Build (X, y)
X, y = build_training_data_from_counts(Y, S_dict)

# 3. Handle class imbalance
n_pos = y.sum()
n_neg = len(y) - n_pos
pos_weight = n_neg / max(1, n_pos)

# 4. Configure the logistic model
cfg = LogisticConfig(
    lr=0.05,       # learning rate (to be tuned)
    epochs=80,     # number of gradient steps (to be tuned)
    l2=1e-2,       # L2 strength (to be tuned)
    pos_weight=pos_weight,
    fit_intercept=True,
    seed=42,
)

# 5. Train
model = LogisticHotspot(cfg).fit(X, y)

# 6. Predict probabilities
p_hat = model.predict_proba(X)   # shape (n_samples,)
y_pred = model.predict(X, threshold=0.5)
```
