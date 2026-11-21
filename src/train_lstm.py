"""
Training of the LSTM learners for the LTMLE nuisance regressions (rebuilt 2026-10).

  fit_treatment(arm)   g_t(a | history), t = 0..K, one many-to-many model per replicate:
                       arm = "multinomial" -> one softmax model over the J treatments;
                       arm = "binomial"    -> J separate sigmoid models (one-vs-rest), not
                                              renormalised.
  fit_censoring()      P(C_t = 0 | history, A_t), t = 1..K, one many-to-many sigmoid model.
  fit_outcome_step()   one iterated-conditional-expectation (ICE) regression at step s:
                       pseudo-outcome Z_s in [0, 1] on the last HP["outcome_window"] months of
                       history ending at s (soft-label binary cross-entropy), warm-started from
                       the previous step of the same recursion.

Losses are masked to the relevant risk sets with per-time sample weights; no class weighting is
used, so predicted probabilities stay calibrated. Early stopping uses a subject-level validation
split.
"""

import numpy as np

import utils
from utils import HP, STATE
import test_lstm


class _BestWeights:
    """Keep the weights of the epoch with the lowest validation loss and restore them at the end."""

    def __new__(cls, patience):
        keras = STATE["keras"]

        class BestWeights(keras.callbacks.Callback):
            def __init__(self, patience):
                super().__init__()
                self.patience = int(patience)
                self.best = np.inf
                self.best_weights = None
                self.wait = 0
                self.epochs_run = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epochs_run = epoch + 1
                cur = (logs or {}).get("val_loss", (logs or {}).get("loss"))
                if cur is not None and np.isfinite(cur) and cur < self.best - 1e-6:
                    self.best, self.wait = cur, 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True

            def on_train_end(self, logs=None):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

        return BestWeights(patience)


def _fit(model, x, y, w, rows, seed, max_epochs, patience):
    tr, va = utils.split_rows(rows, HP["val_fraction"], seed)
    cb = _BestWeights(patience)
    val = (x[va], y[va], w[va]) if len(va) else None
    model.fit(x[tr], y[tr], sample_weight=w[tr], validation_data=val, epochs=int(max_epochs),
              batch_size=int(HP["batch_size"]), callbacks=[cb], verbose=0, shuffle=True)
    return cb.epochs_run


def _treatment_inputs():
    d = STATE["data"]
    return np.concatenate([d["base"], utils.lagged(d["A_onehot"])], axis=-1)


def _censoring_inputs():
    d = STATE["data"]
    # column 5 is V3 (age); see fit_censoring_models()
    base_no_age = np.delete(d["base"], 5, axis=-1)
    return np.concatenate([base_no_age, utils.lagged(d["A_onehot"]), d["A_onehot"]], axis=-1)


def _fold_key(k):
    """Extra seed key of cross-fitting fold k (empty without cross-fitting, so the seeds, and
    hence the results, of the original procedure are unchanged)."""
    return () if k is None else (int(k),)


def _cross_fit_splits(w):
    """Yield (fold, weights, training rows, rows to predict) for one nuisance model. Without
    cross-fitting: one split, fitted and predicted on all subjects. With K-fold cross-fitting
    (STATE["data"]["K"] > 1): for each fold k, the weights of fold k are set to zero, the model is
    fitted on the other K - 1 folds and predicts the subjects of fold k."""
    d = STATE["data"]
    if d["K"] <= 1:
        yield None, w, np.where(w.sum(axis=1) > 0)[0], slice(None)
        return
    for k in range(d["K"]):
        held = np.where(d["fold"] == k)[0]
        wk = w.copy()
        wk[held] = 0.0
        yield k, wk, np.where(wk.sum(axis=1) > 0)[0], held


def fit_treatment(arm="multinomial"):
    """Returns (probs (n, T, J), epochs) with probabilities at every subject and time."""
    d = STATE["data"]
    J = d["J"]
    x = _treatment_inputs()
    n, T, F = x.shape
    epochs = []
    probs = np.zeros((n, T, J), dtype="float32")
    for k, w, rows, held in _cross_fit_splits(d["at_risk"]):
        ks = _fold_key(k)
        if arm == "multinomial":
            y = d["A_onehot"]
            freq = (y * w[..., None]).sum(axis=(0, 1)) / max(w.sum(), 1.0)
            utils.set_seed(utils.sub_seed("treatment", arm, *ks))
            model = utils.build_sequence_model(T, F, J, "softmax", out_bias=np.log(freq + 1e-6))
            epochs.append(_fit(model, x, y, w, rows, utils.sub_seed("split", "treatment", *ks),
                               HP["seq_max_epochs"], HP["seq_patience"]))
            probs[held] = test_lstm.predict_sequence(model, x[held])
        else:
            for j in range(J):
                y = d["A_onehot"][..., j:j + 1]
                p = float((y[..., 0] * w).sum() / max(w.sum(), 1.0))
                utils.set_seed(utils.sub_seed("treatment", arm, j, *ks))
                model = utils.build_sequence_model(T, F, 1, "sigmoid", out_bias=[utils.logit(p)])
                epochs.append(_fit(model, x, y, w, rows,
                                   utils.sub_seed("split", "treatment", *ks),
                                   HP["seq_max_epochs"], HP["seq_patience"]))
                probs[held, :, j] = test_lstm.predict_sequence(model, x[held])[..., 0]
    return probs, np.asarray(epochs, dtype="int32")


def fit_censoring():
    """Returns (P(C_t = 0) (n, T), epochs); t = 0 is not modelled (no censoring at baseline)."""
    d = STATE["data"]
    x = _censoring_inputs()
    w0 = d["at_risk"].copy()
    w0[:, 0] = 0.0
    y = (d["C"] == 0).astype("float32")[..., None]
    n, T, F = x.shape
    probs = np.zeros((n, T), dtype="float32")
    epochs = []
    for k, w, rows, held in _cross_fit_splits(w0):
        ks = _fold_key(k)
        p = float((y[..., 0] * w).sum() / max(w.sum(), 1.0))
        utils.set_seed(utils.sub_seed("censoring", *ks))
        model = utils.build_sequence_model(T, F, 1, "sigmoid", out_bias=[utils.logit(p)])
        epochs.append(_fit(model, x, y, w, rows, utils.sub_seed("split", "censoring", *ks),
                           HP["seq_max_epochs"], HP["seq_patience"]))
        probs[held] = test_lstm.predict_sequence(model, x[held])[..., 0]
    return probs, np.asarray(epochs, dtype="int32")


def _calibrate_intercept(model, x, z, iters=25):
    """Shift the output-layer bias (logit scale) so that the mean prediction among the fitted
    subjects equals the mean pseudo-outcome, as the intercept of a logistic regression guarantees.
    Without it, early-stopped networks warm-started from the previous (shorter-horizon) step can
    drift in level across the backward recursion."""
    if len(z) == 0:
        return
    p = np.clip(test_lstm.predict_sequence(model, x)[:, 0].astype("float64"), 1e-7, 1 - 1e-7)
    eta = np.log(p / (1 - p))
    target = float(np.mean(z))
    delta = 0.0
    for _ in range(iters):  # Newton steps for mean(expit(eta + delta)) = mean(z)
        q = 1 / (1 + np.exp(-(eta + delta)))
        g = q.mean() - target
        h = (q * (1 - q)).mean()
        if abs(g) < 1e-10 or h <= 0:
            break
        delta -= g / h
    w = model.get_weights()
    w[-1] = w[-1] + np.float32(delta)
    model.set_weights(w)


def outcome_model():
    """One compiled outcome model per process; its weights are swapped between fits (no
    retracing)."""
    if STATE["outcome_model"] is None:
        d = STATE["data"]
        F = d["base"].shape[-1] + d["J"]
        utils.set_seed(utils.sub_seed("outcome_init"))
        STATE["outcome_model"] = utils.build_outcome_model(int(HP["outcome_window"]), F)
        STATE["outcome_init_weights"] = STATE["outcome_model"].get_weights()
    return STATE["outcome_model"]


def fit_outcome_step(key, fallback_key, s, rows, z):
    """Fit the ICE regression of z on history ending at s among subjects `rows` (0-based).
    Warm-starts from the last fit stored under `key` (or `fallback_key`); identical fits
    (same s, rows and z, e.g. the first step Z = Y_t shared by all rules) are reused.
    With K-fold cross-fitting, K models are fitted (model k on the subjects of `rows` outside
    fold k, each warm-started from its own previous step) and stored under `key`|k.
    Returns the number of epochs run (0 if reused)."""
    K = STATE["data"]["K"]
    if K <= 1:
        return _fit_outcome_one(key, fallback_key, s, rows, z, None)
    rows = np.asarray(rows, dtype="int64")
    z = np.asarray(z, dtype="float32")
    fold = STATE["data"]["fold"][rows]
    return sum(_fit_outcome_one("%s|cf%d" % (key, k), "%s|cf%d" % (fallback_key, k), s,
                                rows[fold != k], z[fold != k], k) for k in range(K))


def _fit_outcome_one(key, fallback_key, s, rows, z, k):
    s = int(s)
    rows = np.asarray(rows, dtype="int64")
    z = np.asarray(z, dtype="float32")
    cache_key = utils.hash_arrays(np.asarray([s]), rows, z)
    if cache_key in STATE["fit_cache"]:
        STATE["warm"][key] = STATE["fit_cache"][cache_key]
        STATE["n_cached"] += 1
        return 0
    model = outcome_model()
    warm = STATE["warm"].get(key, STATE["warm"].get(fallback_key))
    if warm is not None:
        model.set_weights(warm)
        max_epochs, patience = HP["outcome_warm_max_epochs"], HP["outcome_warm_patience"]
    else:
        w0 = [w.copy() for w in STATE["outcome_init_weights"]]
        w0[-1] = np.full_like(w0[-1], utils.logit(float(np.mean(z)) if len(z) else 0.5))
        model.set_weights(w0)
        max_epochs, patience = HP["outcome_cold_max_epochs"], HP["outcome_cold_patience"]
    utils.reset_optimizer(model)
    x = test_lstm.outcome_window(s, rows)
    y = z[:, None]
    wt = np.ones(len(rows), dtype="float32")
    utils.set_seed(utils.sub_seed("outcome_fit", key, s))  # key carries the fold under cross-fitting
    idx = np.arange(len(rows))
    _calibrate_intercept(model, x, z)  # warm start: move to the level of the new pseudo-outcome
    ep = _fit(model, x, y, wt, idx, utils.sub_seed("split", key, s), max_epochs, patience)
    # mean prediction = mean pseudo-outcome among the fitted subjects
    _calibrate_intercept(model, x, z)
    weights = model.get_weights()
    STATE["warm"][key] = weights
    STATE["fit_cache"][cache_key] = weights
    STATE["fit_cache_order"].append(cache_key)
    while len(STATE["fit_cache_order"]) > 16 * max(STATE["data"]["K"], 1):
        STATE["fit_cache"].pop(STATE["fit_cache_order"].pop(0), None)
    STATE["n_fits"] += 1
    return ep
