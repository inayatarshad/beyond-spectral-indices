import numpy as np

EPS = 1e-9


class Confusion:
    """Pooled pixel confusion counts, optionally per stratum."""

    def __init__(self):
        self.counts = {}

    def update(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray, strata: np.ndarray | None = None):
        keys = np.zeros(pred.shape, dtype=object) if strata is None else strata
        for key in np.unique(keys[valid]):
            sel = valid & (keys == key)
            p, t = pred[sel].astype(bool), target[sel].astype(bool)
            c = self.counts.setdefault(key, np.zeros(4, dtype=np.int64))
            c += [np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), np.sum(~p & ~t)]

    def table(self) -> dict:
        return {k: summarize(*c) for k, c in self.counts.items()}


def summarize(tp, fp, fn, tn) -> dict:
    n = tp + fp + fn + tn
    return {
        "iou": tp / (tp + fp + fn + EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "error_rate": (fp + fn) / (n + EPS),
        "fp_rate": fp / (fp + tn + EPS),
        "fn_rate": fn / (fn + tp + EPS),
        "pixels": int(n),
        "positives": int(tp + fn),
    }
