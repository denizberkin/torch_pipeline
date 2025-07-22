"""
Only provides wrappers for sklearn metrics, these are not BaseLoss instances.
So they do not have to implement `get_alias`.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Accuracy:
    def __init__(self, **kwargs): self.kwargs = kwargs if kwargs else {}
    def __call__(self, y_true, y_pred): return accuracy_score(y_true, y_pred, **self.kwargs)


class Precision:
    def __init__(self, **kwargs): self.kwargs = kwargs if kwargs else {}
    def __call__(self, y_true, y_pred): return precision_score(y_true, y_pred, zero_division=0, **self.kwargs)


class Recall:
    def __init__(self, **kwargs): self.kwargs = kwargs if kwargs else {}
    def __call__(self, y_true, y_pred): return recall_score(y_true, y_pred, zero_division=0, **self.kwargs)


class F1Score:
    def __init__(self, **kwargs): self.kwargs = kwargs if kwargs else {}
    def __call__(self, y_true, y_pred): return f1_score(y_true, y_pred, **self.kwargs)
