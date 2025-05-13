import torch

def multiclass_confusion_matrix(preds, labels, num_classes=None):
    """
    Computes the confusion matrix for multiclass classification.

    Args:
        preds (Tensor): Predicted labels (1D tensor).
        labels (Tensor): Ground truth labels (1D tensor).
        num_classes (int): Number of classes.

    Returns:
        Tensor: Confusion matrix of shape (num_classes, num_classes).
    """
    if not torch.is_tensor(preds):
        preds = torch.tensor(preds)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    # Move preds and labels to CPU for indexing
    preds = preds.cpu()
    labels = labels.cpu()

    if num_classes is None:
        num_classes = max(preds.max(), labels.max()).item() + 1

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(labels, preds):
        conf_matrix[t.long(), p.long()] += 1

    return conf_matrix

def multiclass_f1_score(preds, labels, num_classes=None):
    """
    Computes the macro-averaged F1 score for multiclass classification.
    
    Args:
        preds (Tensor): Predicted labels (1D tensor).
        labels (Tensor): Ground truth labels (1D tensor).
        num_classes (int): Number of classes.
    
    Returns:
        float: Macro F1 score.
    """
    cm = multiclass_confusion_matrix(preds, labels, num_classes)
    f1_scores = []

    for i in range(cm.shape[0]):
        i = i
        TP = cm[i, i].item()
        FP = cm[:, i].sum().item() - TP
        FN = cm[i, :].sum().item() - TP
        denom = (2 * TP + FP + FN)
        f1 = 2 * TP / denom if denom > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)