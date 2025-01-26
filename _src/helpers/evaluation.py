import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Tuple

# Custom imports.
from .constants import CONFIG

def compute_metrics_from_scores(relation_to_scores):
    relation_to_metrics = {relation_type: {'Precision': 0, 'Recall': 0, 'F1': 0} for relation_type in relation_to_scores}
    for rt in relation_to_scores:
        tp = relation_to_scores[rt]['TP']
        fp = relation_to_scores[rt]['FP']
        fn = relation_to_scores[rt]['FN']
        prec = tp / (fp + tp + 1e-7)
        rec = tp / (fn + tp + 1e-7)
        relation_to_metrics[rt]['Precision'] = prec
        relation_to_metrics[rt]['Recall'] = rec
        relation_to_metrics[rt]['F1'] = 2 * prec * rec / (prec + rec + 1e-7)
    return relation_to_metrics

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def compute_metrics(
        eval_pred: Tuple[np.ndarray, np.ndarray],
        thr: float=.5,
        ) -> Dict[str, float]:
    assert 0. <= thr <= 1.
    # Convert logits outputs to binary predictions.
    predictions, labels = eval_pred
    labels = labels.astype(int)
    ONE_CLASS = (CONFIG['one_class']['one_class_on'] == 'Yes')
    if not ONE_CLASS:
        predictions = sigmoid(predictions)
        predictions = (predictions >= thr).astype(int)
    else:
        print(predictions)
        max_indices = np.argmax(predictions, axis=1)
        hard_predictions = np.zeros_like(predictions)
        hard_predictions[np.arange(predictions.shape[0]), max_indices] = 1
        predictions = hard_predictions

    print(np.sum(labels, axis=0))
    print(np.sum(predictions))

    # Compute the metrics.
    # The slicings with [:, 1:] are for excluding the empty class, to only have a true F1-score macro. (the empty class will not be considered in the test dataset evaluation)
    if not ONE_CLASS:
        precision = precision_score(labels[:, 1:], predictions[:, 1:], average='macro')
        recall = recall_score(labels[:, 1:], predictions[:, 1:], average='macro')
        f1 = f1_score(labels[:, 1:], predictions[:, 1:], average='macro')
    else:
        precision = precision_score(labels[:, 1], predictions[:, 1])
        recall = recall_score(labels[:, 1], predictions[:, 1])
        f1 = f1_score(labels[:, 1], predictions[:, 1]) 

    # Compute metrics per relation.
    prec_0 = precision_score(labels[:, 0], predictions[:, 0])
    rec_0 = recall_score(labels[:, 0], predictions[:, 0])
    f1_0 = f1_score(labels[:, 0], predictions[:, 0])

    prec_1 = precision_score(labels[:, 1], predictions[:, 1])
    rec_1 = recall_score(labels[:, 1], predictions[:, 1])
    f1_1 = f1_score(labels[:, 1], predictions[:, 1])

    if not ONE_CLASS:
        prec_2 = precision_score(labels[:, 2], predictions[:, 2])
        rec_2 = recall_score(labels[:, 2], predictions[:, 2])
        f1_2 = f1_score(labels[:, 2], predictions[:, 2])

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_macro': f1,
        'prec_0': prec_0,
        'rec_0': rec_0,
        'f1_0': f1_0,
        'prec_1': prec_1,
        'rec_1': rec_1,
        'f1_1': f1_1,
    }

    if not ONE_CLASS:
        metrics['prec_2'] = prec_2
        metrics['rec_2'] = rec_2
        metrics['f1_2'] = f1_2
    return metrics