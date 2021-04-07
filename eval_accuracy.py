"""Functions for evaluating accuracy."""
from sklearn.metrics import classification_report, accuracy_score

import numpy as np


def eval_accuracy(logits: np.ndarray, label_ids: np.ndarray, classes: list,
                  save_predictions=False) -> dict:
    """Evaluates accuracy from predicted results.

    Args:
        logits: [batch_size, number_of_classes] array of class probabilities
        label_ids: [batch_size] array of integer class IDs
        classes: list of class names
        save_predictions: adds predicted class labels to output

    Returns:
        {'accuracy': F1 score,
         'report': detailed classification report,
         (optional)'predictions': [batch_size] array of class labels}
    """
    preds = logits[0] if isinstance(logits, tuple) else logits
    predictions = np.argmax(preds, axis=1)
    result = {'accuracy': accuracy_score(label_ids, predictions),
              'report': classification_report(label_ids, predictions, digits=3,
                                              target_names=classes)}
    if save_predictions:
        result['predictions'] = predictions
    return result


def get_compute_metrics(classes: list, save_predictions=False):
    """Returns 'compute_metrics' function for Trainer.

    Args:
        classes: list of class names
        save_predictions: adds predicted class labels to output
    """
    def compute_metrics(p):
        return eval_accuracy(p.predictions, p.label_ids, classes,
                             save_predictions=save_predictions)
    return compute_metrics
