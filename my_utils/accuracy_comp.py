import torch


def multi_label_accuracy(outputs_logits, labels, threshold=0.5):
    """
    Compute accuracy for multi-label classification.

    Parameters:
    - outputs: Tensor of model logits or predictions (before sigmoid).
    - labels: Tensor of ground truth labels.
    - threshold: Threshold for converting logits to binary predictions.

    Returns:
    - accuracy: Scalar tensor with the accuracy for the given batch.
    """
    if labels.dtype == torch.float32:
        labels = (labels > 0.9).int()

    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(outputs_logits)

    # Convert probabilities to binary predictions
    preds = (probs > threshold).int()

    # Calculate accuracy
    correct_predictions = (preds == labels).float()
    accuracy = correct_predictions.sum() / (labels.numel())

    return accuracy


def per_class_accuracy(outputs_logits, labels, threshold=0.5):
    """
    Compute per-class accuracy for multi-label classification.

    Parameters:
    - outputs_logits: Tensor of model logits.
    - labels: Tensor of ground truth labels.
    - threshold: Threshold for converting logits to binary predictions.

    Returns:
    - correct_predictions_per_class: Number of correct predictions per class.
    - total_predictions_per_class: Total number of predictions per class.
    """
    if labels.dtype == torch.float32:
        labels = (labels > 0.9).int()

    # Apply sigmoid and threshold
    preds = (torch.sigmoid(outputs_logits) > threshold).int()

    # Calculate correct predictions per class
    correct_predictions = (preds == labels).float()

    return correct_predictions.sum(dim=0), labels.size(0)


if __name__ == "__main__":
    # Example usage
    # Create random logits and labels
    logits = torch.randn(4, 10)
    labels = torch.randn(4, 10)

    # Compute accuracy
    accuracy = multi_label_accuracy(logits, labels)
    print(f"Accuracy: {accuracy.item():.4f}")
