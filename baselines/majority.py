from collections import Counter


def majority_class_predict(y_train, n_samples):
    """
    Predict the most frequent class for every sample.
    """
    majority_class = Counter(y_train).most_common(1)[0][0]

    return [majority_class] * n_samples