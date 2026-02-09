from sklearn.isotonic import IsotonicRegression


def fit_isotonic(scores, labels):
    if len(scores) == 0:
        raise ValueError("No scores available for isotonic fitting.")
    if len(set(labels.tolist())) < 2:
        raise ValueError("Isotonic fitting requires both positive and negative labels.")

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(scores, labels)
    return model


def apply_isotonic(model, scores):
    return model.predict(scores)
