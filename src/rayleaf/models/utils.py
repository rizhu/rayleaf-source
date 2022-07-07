def number_of_correct(preds, target):
    return (preds.squeeze() == target).sum().item()


def get_predicted_labels(preds):
    return preds.argmax(dim=-1)
    