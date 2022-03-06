from sklearn.metrics import accuracy_score, recall_score, precision_score


class Metrics:
    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred)
