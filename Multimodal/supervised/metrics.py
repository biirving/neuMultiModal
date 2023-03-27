from sklearn import metrics

def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)