"""
ref: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
"""

import numpy as np

def inception_score(preds, n_split=10, eps=1E-16):
    # calculate the inception score
    scores = []
    n_part = preds.shape[0] // n_split
    for i in range(n_split):
        part = preds[(i * n_part):((i + 1) * n_part), :]
        kl_d = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, 0), 0) + eps))
        kl_d = np.mean(np.sum(kl_d, 1))
        scores.append(np.exp(kl_d))
    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    # test inception score
    preds = np.random.rand(1000, 10)
    print(inception_score(preds))