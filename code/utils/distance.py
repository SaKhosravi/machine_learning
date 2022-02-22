import numpy as np

"""
implementation some of distances measures for Machine Learning

"""
class Distance:
    def __init__(self):
        pass

    def euclidean(self, x, y):
        x, y = np.array(x), np.array(y)
        distance = np.power((y - x), 2)
        return np.power(np.sum(distance), 1 / 2)

    def manhattan(self, x, y):
        x, y = np.array(x), np.array(y)
        distance = np.absolute(y - x)
        return np.sum(distance)

    def minkowski(self, x, y, p=2):
        x, y = np.array(x), np.array(y)

        distance = np.absolute(y - x)
        distance = np.power(distance, p)
        return np.power(np.sum(distance), 1 / p)

    def hamming(self, x, y):
        x, y = np.array(x), np.array(y)

        distance = np.absolute(y - x)
        return np.sum(distance) / len(x)

    def mahalanobis(self, x, data=None):
        x = np.array(x)

        m = np.mean(data, axis=0)
        x_m = x - m
        covMat = np.cov(np.transpose(data), bias=False)
        inv_covMat = np.linalg.inv(covMat)
        temp1 = np.dot(x_m, inv_covMat)
        md = np.dot(temp1, np.transpose(x_m))
        return np.sqrt(md)

    def cosine_similarity(self, x, y):
        x, y = np.array(x), np.array(y)

        x_dot_y = np.dot(x, y)
        print(x_dot_y)
        return x_dot_y / (np.linalg.norm(x) * np.linalg.norm(y))

    def jaccard_similarity(self, x, y):
        x, y = np.array(x), np.array(y)

        intersection = np.intersect1d(x, y)
        union = np.union1d(x, y)
        return len(intersection) / len(union)

