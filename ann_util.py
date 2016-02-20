import math
import pickle
import random


def logistic(x):
    return 1.0 / (1 + math.exp(-x))


def deriv_logistic(x):
    lgst = logistic(x)
    return (1 - lgst) * lgst


def hyperbolic_tangent(x):
    return math.tanh(x)


def deriv_hyperbolic_tangent(x):
    th = math.tanh(x)
    return 1 - th * th


def between(min, max):
    """
    Return a real random value between min and max.
    """
    return random.random() * (max - min) + min


def make_matrix(N, M):
    """
    Make an N rows by M columns matrix.
    """
    return [[0 for i in range(M)] for i in range(N)]


def serialize(nn, fname):
    with open(fname, 'wb') as f:
        # use protocol 2, default 0 has problems
        pickle.dump(nn, f, 2)


def deserialize(fname):
    with open(fname, 'rb') as f:
        nn = pickle.load(f)
        return nn
