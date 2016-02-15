import random


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
