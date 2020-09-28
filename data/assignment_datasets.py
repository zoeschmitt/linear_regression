import pandas as pd
from scipy.io import loadmat

def assign1():
    """
    get data
    """

    url = 'https://raw.githubusercontent.com/dibgerge/ml-coursera-python-assignments/master/Exercise1/Data/ex1data1.txt'
    data = pd.read_csv(url, names=['population', 'profit'])

    return data


def assign2():
    """
    get data
    """

    url = 'https://raw.githubusercontent.com/dibgerge/ml-coursera-python-assignments/master/Exercise2/Data/ex2data1.txt'
    data = pd.read_csv(url, names=['exam1', 'exam2', 'admitted'])

    return data


def assign3():
    data = loadmat('/content/cs4347/datasets/data.mat')
    X = data['X']
    y = data['y']
    return X, y