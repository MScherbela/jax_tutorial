import numpy as np
import pandas as pd

def load_qm7(n_train=6500):
    df = pd.read_csv("https://raw.githubusercontent.com/MScherbela/jax_tutorial/master/data/qm7.csv")
    X = df.values[:, :-1]
    Y = df.values[:, -1]
    ind = np.random.permutation(len(X))
    X_train, X_test = np.split(X[ind], [n_train])
    y_train, y_test = np.split(Y[ind], [n_train])
    return X_train, y_train, X_test, y_test


