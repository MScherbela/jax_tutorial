import numpy as np

def load_qm7(n_train=6500):
    data = np.load("data/qm7.npz")
    n_samples = len(data['X'])
    ind = np.random.permutation(n_samples)
    X_train, X_test = np.split(data['X'][ind], [n_train])
    y_train, y_test = np.split(data['T'].flatten()[ind], [n_train])
    return X_train, y_train, X_test, y_test


def make_batches(x, y, n_batches):
    batch_size = len(x) // n_batches
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]
    for n in range(n_batches):
        yield x[n*batch_size:(n+1)*batch_size], y[n*batch_size:(n+1)*batch_size]