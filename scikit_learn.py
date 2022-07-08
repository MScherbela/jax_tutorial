from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)
#%% Load and inspect the data

data = load_digits()
print(f"data.images.shape = {data.images.shape}")
plt.close("all")
n_rows, n_cols = 3, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,2*n_rows), dpi=100)
for row in range(n_rows):
    for col in range(n_cols):
        ind = np.random.randint(len(data.images))
        axes[row][col].imshow(data.images[ind, ...], cmap='binary')
        axes[row][col].axis("off")
        axes[row][col].set_title(str(data.target[ind]))
fig.tight_layout()


#%% Prepare the data
X = data.images.reshape([-1, 8*8])
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1400)

def evaluate_model(trained_model):
    y_pred_train = trained_model.predict(X_train)
    y_pred_test = trained_model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Training accuracy: {acc_train:.0%}")
    print(f"Test accuracy    : {acc_test:.0%}")
    return y_pred_train, y_pred_test

#%%
print("Naive Bayes classifier on raw data:")
model = GaussianNB()
model.fit(X_train, y_train)
_,_ = evaluate_model(model)

#%%
print("PCA + Naive Bayes classifier:")
model = Pipeline([('pca', PCA(n_components=10)),
                  ('naive_bayes', GaussianNB())
                  ])
model.fit(X_train, y_train)
_,_ = evaluate_model(model)

#%%
print("Linear classifier on raw input data:")
model = LogisticRegression(C=0.005, max_iter=1000)
model.fit(X_train, y_train)
_, y_pred_test = evaluate_model(model)
ind_wrong = (y_pred_test != y_test)

n_rows = 2
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,2*n_rows), dpi=100)
for i, (img, y_true, y_pred) in enumerate(zip(X_test[ind_wrong],
                                              y_test[ind_wrong],
                                              y_pred_test[ind_wrong])):
    if i >= (n_rows * n_cols):
        break
    ax = axes[i//n_cols][i%n_cols]
    ax.imshow(img.reshape([8,8]), cmap='binary')
    ax.set_title(f"T: {y_true}, P: {y_pred}")
for ax in axes.flatten():
    ax.axis("off")
fig.suptitle("Wrong predictions")
fig.tight_layout()
















