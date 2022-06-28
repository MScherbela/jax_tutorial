import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
np.random.seed(1234)

#%% Load the data
data = np.load("data/qm7.npz")
X_train, X_test = np.split(data['X'], [6500])
y_train, y_test = np.split(data['T'].flatten(), [6500])
print(f"{X_train.shape=}; {X_test.shape=}")
print(f"{y_train.shape=}; {y_test.shape=}")
scale_x = np.std(np.linalg.eigvalsh(X_train), axis=0)
mean_y = np.mean(y_train)
scale_y = np.std(y_train)

#%% Build the model
def mlp(params, x):
    for ind_layer, (weights, bias) in enumerate(params):
        x = x @ weights + bias
        if ind_layer != (len(params) - 1):
            x = jax.nn.relu(x)
    return x

def init_mlp(n_neurons, input_dim):
    params = []
    for ind_layer in range(len(n_neurons)):
        if ind_layer == 0:
            dim_in = input_dim
        else:
            dim_in = n_neurons[ind_layer-1]
        dim_out = n_neurons[ind_layer]

        w_init = np.random.normal(loc=0, scale=2/(dim_in + dim_out), size=(dim_in, dim_out))
        b_init = np.zeros(shape=(dim_out,))
        params.append((w_init, b_init))
    return params

def model(params, x):
    x = jnp.linalg.eigvalsh(x)
    x = x / scale_x
    y = mlp(params, x) * scale_y + mean_y
    return y.flatten()


#%% Define a loss function and an optimizer / training-step
def loss_func(params, x, y_target):
    y_pred = model(params, x)
    residual = (y_pred - y_target) / scale_y
    return jnp.mean(residual**2)

@jax.jit
def training_step(params, x, y_target, learning_rate):
    loss, grad = jax.value_and_grad(loss_func, argnums=0)(params, x, y_target)
    param_update = jax.tree_map(lambda g: -learning_rate * g, grad)
    new_params = jax.tree_map(jnp.add, params, param_update)
    return loss, new_params

def make_batches(x, y, n_batches):
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]
    return np.split(x, n_batches), np.split(y, n_batches)

#%% Initialize the network and run the actual training
n_neurons_per_layer = [50, 50, 20, 1]
learning_rate = 0.05
params = init_mlp(n_neurons_per_layer, input_dim=X_train.shape[-1])

loss_values = []
mae_test_values = []
mae_train_values = []
for n_epoch in range(20):
    batches_x, batches_y = make_batches(X_train, y_train, 100)
    for batch_x, batch_y in zip(batches_x, batches_y):
        loss, params = training_step(params, batch_x, batch_y, learning_rate)
        loss_values.append(loss)

    # Calculate metrics after every epoch
    y_test_predicted = model(params, X_test)
    y_train_predicted = model(params, X_train)
    mae_test = np.mean(np.abs(y_test_predicted - y_test))
    mae_train = np.mean(np.abs(y_train_predicted - y_train))
    mae_test_values.append(mae_test)
    mae_train_values.append(mae_train)
    print(f"epoch={n_epoch:<3d}, loss={loss:.3f}, mae_train={mae_train:.1f} kcal/mol, mae_test={mae_test:.1f} kcal/mol")

#%% Analyze final model
print(f"Mean absolute error (MAE) on training set: {mae_train:.1f} kcal/mol")
print(f"Mean absolute error (MAE) on test set   : {mae_test:.1f} kcal/mol")

plt.figure()
fig, axes = plt.subplots(1,3)
axes[0].semilogy(loss_values)
axes[0].set_title("Training loss")

axes[1].plot(mae_train_values, label="Train")
axes[1].plot(mae_test_values, label="Test")
axes[1].set_title("Mean absolute error [kcal/mol]")
axes[1].legend()

axes[2].scatter(y_train, y_train_predicted, label="Train")
axes[2].scatter(y_test, y_test_predicted, label="Test")
axes[2].legend()
axes[2].set_xlabel("Ground truth [kcal/mol]")
axes[2].set_ylabel("Predicted [kcal/mol]")










