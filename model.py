import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def initialize_weights(input_size, hidden_sizes, output_size):
    np.random.seed(0)
    layers = [input_size] + hidden_sizes + [output_size]
    weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / (layers[i] + layers[i+1]))
               for i in range(len(layers)-1)]
    biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
    return weights, biases

def forward(X, weights, biases, training=False, dropout_rate=0.0):
    """
    Forward propagation that returns list of activations.
    - dropout_rate applies only if training==True and dropout_rate>0
    - last layer uses softmax
    """
    activations = [X]
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = np.dot(activations[-1], W) + b  # shape (n_samples, units)
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)
            if training and dropout_rate > 0:
                mask = (np.random.rand(*a.shape) > dropout_rate).astype(float)
                a *= mask / (1 - dropout_rate)
        activations.append(a)
    return activations

def predict(X, weights, biases):
    """
    Predict class indices given features X, weights and biases.
    """
    activations = forward(X, weights, biases, training=False, dropout_rate=0.0)
    preds = np.argmax(activations[-1], axis=1)
    return preds

def save_model(weights, biases, filename="model_weights.npz"):
    """
    Save weights and biases into a single .npz file.
    Saved order: weights0, weights1, ..., bias0, bias1, ...
    """
    arrs = list(weights) + list(biases)
    np.savez(filename, *arrs)

def load_model(filename="model_weights.npz"):
    """
    Load model saved by save_model.
    Returns (weights_list, biases_list)
    """
    data = np.load(filename, allow_pickle=True)
    files = data.files  # arr_0, arr_1, ...
    n = len(files)
    if n % 2 != 0:
        raise ValueError("Saved model arrays count is odd; expected weights+biases pairs.")
    num_layers = n // 2
    weights = [data[f] for f in files[:num_layers]]
    biases  = [data[f] for f in files[num_layers:]]
    return weights, biases
