import numpy as np
import os
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
from model import initialize_weights, forward, save_model  

# ====== Utility functions ======
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return np.where(x > 0, 1, 0)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def one_hot_encode(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ====== initialize weights & biases ======
def initialize_for_training(input_size, hidden_sizes, output_size):
    np.random.seed(0)
    sizes = [input_size] + hidden_sizes + [output_size]
    weights, biases = [], []
    for i in range(len(sizes)-1):
        stddev = np.sqrt(2 / (sizes[i] + sizes[i+1]))
        weights.append(np.random.randn(sizes[i], sizes[i+1]) * stddev)
        biases.append(np.zeros((1, sizes[i+1])))
    return weights, biases

# ====== forward & backward for training (with biases) ======
def forward_train(X, weights, biases, training=True, dropout_rate=0.0):
    activations = [X]
    zs = []
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        zs.append(z)
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)
            if training and dropout_rate > 0:
                mask = (np.random.rand(*a.shape) > dropout_rate).astype(float)
                a *= mask / (1 - dropout_rate)
        activations.append(a)
    return activations, zs

def backward_train(X, y, activations, zs, weights, biases, lr):
    m = X.shape[0]
    deltas = [activations[-1] - y]  
    for i in range(len(weights)-2, -1, -1):
        delta = np.dot(deltas[0], weights[i+1].T) * relu_derivative(zs[i])
        deltas.insert(0, delta)
    for i in range(len(weights)):
        weights[i] -= lr * np.dot(activations[i].T, deltas[i]) / m
        biases[i]  -= lr * np.sum(deltas[i], axis=0, keepdims=True) / m
    return weights, biases

# ====== GLCM helper ======
def fungsi_glcm(citra, label, props):
    glcm = graycomatrix(citra, distances=[5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    features = []
    for p in props:
        features.extend(graycoprops(glcm, p)[0])
    features.append(label)
    return features

# ====== MAIN ======
if __name__ == "__main__":
    base_dir = "E:/Citra/batik"  
    props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    data = []

    for cls in sorted(os.listdir(base_dir)):  
        folder = os.path.join(base_dir, cls)
        if not os.path.isdir(folder):
            continue
        for img_name in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop = gray[h//3:h*2//3, w//3:w*2//3]
            resize = cv2.resize(crop, (128, 128))
            data.append(fungsi_glcm(resize, cls, props))

    cols = [f"{p}_{a}" for p in props for a in ['0','45','90','135']] + ["label"]
    df = pd.DataFrame(data, columns=cols)
    # Map labels to integers (must be consistent)
    mapping = {'jelamprang':0, 'jlamprang':1, 'buketan':2}
    df["label"] = df["label"].map(mapping)

    X = df.drop(columns="label").values
    y = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # STANDARD SCALER (save scaler for inference)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")

    # TRAIN NN
    hidden_sizes = [64, 32]
    lr = 0.05
    epochs = 100

    weights, biases = initialize_for_training(x_train.shape[1], hidden_sizes, 3)  # returns lists
    # Actually initialize_for_training expects (input_size, hidden_sizes, output_size) but signature used differently
    # Update: use direct call with inputs:
    weights, biases = initialize_for_training(x_train.shape[1], hidden_sizes, 3)

    # Training loop
    for epoch in range(epochs):
        activations, zs = forward_train(x_train, weights, biases, training=True, dropout_rate=0.0)
        weights, biases = backward_train(x_train, one_hot_encode(y_train, 3), activations, zs, weights, biases, lr)
        y_pred = activations[-1]
        loss = cross_entropy(one_hot_encode(y_train, 3), y_pred)
        acc = accuracy(y_train, np.argmax(y_pred, axis=1))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    # Final evaluation
    # forward without dropout
    act_train = forward(x_train, weights, biases, training=False)
    act_test  = forward(x_test, weights, biases, training=False)
    train_pred = np.argmax(act_train[-1], axis=1)
    test_pred  = np.argmax(act_test[-1], axis=1)
    print(f"\nFinal NN Train Accuracy: {accuracy(y_train, train_pred):.4f}")
    print(f"Final NN Test Accuracy: {accuracy(y_test, test_pred):.4f}")

    cm = confusion_matrix(y_test, test_pred)
    pd.DataFrame(cm, index=['jelamprang','jlamprang','buketan'],
                 columns=['jelamprang','jlamprang','buketan']).to_csv("confusion_matrix.csv", index=True)

    # Save model (weights then biases)
    np.savez("model_weights.npz", *weights, *biases)
    print("Saved model_weights.npz and scaler.joblib.")
