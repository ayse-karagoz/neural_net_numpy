from .layers import Dense
import numpy as np
import seaborn as sns
import pandas as pd
def preparing_data(target_col:str,datasheet : str,ordinal_parameters=None,X_ordinal_parameters=None):
    training_data = sns.load_dataset(datasheet)
    df = training_data.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])
    if y.dtype in [object, "category"]:
        if(ordinal_parameters):
            mapping = {
            category: i
            for i, category in enumerate(ordinal_parameters)
            }
            y = y.map(mapping)
        else:
            if(y.value_counts().shape[0]==2):
                y = pd.Categorical(y).codes
            else:
                y = pd.get_dummies(y)
    if(X_ordinal_parameters):
        for column, categories in X_ordinal_parameters.items():
            mapping = {category: i for i, category in enumerate(categories)}
            X[column] = X[column].map(mapping)
    X = pd.get_dummies(X)
    y = np.array(y)
    X = X.to_numpy(dtype=float)
    return X,y
def train(network, X, y, optimizer, epochs=100, batch_size=32):
    history = []
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(X))
        x_shuffled = X[indices]
        y_shuffled = y[indices]
        for start in range(0, len(x_shuffled), batch_size):
            X_batch = x_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]
            y_pred = network.forward(X_batch)
            loss = network.loss_fn(y_pred, y_batch)
            epoch_loss += loss * len(X_batch)
            network.backward(y_pred, y_batch)
            for layer in network.layers:
                if isinstance(layer, Dense):
                    optimizer.step(layer)
        epoch_loss /= len(X)
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.6f}")
        history.append(epoch_loss)
    return history