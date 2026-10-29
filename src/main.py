
import numpy as np

from layers import Dense
from network import Network
from optimizer import SGD, Momentum
from losses import mse, mse_deriv
from activations import ReLU
from __init__ import train  # train() hangi dosyadaysa onun adını yaz


import numpy as np
from __init__ import  preparing_data  # Fonksiyonun bulunduğu dosya adı

# 1. Sayısal hedef testi
X,y = preparing_data(
    target_col="tip",
    datasheet="tips"
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

assert X.ndim == 2, "X iki boyutlu olmalı."
assert y.ndim == 1, "Sayısal hedef y tek boyutlu olmalı."
assert X.shape[0] == y.shape[0], "X ve y örnek sayıları eşleşmeli."
assert np.issubdtype(X.dtype, np.number), "X sayısal olmalı."
assert np.issubdtype(y.dtype, np.number), "y sayısal olmalı."
assert np.isfinite(X).all(), "X içinde NaN veya sonsuz değer var."
assert np.isfinite(y).all(), "y içinde NaN veya sonsuz değer var."

# 2. İkili kategorik hedef testi
X_binary,y_binary = preparing_data(
    target_col="sex",
    datasheet="tips"
)

print("\nİkili hedef değerleri:", np.unique(y_binary))
print("İkili hedef shape:", y_binary.shape)

assert set(np.unique(y_binary)).issubset({0, 1})
assert X_binary.shape[0] == y_binary.shape[0]

# 3. Ordinal hedef testi
X_ordinal ,y_ordinal= preparing_data(
    target_col="day",
    datasheet="tips",
    ordinal_parameters=["Thur", "Fri", "Sat", "Sun"]
)

print("\nOrdinal hedef değerleri:", np.unique(y_ordinal))
print("Ordinal hedef shape:", y_ordinal.shape)

assert set(np.unique(y_ordinal)).issubset({0, 1, 2, 3})
assert X_ordinal.shape[0] == y_ordinal.shape[0]

# 4. Ordinal özellik testi
X_features,y_features = preparing_data(
    target_col="tip",
    datasheet="tips",
    X_ordinal_parameters={
        "day": ["Thur", "Fri", "Sat", "Sun"]
    }
)

print("\nOrdinal özellik X shape:", X_features.shape)

assert X_features.shape[0] == y_features.shape[0]
assert np.issubdtype(X_features.dtype, np.number)
assert np.isfinite(X_features).all()

print("\nTüm testler başarılı!")
