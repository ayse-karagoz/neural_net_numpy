
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.layers import Dense
from src.activations import ReLU, Sigmoid, Softmax
from src.network import Network
from src.losses import (
    categorical_cross_entropy,
    categorical_cross_entropy_deriv,
    bce,
    bce_deriv,
)
from src.optimizer import SGD
from src import train


st.set_page_config(
    page_title="NumPy Neural Network Lab",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NumPy Neural Network Lab")
st.write(
    "Kendi yazdığın sinir ağını farklı veri kümelerinde eğit, "
    "performansını ölç ve sonuçlarını incele."
)

DATASETS = {
    "Iris": "3 sınıflı çiçek sınıflandırması",
    "Wine": "3 sınıflı şarap sınıflandırması",
    "Breast Cancer": "İkili tümör sınıflandırması",
    "Moons": "Doğrusal olmayan iki sınıflı sentetik veri",
}


@st.cache_data
def load_dataset(name, samples, noise):
    if name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)

    elif name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)

    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = list(data.target_names)

    else:
        X, y = make_moons(
            n_samples=samples,
            noise=noise,
            random_state=42,
        )
        feature_names = ["Feature 1", "Feature 2"]
        target_names = ["Sınıf 0", "Sınıf 1"]

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        feature_names,
        target_names,
    )


with st.sidebar:
    st.header("Veri ve model ayarları")

    dataset_name = st.selectbox(
        "Veri kümesi",
        list(DATASETS.keys()),
    )
    st.caption(DATASETS[dataset_name])

    test_size = st.slider(
        "Test verisi oranı",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )

    hidden_units = st.select_slider(
        "Gizli katman nöron sayısı",
        options=[4, 8, 16, 32],
        value=8,
    )

    epochs = st.slider(
        "Epoch sayısı",
        min_value=50,
        max_value=1500,
        value=500,
        step=50,
    )

    batch_size = st.select_slider(
        "Batch size",
        options=[8, 16, 32, 64],
        value=16,
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.05,
    )

    if dataset_name == "Moons":
        samples = st.slider(
            "Örnek sayısı",
            200,
            1500,
            500,
            step=100,
        )
        noise = st.slider(
            "Veri gürültüsü",
            0.0,
            0.5,
            0.2,
            step=0.05,
        )
    else:
        samples, noise = 500, 0.2

    run_button = st.button(
        "Modeli eğit",
        type="primary",
        use_container_width=True,
    )


X, y, feature_names, target_names = load_dataset(
    dataset_name, samples, noise
)

n_classes = len(np.unique(y))
multiclass = n_classes > 2

X_train, X_test, y_train_labels, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if multiclass:
    y_train = np.eye(n_classes)[y_train_labels]
    loss_fn = categorical_cross_entropy
    loss_deriv = categorical_cross_entropy_deriv
    output_layer = Softmax()
else:
    y_train = y_train_labels.reshape(-1, 1).astype(float)
    loss_fn = bce
    loss_deriv = bce_deriv
    output_layer = Sigmoid()


# Veri kümesini göster
col1, col2, col3 = st.columns(3)
col1.metric("Toplam örnek", len(X))
col2.metric("Özellik sayısı", X.shape[1])
col3.metric("Sınıf sayısı", n_classes)

with st.expander("Veri kümesini incele"):
    st.write("Özellik isimleri:", feature_names)
    st.write("Sınıflar:", target_names)
    st.dataframe(
        pd.DataFrame(X, columns=feature_names).head(10),
        use_container_width=True,
    )


if run_button:
    np.random.seed(42)

    layers = [
        Dense(X_train.shape[1], hidden_units),
        ReLU(),
        Dense(hidden_units, n_classes if multiclass else 1),
        output_layer,
    ]

    network = Network(layers, loss_fn, loss_deriv)
    optimizer = SGD(lr=learning_rate)

    progress = st.progress(0)
    status = st.empty()

    # Eğitim, mevcut train fonksiyonunun çıktısını kullanır.
    with st.spinner("Model eğitiliyor..."):
        history = train(
            network,
            X_train,
            y_train,
            optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )

    progress.progress(100)
    status.success("Eğitim tamamlandı.")

    raw_predictions = network.forward(X_test)

    if multiclass:
        y_pred = np.argmax(raw_predictions, axis=1)
    else:
        y_pred = (raw_predictions.ravel() >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    st.session_state["results"] = {
        "history": history,
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
        "target_names": target_names,
        "dataset_name": dataset_name,
    }


if "results" in st.session_state:
    result = st.session_state["results"]

    if result["dataset_name"] != dataset_name:
        st.info("Yeni veri kümesi için modeli yeniden eğit.")
    else:
        st.divider()
        st.header("Eğitim sonuçları")

        metric1, metric2 = st.columns(2)
        metric1.metric(
            "Test doğruluğu",
            f"{result['accuracy']:.2%}",
        )
        metric2.metric(
            "Son epoch kaybı",
            f"{result['history'][-1]:.6f}",
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Loss eğrisi")
            fig, ax = plt.subplots()
            ax.plot(result["history"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Eğitim kaybı")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(
                result["y_test"],
                result["y_pred"],
            )
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Tahmin")
            ax.set_ylabel("Gerçek")
            ax.set_xticks(range(len(result["target_names"])))
            ax.set_yticks(range(len(result["target_names"])))
            ax.set_xticklabels(result["target_names"], rotation=30)
            ax.set_yticklabels(result["target_names"])

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Sınıflandırma raporu")
        report = classification_report(
            result["y_test"],
            result["y_pred"],
            labels=list(range(len(result["target_names"]))),
            target_names=result["target_names"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True,
        )