import numpy as np
from src.activations import Softmax

def test_forward():
    Z = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 0.0]
    ])

    layer = Softmax()
    output = layer.forward(Z)

    # Her satırın olasılıkları toplamı 1 olmalı
    np.testing.assert_allclose(
        np.sum(output, axis=1),
        np.ones(2),
        atol=1e-7
    )

    # Olasılıklar 0 ile 1 arasında olmalı
    assert np.all(output >= 0)
    assert np.all(output <= 1)

    # En büyük giriş, en büyük olasılığı vermeli
    assert np.argmax(output[0]) == 2
    assert np.argmax(output[1]) == 0