import numpy as np
from src.activations import Softmax

def test_gradient():
    Z = np.array([
        [0.2, -0.4, 1.0],
        [0.7, 0.1, -0.3]
    ])

    upstream = np.array([
        [0.3, -0.2, 0.5],
        [-0.1, 0.4, 0.2]
    ])

    layer = Softmax()
    layer.forward(Z)
    analytical = layer.backward(upstream)

    epsilon = 1e-6
    numerical = np.zeros_like(Z)

    # Her giriş elemanını küçük miktarda değiştir
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z_plus = Z.copy()
            Z_minus = Z.copy()

            Z_plus[i, j] += epsilon
            Z_minus[i, j] -= epsilon

            out_plus = Softmax().forward(Z_plus)
            out_minus = Softmax().forward(Z_minus)

            # Çıktıların upstream ile ağırlıklandırılmış toplamı
            loss_plus = np.sum(out_plus * upstream)
            loss_minus = np.sum(out_minus * upstream)

            numerical[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7
    )