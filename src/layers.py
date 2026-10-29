import numpy as np

class Dense :
    def __init__(self,n_giris,
                n_cikis):
        self.l_info ={"W" : np.random.randn(n_giris,n_cikis)*np.sqrt(2.0 /n_giris),
                      "b" : np.random.randn(1,n_cikis)}
    def forward(self,X):
        self.X_cache = X
        Z = X@self.l_info["W"]+self.l_info["b"]
        return Z
    def backward(self, dZ):
        m = self.X_cache.shape[0]
        dW = (self.X_cache.T @ dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.l_info["W"].T
        self.grad_info = {"dW":dW,"db":db,"dX":dX}
        return dX
    def l_parameters(self):
        return self.l_info["W"],self.l_info["b"]
    def l_grads(self):
        return self.grad_info["dW"],self.grad_info["db"]