from . import activations as act
import numpy as np
from . import losses
from .layers import Dense
class Network :
    def __init__(self,layers,loss_fn:callable,loss_fn_deriv:callable):
        self.layers = layers
        self.loss_fn= loss_fn
        self.loss_fn_deriv= loss_fn_deriv
    def forward(self,X):
        A = X
        for layer in self.layers :
            A= layer.forward(A)
        return A
    def backward(self,y_pred,y_real):
        gradient = self.loss_fn_deriv(y_pred,y_real)
        for layer in self.layers[::-1]:
            gradient=layer.backward(gradient)
        return gradient
    def gradient_check(self,layer_index,X,y_real,epsilon= 1e-7):
        if(0<=layer_index-1<len(self.layers)):
            layer_index -= 1
            W = self.layers[layer_index].l_info["W"]
            dW = self.layers[layer_index].grad_info["dW"]
            dW_pred = np.zeros_like(W)
            for i,j in np.ndindex(W.shape):
                W[i,j] += epsilon
                loss_plus = self.loss_fn(self.forward(X),y_real)
                W[i,j] -= 2*epsilon
                loss_minus = self.loss_fn(self.forward(X),y_real)
                dW_numerical =(loss_plus-loss_minus)/(2*epsilon)
                dW_pred[i,j]= dW_numerical
                W[i,j]+= epsilon
            difference_W = np.linalg.norm(dW - dW_pred) / (
            np.linalg.norm(dW) + np.linalg.norm(dW_pred))
            layer_index -= 1
            b = self.layers[layer_index].l_info["b"]
            db = self.layers[layer_index].grad_info["db"]
            db_pred = np.zeros_like(b)
            for i,j in np.ndindex(b.shape):
                b[i,j] += epsilon
                loss_plus = self.loss_fn(self.forward(X),y_real)
                b[i,j] -= 2*epsilon
                loss_minus = self.loss_fn(self.forward(X),y_real)
                db_numerical =(loss_plus-loss_minus)/(2*epsilon)
                db_pred[i,j]= db_numerical
                b[i,j]+= epsilon
            difference_b = np.linalg.norm(db - db_pred) / (
            np.linalg.norm(db) + np.linalg.norm(db_pred))
            return difference_W,difference_b