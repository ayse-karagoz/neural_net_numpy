import numpy as np
import math
def relu(Z):
    return np.maximum(0, Z)
def relu_deriv(Z):
    return (Z > 0).astype(float)
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
def sigmoid_deriv(Z):
    s = sigmoid(Z)
    return s * (1 - s)
def tanh(Z):
    return np.array([math.tanh(number) for number in Z])
def tanh_deriv(Z):
    return np.array([1-(math.tanh(number)**2)for number in Z])
def softmax(Z):
    shift = Z-np.max(Z,axis=1,keepdims=True)
    return np.exp(shift)/np.sum(np.exp(shift),axis=1, keepdims=True)
def softmax_deriv(Z):
    S = softmax(Z)
    J = np.zeros((S.shape[0],S.shape[1],S.shape[1]))
    i = np.array([i for i in range(S.shape[1])])
    for s in range(len(S)):
        J[s] = -np.outer(S[s], S[s])
        J[s,i,i] = S[s,i]*(1-S[s,i])
    return J
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return relu(Z)
    def backward(self, gradient):
        return gradient * relu_deriv(self.Z)
class Sigmoid:
    def forward(self, Z):
        self.Z = Z
        return sigmoid(Z)
    def backward(self, gradient):
        return gradient * sigmoid_deriv(self.Z)
class Softmax :
    def forward(self,Z):
        self.output = softmax(Z)
        return self.output

    def backward(self, gradient):
        S = self.output
        return S * (
            gradient - np.sum(gradient * S, axis=1, keepdims=True)
        )
