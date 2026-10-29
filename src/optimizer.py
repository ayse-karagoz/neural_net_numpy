import numpy as np
class SGD:
    def __init__(self,lr=0.1):
        self.lr = lr
    def step(self,layer):
        layer.l_info["W"]=layer.l_info["W"]-self.lr*layer.grad_info["dW"]
        layer.l_info["b"] =layer.l_info["b"]-self.lr*layer.grad_info["db"]
class Momentum :
    def __init__(self,lr=0.1,beta=0.9):
        self.lr = lr
        self.beta = beta
    def step(self,layer):
        self.v_old = {"vW_old":np.zeros_like(layer.grad_info["dW"]),"vb_old":np.zeros_like(layer.grad_info["db"])}
        vW_new = self.beta*self.v_old["vW_old"]+(1-self.beta)*layer.grad_info["dW"]
        vb_new = self.beta*self.v_old["vb_old"]+(1-self.beta)*layer.grad_info["db"]
        self.v_old["vW_old"],self.v_old["vb_old"]= vW_new,vb_new
        layer.l_info["W"] = layer.l_info["W"] - self.lr*vW_new
        layer.l_info["b"] = layer.l_info["b"] - self.lr*vb_new