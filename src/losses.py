import numpy as np
def mse(y_pred,y_real):
    return np.sum((y_pred-y_real)**2)/y_real.shape[0]
def mse_deriv(y_pred,y_real):
    return 2*(y_pred-y_real)/y_real.shape[0]
def bce(y_pred,y_real):
    return -np.sum(y_real*np.log(y_pred)+(1-y_real)*np.log(1-y_pred))/y_real.shape[0]
def bce_deriv(y_pred,y_real):
    return ((-y_real/y_pred) + ((1-y_real)/(1-y_pred)))/y_real.shape[0]
def categorical_cross_entropy(y_pred, y_real):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_real * np.log(y_pred)) / y_real.shape[0]
def categorical_cross_entropy_deriv(y_pred, y_real):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_real / y_pred) / y_real.shape[0]