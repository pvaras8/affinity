from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel

class GaussianProcessModel:
    def __init__(self, kernel='RBF', length_scale=1.0, nu=1.5, **kwargs):
        
        kernel_obj = self._select_kernel(kernel=kernel, length_scale=length_scale, nu=nu)
        self.model = GaussianProcessRegressor(kernel=kernel_obj, **kwargs)

    def _select_kernel(self, kernel, length_scale, nu):
        
        if kernel == 'RBF':
            return RBF(length_scale=length_scale)
        elif kernel == 'Matern':
            return Matern(length_scale=length_scale, nu=nu)
        elif kernel == 'RationalQuadratic':
            return RationalQuadratic(length_scale=length_scale)
        elif kernel == 'ExpSineSquared':
            return ExpSineSquared(length_scale=length_scale)
        elif kernel == 'WhiteKernel':
            return WhiteKernel(noise_level=1.0)
        else:
            raise ValueError(f"Kernel '{kernel}' no reconocido.")
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_model(self):
        return self.model


