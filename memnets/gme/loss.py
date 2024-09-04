import numpy as np
import torch
from .utils import eig_decomposition, calculate_inverse, estimate_c_tilde_matrix

class MEMLoss:
    """ Compute MEMNet loss.

    Parameters
    ----------
    lagtimes : list
        Encoder lag times of MEMnets.

    device : torch.device, default = None
        The device on which the torch modules are executed.

    epsilon : float, default = 1e-6
        The strength of the regularization/truncation under which matrices are inverted.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    reversible : boolean, default = True
        Whether to enforce detailed balance constraint. 
    """

    def __init__(self, lagtimes, device=None, epsilon=1e-6, mode='regularize', reversible=True):

        self._lagtime_0 = lagtimes[0]
        self._lagtime_hat = lagtimes[1:]
        self._epsilon = epsilon
        self._mode = mode
        self._device = device
        self._X = None
        self._Y = None
        self._W = None
        self._m0_tilde = None
        self._log_lambda_hat = None

        self._lm_list = []
        self._m0_tilde_list = []
        self._log_lambda_hat_list = []
        self._rmse_list = []
        self._Y_list = []
        self._Y_predict_list = []

        self._reversible = reversible

    @property
    def m0_tilde(self):
        return self._m0_tilde

    @property
    def log_lambda_hat(self):
        return self._log_lambda_hat

    @property
    def Y(self):
        return self._Y

    def fit(self, data):

        assert len(data) > 2

        matrices = [estimate_c_tilde_matrix(data[0], data[i + 1], reversible=self._reversible) for i in range(len(data) - 1)]
        log_eigvals = []
        for i in range(len(data) - 1):
            tmp, _ = eig_decomposition(matrices[i], epsilon=self._epsilon, mode=self._mode)
            idx = torch.argsort(tmp, descending=True)
            tmp = tmp[idx]
            log_eigvals.append(torch.log(tmp))
        log_eigvals = torch.stack(log_eigvals)

        log_eigvals_0 = log_eigvals[0]
        ones = torch.ones(len(log_eigvals) - 1).to(device=self._device)
        lagtimes = torch.tensor(self._lagtime_hat).to(device=self._device)

        self._X = torch.cat((lagtimes, ones), dim=0).reshape(2, len(log_eigvals) - 1).t()
        self._Y = log_eigvals[1:]
        left = calculate_inverse(torch.matmul(self._X.t(), self._X), epsilon=self._epsilon, mode=self._mode)
        right = torch.matmul(self._X.t(), self._Y)
        self._W = torch.matmul(left, right)

        self._log_lambda_hat = self._W[0, :]
        self._m0_tilde = torch.clip((log_eigvals_0 / self._lagtime_0 - self._log_lambda_hat),max=0)

        return self

    def loss(self, weight=1.):

        loss = torch.sum(torch.abs(self._m0_tilde) + weight * torch.abs(self._log_lambda_hat))

        return loss
    
    def lm(self, weight=1.):

        lm = torch.abs(self._m0_tilde) + weight * torch.abs(self._log_lambda_hat)

        return lm

    @property
    def Y_predict(self):

        Y_predict = torch.matmul(self._X, self._W)

        return Y_predict

    @property
    def rmse(self):

        Y_predict = self.Y_predict
        num_lagtimes, num_modes = Y_predict.shape[0], Y_predict.shape[1]
        delta_sq = (Y_predict - self._Y) ** 2
        rmse = torch.sqrt(torch.sum(delta_sq, dim=0) / num_lagtimes)

        return rmse

    def save(self, weight=1.):

        with torch.no_grad():

            self._lm_list.append(self.lm(weight=weight).cpu().numpy())
            self._m0_tilde_list.append(self.m0_tilde.cpu().numpy())
            self._log_lambda_hat_list.append(self.log_lambda_hat.cpu().numpy())

            self._rmse_list.append(self.rmse.cpu().numpy())
            self._Y_list.append(self.Y.cpu().numpy())
            self._Y_predict_list.append(self.Y_predict.cpu().numpy())

        return self

    def output_mean_lm(self):

        mean_lm = np.mean(np.stack(self._lm_list), axis=0)

        return mean_lm

    def output_mean_m0_tilde(self):

        mean_m0_tilde = np.mean(np.stack(self._m0_tilde_list), axis=0)

        return mean_m0_tilde

    def output_mean_log_lambda_hat(self):

        mean_log_lambda_hat = np.mean(np.stack(self._log_lambda_hat_list), axis=0)

        return mean_log_lambda_hat

    def output_mean_rmse(self):

        mean_rmse = np.mean(np.stack(self._rmse_list), axis=0)

        return mean_rmse

    def output_mean_Y(self):

        mean_Y = np.mean(np.stack(self._Y_list), axis=0)

        return mean_Y

    def output_mean_Y_predict(self):

        mean_Y_predict = np.mean(np.stack(self._Y_predict_list), axis=0)

        return mean_Y_predict

    def clear(self):

        self._lm_list = []
        self._m0_tilde_list = []
        self._log_lambda_hat_list = []
        self._rmse_list = []
        self._Y_list = []
        self._Y_predict_list = []

        return self