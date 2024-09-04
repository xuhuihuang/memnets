import torch
import torch.nn as nn
from tqdm import *
import numpy as np
from .loss import MEMLoss
from .utils import map_data
from ..processing.dataprocessing import Preprocessing, Postprocessing

class MEMLayer(nn.Module):
    """ Create MEMnets lobe.

    Parameters
    ----------
    layer_sizes : list
        The size of each layer of the encoder.
        The last component should represent the number of CVs.
    """

    def __init__(self, layer_sizes:list):
        super().__init__()
        
        encoder = [nn.BatchNorm1d(layer_sizes[0])]
        for i in range(len(layer_sizes)-1):
            encoder.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            encoder.append(nn.ELU()) if i<(len(layer_sizes)-2) else None
        # note that the last element in layer_sizes represent the num of cvs.
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        
        outputs = self.encoder(x)

        return outputs
    
class MEMModel:
    """ The MEMNet model from MEMNet.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the encoder outputs.

    device : torch device, default = None
        The device on which the torch modules are executed.

    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, device=None, dtype=np.float32):

        self._lobe = lobe
        if dtype == np.float32:
            self._lobe = self._lobe.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()

        self._dtype = dtype
        self._device = device

    @property
    def lobe(self):
        return self._lobe

    def transform(self, data, return_cv=True, tau=None):
        """ Transform the original trajectores to different outputs after training.

        Parameters
        ----------
        data : list or tuple or ndarray
            The original trajectories.

        return_cv : boolean
            Return the neural networks outputs or the collective variables.

        tau : int
            The lag time \tau to generate CVs from neural networks outputs.

        Returns
        -------
        output : array_like
            List of numpy array or numpy array containing transformed data.
        """

        self._lobe.eval()
        net = self._lobe

        output = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            output.append(net(data_tensor).cpu().numpy())

        if not return_cv:
            return output if len(output) > 1 else output[0]
        else:
            if tau is None:
                raise ValueError('Please input the lag time for transformation to CVs')
            else:
                post = Postprocessing(tau=tau, dtype=self._dtype)
                output_cv = post.fit_transform(output)
            return output_cv if len(output_cv) > 1 else output_cv[0]

class MEMNet:
    """ The method used to train MEMNet.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the encoder outputs.

    lagtimes : list
        Encoder lag times of MEMnets.

    optimizer : str, default = 'Adam'
        The type of optimizer used for training.

    device : torch.device, default = None
        The device on which the torch modules are executed.

    learning_rate : float, default = 5e-4
        The learning rate of the optimizer.

    decay_rate : float, default = 5e-3
        The exponential decay rate in the dynamic scheme of gamma.

    thres : float, default = 0.015
        The threshold (i.e., gamma-0.5) to switch gamma to 0.5.

    epsilon : float, default = 1e-6
        The strength of the regularization/truncation under which matrices are inverted.

    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.

    save_model_interval : int, default = None
        Saving the model every 'save_model_interval' epochs.

    reversible : boolean, default = True
        Whether to enforce detailed balance constraint. 

    print : boolean, default = False
        Whether to print the validation loss every epoch during the training. 
    """

    def __init__(self, lobe, lagtimes, optimizer='Adam', device=None, learning_rate=5e-4, decay_rate=0.005, thres=0.015,
                 epsilon=1e-6, mode='regularize', dtype=np.float32, save_model_interval=None, reversible=True, print=False):

        self._lobe = lobe
        self._lagtimes = lagtimes
        self._device = device
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._thres = thres
        self._pre_train = 1.
        self._epsilon = epsilon
        self._mode = mode
        self._dtype = dtype
        self._save_model_interval = save_model_interval
        self._reversible = reversible
        self._print = print

        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()

        self._step = 0
        self._save_models = []
        optimizer_types = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
        if optimizer not in optimizer_types.keys():
            raise ValueError(f"Unknown optimizer type, supported types are {optimizer_types.keys()}")
        else:
            self._optimizer = optimizer_types[optimizer](self._lobe.parameters(), lr=self._learning_rate)

        self._memloss = MEMLoss(lagtimes=lagtimes, device=device, epsilon=epsilon, mode=mode, reversible=self._reversible)

        self._training_loss = []
        self._training_lm = []
        self._training_m0_tilde = []
        self._training_log_lambda_hat = []
        self._training_rmse = []
        self._training_gamma = []

        self._validation_lm = []
        self._validation_m0_tilde = []
        self._validation_log_lambda_hat = []
        self._validation_rmse = []
        self._validation_Y = []
        self._validation_Y_predict = []
        self._validation_gamma = []

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
        self._optimizer.param_groups[0]['lr'] = self._learning_rate

    @property
    def training_loss(self):
        return np.array(self._training_loss)

    @property
    def training_lm(self):
        return np.array(self._training_lm)

    @property
    def training_m0_tilde(self):
        return np.array(self._training_m0_tilde)

    @property
    def training_log_lambda_hat(self):
        return np.array(self._training_log_lambda_hat)

    @property
    def training_rmse(self):
        return np.array(self._training_rmse)

    @property
    def training_gamma(self):
        return np.array(self._training_gamma)

    @property
    def validation_lm(self):
        return np.array(self._validation_lm)

    @property
    def validation_m0_tilde(self):
        return np.array(self._validation_m0_tilde)

    @property
    def validation_log_lambda_hat(self):
        return np.array(self._validation_log_lambda_hat)

    @property
    def validation_rmse(self):
        return np.array(self._validation_rmse)
    
    @property
    def validation_Y(self):
        return np.array(self._validation_Y)
    
    @property
    def validation_Y_predict(self):
        return np.array(self._validation_Y_predict)

    @property
    def validation_gamma(self):
        return np.array(self._validation_gamma)

    def partial_fit(self, data):

        self._lobe.train()
        self._optimizer.zero_grad()

        x = [self._lobe(data[i]) for i in range(len(data))]
        gamma = 0.5+1.5*np.exp(-self._decay_rate*self._step)*self._pre_train
        loss = self._memloss.fit(x).loss(weight=gamma)

        loss.backward()
        self._optimizer.step()

        self._training_loss.append(loss.item())
        self._training_lm.append(self._memloss.lm(weight=gamma).detach().cpu().numpy())
        self._training_m0_tilde.append(self._memloss.m0_tilde.detach().cpu().numpy())
        self._training_log_lambda_hat.append(self._memloss.log_lambda_hat.detach().cpu().numpy())
        self._training_rmse.append(self._memloss.rmse.detach().cpu().numpy())
        self._training_gamma.append(gamma)

        self._step += 1

        return self

    def validate(self, val_data):

        self._lobe.eval()

        with torch.no_grad():
            val_outputs = [self._lobe(val_data[i]) for i in range(len(val_data))]
            self._memloss.fit(val_outputs)
            self._memloss.save(weight=(0.5+1.5*np.exp(-self._decay_rate*self._step)*self._pre_train))

            return self

    def fit(self, train_loader, n_epochs=1, validation_loader=None, progress=tqdm):
        """ Performs fit on data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Yield batch of time-series for training.

        n_epochs : int, default=1
            The number of epochs to use for training.

        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Yield batch of time-series for validation.

        progress : context manager, default=tqdm

        Returns
        -------
        self : MEMNet
        """

        self._step = 0

        for epoch in progress(range(n_epochs), desc="epoch", total=n_epochs, leave=False):

            for batches in train_loader:
                batches = [batches[i].to(device=self._device) for i in range(len(batches))]
                self.partial_fit(batches)

            if self._pre_train != 0.:
                gamma = 0.5+1.5*np.exp(-self._decay_rate*self._step)*self._pre_train
                if gamma <= 0.5+self._thres:
                    self._pre_train = 0.
            else:
                gamma = 0.5

            if validation_loader is not None:
                with torch.no_grad():

                    for val_batches in validation_loader:
                        val_batches = [val_batches[i].to(device=self._device) for i in range(len(val_batches))]
                        self.validate(val_batches)

                    mean_lm = self._memloss.output_mean_lm()
                    mean_m0_tilde = self._memloss.output_mean_m0_tilde()
                    mean_log_lambda_hat = self._memloss.output_mean_log_lambda_hat()
                    mean_rmse = self._memloss.output_mean_rmse()

                    mean_Y = self._memloss.output_mean_Y()
                    mean_Y_predict = self._memloss.output_mean_Y_predict()

                    self._validation_lm.append(mean_lm)
                    self._validation_m0_tilde.append(mean_m0_tilde)
                    self._validation_log_lambda_hat.append(mean_log_lambda_hat)
                    self._validation_rmse.append(mean_rmse)
                    self._validation_gamma.append(gamma)

                    self._validation_Y.append(mean_Y)
                    self._validation_Y_predict.append(mean_Y_predict)

                    self._memloss.clear()

                    if self._print:
                        print(epoch, gamma, mean_lm, mean_log_lambda_hat)
                    
                    if self._save_model_interval is not None:
                        if (epoch + 1) % self._save_model_interval == 0:
                            m = self.fetch_model()
                            self._save_models.append(m)

        return self

    def fetch_model(self) -> MEMModel:

        from copy import deepcopy
        lobe = deepcopy(self._lobe)
        return MEMModel(lobe, device=self._device, dtype=self._dtype)

class MEMEstimator:
    """ The MEMNet estimator generate the MEMnets time scales and other results from original trajectories.

    Parameters
    ----------
    memnet_model : MEMModel
        The trained MEMModel model.

    lagtimes : list
        Encoder lag times of MEMnets.
    """

    def __init__(self, memnet_model: MEMModel, lagtimes):

        self._model = memnet_model
        self._lagtimes = lagtimes
        self._time_scales = None
        self._log_lambda_hat = None
        self._m0_tilde = None
        self._lm = None
        self._Y = None
        self._Y_predict = None
        self._rmse = None
        self._is_fitted = False

    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._time_scales
        
    @property
    def log_lambda_hat(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._log_lambda_hat
        
    @property
    def m0_tilde(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._m0_tilde
        
    @property
    def lm(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._lm
        
    @property
    def Y(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._Y
        
    @property
    def Y_predict(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._Y_predict
        
    @property
    def rmse(self):
        if not self._is_fitted:
            raise ValueError('Please fit the model first')
        else:
            return self._rmse
    
    def fit(self, data):
        """ Fit the MEMNet model with original trajectories.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        Returns
        -------
        self : MEMEstimator
        """

        outputs = self._model.transform(data, return_cv=False)

        pre = Preprocessing()
        dataset = pre.create_time_series_dataset(outputs, self._lagtimes)
        
        data = [torch.from_numpy(np.array(i)).to(device='cpu') for i in zip(*dataset)]

        estimator = MEMLoss(lagtimes=self._lagtimes,device='cpu')
        estimator.fit(data).save(weight=0.5)
        self._time_scales = 1/np.abs(estimator.output_mean_log_lambda_hat())
        self._log_lambda_hat = estimator.output_mean_log_lambda_hat()
        self._m0_tilde = estimator.output_mean_m0_tilde()
        self._lm = estimator.output_mean_lm()
        self._Y = estimator.output_mean_Y()
        self._Y_predict = estimator.output_mean_Y_predict()
        self._rmse = estimator.output_mean_rmse()

        self._is_fitted = True

        return self