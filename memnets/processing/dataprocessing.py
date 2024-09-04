import numpy as np
from ..utils import rao_blackwell_ledoit_wolf

class Preprocessing:
    """ Preprocess the original trajectories to create datasets for training.

    Parameters
    ----------
    dtype : dtype, default = np.float32
    """

    def __init__(self, dtype=np.float32):

        self._dtype = dtype

    def _seq_trajs(self, data):

        data = data.copy()
        if not isinstance(data, list):
            data = [data]
        for i in range(len(data)):
            data[i] = data[i].astype(self._dtype)
        
        return data   

    def create_dataset(self, data, lagtime):
        """ Create the dataset with one-step transition.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        lagtime : int
            The lagtime used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has two elements: one is the instantaneous data frame, the other is the corresponding time-lagged data frame.
        """

        data = self._seq_trajs(data)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lagtime
            for i in range(L_re):
                dataset.append((data[k][i,:], data[k][i+lagtime,:]))

        return dataset  
    
    def create_time_series_dataset(self, data, lagtimes):
        """ Create the dataset as the input to MEMnets.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        lagtimes : list
            List of lagtimes to create the time-series dataset.
            [\delta t, n_1\delta t, ... , n_k\delta t].

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has k+2 elements: the first one is the instantaneous data frame, 
            the rest are the time-lagged data frames at [\delta t, n_1\delta t, ... , n_k\delta t].
        """
        
        if not isinstance(lagtimes, list):
            lagtimes = [lagtimes]
        num_lagtimes = len(lagtimes)

        data = self._seq_trajs(data)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lagtimes[-1]
            for i in range(L_re):
                tmp = [data[k][i, :]]
                for j in range(num_lagtimes):
                    tmp.append(data[k][i + lagtimes[j], :])
                tmp = tuple(tmp)
                dataset.append(tmp)

        return dataset

    
class Postprocessing(Preprocessing):
    """ Transform the outputs from neural networks to slow CVs.

    Parameters
    ----------
    tau : int
        The lag time used for transformation.

    dtype : dtype, default = np.float32

    shrinkage : boolean, default = True
        To tell whether to do the shrinkaged estimation of covariance matrix. 
    
    n_dims : int, default = None
        The number of slow collective variables to obtain.
    """
    
    def __init__(self, tau=1, dtype=np.float32, shrinkage=True, n_dims=None):
        super().__init__(dtype)
        self._n_dims = n_dims
        self._tau = tau
        self._dtype = dtype
        self._shrinkage = shrinkage

        self._is_fitted = False
        self._mean = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._time_scales = None

    @property
    def shrinkage(self):
        return self._shrinkage

    @shrinkage.setter
    def shrinkage(self, value: bool):
        self._shrinkage = value

    @property
    def tau(self):
        return self._tau
    
    @tau.setter
    def tau(self, value: int):
        self._tau = value

    @property
    def mean(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._mean

    @property
    def eigenvalues(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._eigenvectors
    
    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._time_scales

    def fit(self, data):
        """ Fit the model for transformation.

        Parameters
        ----------
        data : list or ndarray
        """
        
        self._mean = self._cal_mean(data)
        self._eigenvalues, self._eigenvectors = self._cal_eigvals_eigvecs(data)
        self._time_scales = -self._tau / np.log(np.abs(self._eigenvalues))
        self._is_fitted = True
        
        return self

    def _cal_mean(self, data):

        dataset = self.create_dataset(data, self._tau)
        d0, d1 = map(np.array, zip(*dataset))
        mean = (d0.mean(0) + d1.mean(0)) / 2.

        return mean

    def _cal_cov_matrices(self, data):

        num_trajs = 1 if not isinstance(data, list) else len(data)
        dataset = self.create_dataset(data, self._tau)

        batch_size = len(dataset)
        d0, d1 = map(np.array, zip(*dataset))

        mean = 0.5 * (d0.mean(0) + d1.mean(0))

        d0_rm = d0 - mean
        d1_rm = d1 - mean

        c00 = 1. / batch_size * np.dot(d0_rm.T, d0_rm)
        c11 = 1. / batch_size * np.dot(d1_rm.T, d1_rm)
        c01 = 1. / batch_size * np.dot(d0_rm.T, d1_rm)
        c10 = 1. / batch_size * np.dot(d1_rm.T, d0_rm)

        c0 = 0.5 * (c00 + c11)
        c1 = 0.5 * (c01 + c10)

        if self.shrinkage:
            n_observations_ = batch_size + self._tau * num_trajs
            c0, _ = rao_blackwell_ledoit_wolf(c0, n_observations_)

        return c0, c1

    def _cal_eigvals_eigvecs(self, data):

        c0, c1 = self._cal_cov_matrices(data)

        import scipy.linalg
        eigvals, eigvecs = scipy.linalg.eigh(c1, b=c0)

        idx = np.argsort(eigvals)[::-1]

        if self._n_dims is not None:
            assert self._n_dims <= len(idx)
            idx = idx[:self._n_dims]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return eigvals, eigvecs

    def transform(self, data):
        """ Transfrom the basis funtions (or outputs of neural networks) to the slow CVs.
            Note that the model must be fitted first before transformation.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = []

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data = self._seq_trajs(data)
        num_trajs = len(data)

        for i in range(num_trajs):
            x_rm = data[i] - self._mean
            modes.append(np.dot(x_rm, self._eigenvectors).astype(self._dtype))

        return modes if num_trajs > 1 else modes[0]
    
    def fit_transform(self, data):
        """ Fit the model and transfrom to the slow CVs.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = self.fit(data).transform(data)

        return modes