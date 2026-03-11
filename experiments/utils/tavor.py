import os
import os.path as op

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


class DataLoader:
    """
    A generator class for loading data from a directory of NumPy files.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the NumPy files.
    batch_size : int
        The number of samples to yield per batch. Default is 1.
    idx : int or None, optional
        The indices of the samples to load. If None, all samples are loaded.
    ids : array-like or None, optional
        The IDs of the samples to load. If None, all samples are loaded.
    sample: int or None, optional
        The sample to load. If None, all samples are loaded.

    Yields
    ------
    batch : ndarray
        A batch of data with shape (batch_size, n_features), where n_features is the total number of features
        across all parcels.

    Raises
    ------
    ValueError
        If the data files have different numbers of features, or if the number of samples is not a multiple of
        the batch size.

    Examples
    --------
    >>> loader = DataLoader('data/', batch_size=2, idx=0)
    >>> for batch in loader:
    ...     # process batch of data
    ...     print(batch.shape)

    Notes
    -----
    This class assumes that the data is stored in NumPy files with the extension '.npy'. Each file should contain
    an array of shape (n_samples, n_parcels, n_voxels), where n_samples is the number of samples, n_parcels is the
    number of parcels, and n_voxels is the number of voxels per parcel. The data is flattened along the parcel and
    voxel dimensions to create a 2D array of shape (n_samples, n_features), where n_features is the total number of
    features across all parcels. If the `idx` parameter is not None, only the samples with the specified indices
    are loaded.

    """

    def __init__(
        self, data_path, batch_size=1, idx=None, ids=None, sample=None, random_seed=42
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.ids = ids
        self.idx = idx
        self.files = [
            op.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".npy")
        ]

        if self.ids is not None:
            self.files = [f for f in self.files if op.basename(f).split("_")[0] in ids]  # type: ignore

        if sample is not None:
            np.random.seed(random_seed)
            self._sample = np.random.choice(sample, 1)[0]
            self.files = [f for f in self.files if f"sample{self._sample}" in f]

    def __iter__(self):
        batch = []
        for file in self.files:
            data = np.load(file)
            data = data.reshape(data.shape[0], -1)
            if self.idx is not None:
                data = data[self.idx]
            batch.append(data)
            if len(batch) == self.batch_size:
                yield np.squeeze(np.stack(batch))
                batch = []
        if batch:
            yield np.squeeze(np.stack(batch))

    def __len__(self):
        return len(self.files) // self.batch_size


class TaskPredictionTavor:
    """
    A class for predicting task performance from fMRI data using the Tavor method.

    Parameters
    ----------
    n_jobs : int, optional
        The number of parallel jobs to run. Default is 1.

    Attributes
    ----------
    coef_ : ndarray
        The coefficients of the linear regression model.
    intercept_ : float
        The intercept of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit the linear regression model to the data.
    predict(X)
        Predict task performance from fMRI data.
    score(X, y, scoring=None)
        Compute the score of the linear regression model.

    Notes
    -----
    This class implements the Tavor method for predicting task performance from rs-fMRI connectome. The method
    involves normalizing the data such that each parcel has zero mean and unit variance across samples, fitting a
    linear regression model to the normalized data, and using the model to predict task performance from novel data.

    """

    def __init__(self, n_jobs=1):
        self._estimator = LinearRegression()
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Fit the linear regression model to the data.

        Parameters
        ----------
        X : ndarray, or DataLoader
            The fMRI data with shape (n_samples, n_voxels).
        y : ndarray, or DataLoader
            The task performance data with shape (n_voxels).

        """
        pbar = tqdm(zip(X, y), desc="Fitting...", total=len(X))
        self.coef_, self.intercept_ = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit)(X_subj, y_subj) for X_subj, y_subj in pbar
            )
        )
        self.coef_ = np.mean(self.coef_, axis=0)
        self.intercept_ = np.mean(self.intercept_, axis=0)
        self._estimator.coef_ = self.coef_
        self._estimator.intercept_ = self.intercept_
        return self

    def predict(self, X):
        """
        Predict task performance from fMRI data.

        Parameters
        ----------
        X : ndarray
            The fMRI data with shape (n_samples, n_voxels).

        Returns
        -------
        y_pred : ndarray
            The predicted task performance with shape (n_samples,).

        """
        y_pred = []
        for X_subj in X:
            self._check_input(X_subj)
            X_subj = self._normalize(X_subj).T
            y_pred.append(self._estimator.predict(X_subj))
        return np.array(y_pred)

    def score(self, X, y, scoring=None):
        """
        Compute the score of the linear regression model.

        Parameters
        ----------
        X : ndarray
            The fMRI data with shape (n_samples, n_voxels).
        y : ndarray
            The task performance data with shape (n_voxels).
        scoring : callable, optional
            The scoring function to use. Default is mean squared error.

        Returns
        -------
        score : float
            The score of the linear regression model.

        """
        if scoring is None:
            scoring = mean_squared_error

        y_pred = self.predict(X)
        return scoring(y, y_pred)

    def _fit(self, X, y):
        """Fit the linear regression model to a single subject's data."""
        self._check_input(X, y)
        X = self._normalize(X).T
        self._estimator.fit(X, y)
        return self._estimator.coef_, self._estimator.intercept_

    def _normalize(self, X):
        """Normalize X in a way that each parcel has zero mean and unit variance across samples."""
        return (X - X.mean(axis=-1, keepdims=True)) / X.std(axis=-1, keepdims=True)

    def _check_input(self, X, y=None):
        """Check that X is a 2D array of shape (n_samples, n_voxels) and y is a 1D array of shape (n_voxels)."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_voxels)")
        if y is not None and y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_voxels)")
        if y is not None and X.shape[1] != y.shape[0]:
            raise ValueError("Number of voxels in X and y must be the same")


def parallel_fit_wrapper(X, y, n_jobs=1):
    est = TaskPredictionTavor(n_jobs=n_jobs)
    return est.fit(X, y)
