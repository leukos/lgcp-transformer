import torch
import json
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from sklearn.preprocessing import StandardScaler


class LGCPDataSet(Dataset):
    """
    This class is a dataset loader for point samples from LGCP .
    """
    def __init__(self, path, parameters=['mu', 'var', 'scale'], standardize=True, discretize=False, num_cells=10, return_points=False, return_raw_points=False, return_L=False, num_points=None, num_samples=None, max_points=None):
        """
        Constructor for the LGCPDataSet class.
        :param path str: Path to the JSON file containing the LGCP samples.
        :param standardize bool: Whether to use standardization.
        :param discretize bool: Whether to create a 2D grid approximation for the intensity field.
        :param points bool: Add the raw points to the dataset.
        :param num_points int: How many points to include at most from each point sample. If #points < num_points the points will be sampled.
        :param max_points int: Indicates the maximum number of samples in any point pattern. Samples with more than max_ponits points will be discarded
        """
        self.standardize = standardize
        self.discretize = discretize
        self.return_points = return_points
        self.num_samples = num_samples
        self.num_cells = num_cells
        self.return_raw_points = return_raw_points
        self.return_L = return_L
        self.parameters = parameters

        with open(path, 'r') as f:
          data = json.load(f)

        # filter out data samples with more than num_points
        if max_points is not None:
          idx = np.squeeze(np.argwhere(np.array(data['N']) <= max_points))
          for param in parameters:
            data[param] = list(map(data[param].__getitem__, idx))
          data['X'] = list(map(data['X'].__getitem__, idx))
          data['Y'] = list(map(data['Y'].__getitem__, idx))
          data['N'] = list(map(data['N'].__getitem__, idx))
          data['L'] = list(map(data['L'].__getitem__, idx))

        # select a random subsample of the total available data of size num_samples
        if num_samples is not None:
          idx = np.random.choice(list(range(len(data['X']))), num_samples, replace=False)
          for param in parameters:
            data[param] = list(map(data[param].__getitem__, idx))
          data['X'] = list(map(data['X'].__getitem__, idx))
          data['Y'] = list(map(data['Y'].__getitem__, idx))
          data['N'] = list(map(data['N'].__getitem__, idx))
          data['L'] = list(map(data['L'].__getitem__, idx))

        if discretize:
          self.__discretize(data, num_cells)

        if return_raw_points:
          self.raw_points = []
          for d in zip(data['X'], data['Y']):
            self.raw_points.append(torch.tensor(np.array(d).transpose(), dtype=torch.float32))

        if type(standardize) == bool and not standardize and return_points:
          dim_points = max_points
          self.points = torch.zeros((len(data['X']), dim_points, 2))
          for i, d in enumerate(zip(data['X'], data['Y'])):
              self.points[i, :len(d[0]), :] = torch.tensor(np.array(d).transpose())

        self.n_raw = np.array(data['N'])
        if num_points:
          self.n_raw[self.n_raw >= num_points] = num_points

        if type(standardize) == bool and standardize:
          self.param_scalers = {}
          self.params = {}
          for param in parameters:
            self.param_scalers[param] = StandardScaler()
            self.params[param] = self.param_scalers[param].fit_transform(np.array(data[param])[:, np.newaxis])

          self.n_scaler = StandardScaler()
          self.n = self.n_scaler.fit_transform(np.array(data['N'])[:, np.newaxis])

          if return_L:
            self.L_scaler = StandardScaler()
            self.L = self.L_scaler.fit_transform(np.array(data['L']))

          if self.return_points:
            raw_points = []
            for d in zip(data['X'], data['Y']):
              raw_points.append(torch.tensor(np.array(d).transpose(), dtype=torch.float32))
            self.points_std, self.points_mean = torch.std_mean(torch.cat(raw_points, 0), 0)

            dim_points = max_points
            self.points = torch.zeros((len(data['X']), dim_points, 2))
            for i, d in enumerate(zip(data['X'], data['Y'])):
              self.points[i, :len(d[0]), :] = (torch.tensor(np.array(d).transpose()) - self.points_mean) / self.points_std

          if self.return_raw_points:
            self.raw_points_std, self.raw_points_mean = torch.std_mean(torch.cat(self.raw_points, 0), 0)
            for i, raw_p in enumerate(self.raw_points):
              self.raw_points[i] = (raw_p - self.raw_points_mean)/self.raw_points_std

          if discretize:
            self.m_std, self.m_mean = torch.std_mean(self.m)
            self.m = (self.m - self.m_mean) / self.m_std

        elif type(standardize) == dict:
          for param in parameters:
            self.params[param] = self.param_scalers[param].transform(np.array(data[param])[:, np.newaxis])

          self.n = standardize['n_scaler'].transform(np.array(data['N'])[:, np.newaxis])

          if return_L:
            self.L = standardize['L_scaler'].transform(np.array(data['L']))

          if return_points:
            dim_points = max_points
            self.points = torch.zeros((len(data['X']), dim_points, 2))
            for i, d in enumerate(zip(data['X'], data['Y'])):
              self.points[i, :len(d[0]), :] = (torch.tensor(np.array(d).transpose()) - standardize['points_mean']) / standardize['points_std']

          if self.return_raw_points:
            for i, raw_p in enumerate(self.raw_points):
              self.raw_points[i] = (raw_p - standardize['raw_points_mean']) / standardize['raw_points_std']

          if discretize:
            self.m = (self.m - standardize['m_mean']) / standardize['m_std']

        else:
          self.params = {}
          for param in parameters:
            self.params[param] = data[param]

          self.n = data['N']
          if return_L:
            self.L = data['L']

    def get_standardizers(self):

      scalers = {'param_scalers': self.param_scalers, 'n_scaler': self.n_scaler}
      if self.return_L:
        scalers['L_scaler'] = self.L_scaler

      if self.return_points:
        scalers['points_mean'] = self.points_mean
        scalers['points_std'] = self.points_std
      
      if self.return_raw_points:
        scalers['raw_points_mean'] = self.raw_points_mean
        scalers['raw_points_std'] = self.raw_points_std

      if self.discretize:
        scalers['m_mean'] = self.m_mean
        scalers['m_std'] = self.m_std

      return scalers
      

    def __getitem__(self, i):
      res = dict()
      res['n'] = torch.tensor(self.n[i], dtype=torch.float32)
      res['n_raw'] = self.n_raw[i]
      res['params'] = torch.tensor(np.array([self.params[param] for param in self.parameters]), dtype=torch.float32)


      if self.return_L:
        res['L'] = torch.tensor(self.L[i], dtype=torch.float32)
      if self.return_points:
        res['points'] = self.points[i, :, :]
      if self.return_raw_points:
        res['raw_points'] = self.raw_points[i]
      if self.discretize:
        res['grid'] = self.m[i, :, :]
      return res

    def __len__(self):
      return len(self.mu)

    def __discretize(self, data, num_cells):
      self.m = torch.zeros(len(data['X']), num_cells, num_cells, dtype=torch.float32)
      for i in range(len(data['X'])):
        xy = torch.stack([torch.tensor(data['X'][i]), torch.tensor(data['Y'][i])], dim=1)
        self.m[i, :, :] = torch.histogramdd(xy, bins=(num_cells, num_cells), range=(0.0, 1.0, 0.0, 1.0), dtype=torch.float32)

