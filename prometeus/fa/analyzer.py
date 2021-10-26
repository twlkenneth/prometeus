from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from factor_analyzer import FactorAnalyzer

from .._base import BaseAnalyzer, CUTOFF_METHOD

__all__ = ['FAnalyzer']


class FAnalyzer(BaseAnalyzer):
    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna()[[c for c in df.columns if df[c].sum() != 0]]
        self.sum_sq, self.pro_var, self.cum_var = self._get_variance_info()

    def get_loadings(self, by: CUTOFF_METHOD = "scree", threshold: float = 1, n_factors: Optional[int] = None,
                        is_filter: bool = False) -> pd.DataFrame:
        """
        Get PCA loading dataframe

        :param by:
            Cutoff method using Cummulative Variance Plot or Scree Plot
        :param threshold:
            Percentage of variance explained, default 80%
        :param n_factors:
            Number of factors.
        :param is_filter:
            If False will show all PCA loadings heatmap. If True, will only show attributes with cells > 0.55.
        """
        df = self.df
        _factors = self._get_factors(by, threshold) if n_factors is None else n_factors
        fa = FactorAnalyzer(rotation='varimax', n_factors=_factors, method='ml')
        fa.fit(df)
        fa_loading_matrix = pd.DataFrame(fa.loadings_, columns=[f'FA{i}' for i in range(1, _factors + 1)],
                                         index=df.columns)
        return self._process_loading_matrix(fa_loading_matrix, is_filter)

    def get_clustering(self, by: CUTOFF_METHOD, threshold: float = 0.8, n_factors: Optional[int] = None,
                       cluster_size: Optional[int] = None) -> pd.DataFrame:
        """
        Get original DataFrame with cluster labelling

        :param by:
            Cutoff method using Cummulative Variance Plot or Scree Plot
        :param threshold:
            Percentage of variance explained, default 80%
        :param n_factors:
            Number of factors.
        :param cluster_size:
            Number of cluster size desired. If None will autogenerated using PCAnalyzer Library determined by elbow
            method, else overwrite.
        """
        _factors = self._get_factors(by, threshold) if n_factors is None else n_factors
        df = self.df.reset_index(drop=True)
        factorDf  = self._get_factor_df(_factors)
        clusters_range, inertias = self._get_elbow_info(_factors)
        _cluster_size = self._get_cluster_size(inertias) if cluster_size is None else cluster_size
        labels = self._generate_kmean_labels(_cluster_size, factorDf)
        return pd.concat([df, labels], axis=1)

    def _get_factors(self, by: CUTOFF_METHOD, threshold: float = 0.8) -> int:
        """
        Determine number of components require after Dimensional Reduction

        :param by:
            Cutoff method using Cummulative Variance Plot or Scree Plot
        :param threshold:
            Percentage of variance explained, default 80%
        """
        if by == 'cum_var':
            if threshold < 0.75:
                print("WARNING: Please be advised to have at least 75% of variance being explained!")
            return self._get_cumvariance_crossover(self.cum_var, threshold)
        return self._get_eigen_values_crossover(self.sum_sq, 1)

    def _get_variance_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a Tuple consisting of 3 arrays:
        1. Sum of squared loadings (variance)
        2. Proportional variance
        3. Cumulative variance
        """
        fa = FactorAnalyzer(rotation=None)
        # Factor analysis cannot have whole column sum = 0
        fa.fit(self.df)
        return fa.get_factor_variance()

    def _get_factor_df(self, n_factors: int) -> pd.DataFrame:
        """
        Dimension reduced pandas Dataframe from original dataframe

        :param n_factors:
            Number of factors
        """
        fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors, method='ml')
        factorDf = fa.fit(self.df)
        return pd.DataFrame(data=factorDf
                            , columns=[f'factor {i}' for i in range(1, n_factors + 1)])

    def _get_elbow_info(self, n_factors: int) -> Tuple[range, List[float]]:
        """
        Prior preprocessing to extract information to draw elbow graph

        :param threshold:
            Percentage of variance explained, default 80%
        :param components:
            Number of principal components. If None will autogenerated using PCAnalyzer Library determined by threshold,
             else overwrite.
        """
        factorDf = self._get_factor_df(n_factors)
        return self._generate_intertias(n_factors, factorDf)
