from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets

from factor_analyzer import FactorAnalyzer

from prometeus.utils import *
from .analyzer import FAnalyzer
from .._base import CUTOFF_METHOD, BasePlot

__all__ = ['FAPlot']


class FAPlot(FAnalyzer, BasePlot):
    def __init__(self, df: pd.DataFrame, clustered_labels: Optional[pd.Series] = None):
        super().__init__(df)
        self.clustered_labels = self._validate_clusters_data(clustered_labels) if clustered_labels else None

    def generate_graphs(self, by: CUTOFF_METHOD, threshold: float = 0.8, n_factors: Optional[int] = None,
                        is_filter: bool = False) -> widgets.Tab:
        """
        Create tabs in jupyter notebook and saves graphs in directory

        :param threshold:
            Percentage of variance explained, default 80%
        :param n_factors:
            Number of factors
        :param is_filter:
            If False will show all PCA loadings heatmap. If True, will only show attributes with cells > 0.55.
        """
        _factors = self._get_factors(by, threshold) if n_factors is None else n_factors
        fa_loadings = self.get_loadings(by, threshold, n_factors, is_filter).drop('highest_loading', axis=1).round(
            3)

        res = {
            'Cutoff Plots': [self.plot_cum_variance(threshold), self.scree_plot],
            'PCA Loading Heatmap': [self.plot_loading_heatmap(fa_loadings),
                                    self.generate_pca_loadings_summary_table(fa_loadings)],
            'Clusters Decision Plot': [self.plot_elbow(_factors),
                                       self.plot_silhouette(_factors)] if self.clustered_labels is not None else None,
            'Scatter Plots': [self.scatter_2d, self.scatter_3d,
                              self.generate_top_3_fa_scores_table(_factors)]
                              if self.clustered_labels is not None else None,
        }
        return create_tabs_widgets(res)

    def plot_cum_variance(self, threshold: float) -> go.Figure:
        """
        Cummulative explanatory variance plot

        :param threshold:
            Percentage of variance explanined, default 80%
        """
        x = np.arange(1, self.df.shape[1] + 1, step=1)
        y = self.cum_var

        return self._plot_cum_variance(threshold, x, y)

    def plot_elbow(self, n_factors: int) -> go.Figure:
        """ Elbow method plot """
        clusters_range, inertias = self._get_elbow_info(n_factors)

        return self._plot_elbow(clusters_range, inertias)

    def plot_silhouette(self, n_components: int) -> go.Figure:
        """ Silhouette plot """
        factorslDf = self._get_factor_df(n_components)
        clusters_range, score = self._generate_silhouette_score(n_components, factorslDf)

        return self._plot_silhouette(clusters_range, score)

    @property
    def scree_plot(self):
        """ Scree Plot """
        fa_list = ["FA" + str(i) for i in list(range(1, self.df.shape[1] + 1))]

        return self._plot_scree(fa_list, self.sum_sq)

    def plot_loading_heatmap(self, fa_loadings: pd.DataFrame) -> go.Figure:
        """
        PCA loading heatmap visualization

        :param fa_loadings:
            FA loadings dataframe input. Can be generated from FAnalyzer
        """
        return self._plot_loading_heatmap(fa_loadings)

    def generate_pca_loadings_summary_table(self, fa_loadings: pd.DataFrame) -> pd.DataFrame:
        """ Get summary table for PCA loadings """
        return self._generate_loadings_summary_table(fa_loadings)

    @property
    def scatter_3d(self) -> go.Figure:
        """ 3D scatter plot for clustered data """
        fa = FactorAnalyzer(rotation='varimax', n_factors=3, method='ml')
        components = fa.fit_transform(self.df)
        total_var = self.pro_var.sum() * 100
        return self._plot_scatter_3d(components, self.clustered_labels.cluster, total_var)

    @property
    def scatter_2d(self) -> go.Figure:
        """ 2D scatter plot for clustered data """
        fa = FactorAnalyzer(rotation='varimax', n_factors=2, method='ml')
        components = fa.fit_transform(self.df)
        total_var = self.pro_var.sum() * 100
        return self._plot_scatter_2d(components, self.clustered_labels.cluster, total_var)

    def generate_top_3_fa_scores_table(self, n_factors: int = None) -> pd.DataFrame:
        """
        Get summary table for top 3 PCA scores according to clusters

        :param n_factors:
            Number of factors
        """
        return self._generate_scores_table(self._get_factor_df(n_factors), self.clustered_labels)
