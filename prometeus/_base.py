from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from kneed import KneeLocator
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

__all__ = ['BaseAnalyzer', 'BasePlot', 'CUTOFF_METHOD']

CUTOFF_METHOD = Literal["cum_var", "scree"]


class BaseAnalyzer(ABC):
    @abstractmethod
    def get_loadings(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_clustering(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_cumvariance_crossover(cum_variance: Union[List[float], np.ndarray], threshold: float = 0.8) -> int:
        """ Get number of components prior to threshold crossover """
        assert cum_variance[0] < threshold, "Please check if current dataset have been normalized!"
        return len([c for c in cum_variance if c <= threshold]) + 1

    @staticmethod
    def _get_eigen_values_crossover(eigen_values: Union[List[float], np.ndarray], threshold: float = 1) -> int:
        assert min(eigen_values) <= 1, "Please check if current dataset have been normalized!"
        return len([e for e in eigen_values if e >= threshold])

    @staticmethod
    def _get_cluster_size(inertias: List[float]) -> int:
        """
        Get location of knee, which is turning point of under-fitting to over-fitting,
        according to inertias from Elbow graph . This will determine the optimal cluster size.
        """
        x = range(2, len(inertias) + 2)
        y = inertias

        kn = KneeLocator(
            x,
            y,
            curve='convex',
            direction='decreasing',
            interp_method='interp1d',
        )
        if kn.knee is not None:
            return kn.knee

        # second deravitives
        tmp = []
        for i in range(1, len(inertias) - 1):
            tmp.append(inertias[i + 1] + inertias[i - 1] - 2 * inertias[i])

        s = {v: k for k, v in enumerate(tmp, 3)}

        return s[max(tmp)]

    @staticmethod
    def _generate_intertias(n_range: int, df: pd.DataFrame) -> Tuple[range, List[float]]:
        """
        Prior preprocessing to extract information to draw elbow graph
        """
        clusters_range = range(2, n_range + 1)
        inertias = []

        for c in clusters_range:
            kmeans = KMeans(n_clusters=c, random_state=0).fit(df)
            inertias.append(kmeans.inertia_)

        return clusters_range, inertias

    @staticmethod
    def _generate_silhouette_score(n_range: int, df: pd.DataFrame) -> Tuple[range, List[float]]:
        """
        Prior preprocessing to extract information to draw silhouette graph
        """
        clusters_range = range(2, n_range + 1)
        score = []

        for c in clusters_range:
            clusterer = KMeans(n_clusters=c, random_state=0)
            y = clusterer.fit_predict(df)
            score.append(round(silhouette_score(df, y) * 100, 2))

        return clusters_range, score

    @staticmethod
    def _generate_kmean_labels(cluster_size: int, df: pd.DataFrame) -> pd.DataFrame:
        kmeans_sel = KMeans(n_clusters=cluster_size, random_state=1).fit(df)
        labels = pd.DataFrame(kmeans_sel.labels_)
        labels.columns = ['Cluster']
        return labels

    def _process_loading_matrix(self, loading_matrix: pd.DataFrame, is_filter: bool) -> pd.DataFrame:
        _loading_matrix = loading_matrix
        if is_filter:
            tmp_loadings = self._filter_loadings(loading_matrix)
            if len(tmp_loadings) == 0:
                print("No loadings > 0.55, thus no filter will be apply! Skipping process.....")
            else:
                _loading_matrix = tmp_loadings
        _loading_matrix['highest_loading'] = _loading_matrix.idxmax(axis=1)
        return _loading_matrix.sort_values('highest_loading')

    @staticmethod
    def _filter_loadings(loadings_df: pd.DataFrame) -> pd.DataFrame:
        """ Filter loadings > 55%, which are consider good loadings """
        loadings_df['flags'] = loadings_df.apply(lambda x: 1 if max(x.values) > 0.55 else 0, axis=1)
        return loadings_df[loadings_df['flags'] == 1].drop(columns=['flags'])


class BasePlot(ABC):
    @abstractmethod
    def generate_graphs(self, **kwargs):
        raise NotImplementedError

    def _plot_cum_variance(self, threshold: float, x: np.ndarray, y: np.ndarray) ->  go.Figure:
        """
        Cummulative explanatory variance plot

        :param threshold:
            Percentage of variance explained, default 80%
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines+markers',
                                 name="Cummulative Variance",
                                 hovertemplate="%{y}%"))

        fig.add_shape(go.layout.Shape(type="line",
                                      xref="paper", yref="y",
                                      x0=0, x1=1, y0=threshold, y1=threshold,
                                      line=dict(color="red", dash="dash")))

        fig.update_layout(title=dict(text="Cummulative Variance Plot"),
                          xaxis=dict(title="Principal Components"),
                          yaxis=dict(title="Variance Explained (%)", rangemode="tozero"),
                          hovermode="x",
                          showlegend=False)

        return fig

    def _plot_elbow(self, clusters_range: range, inertias: List[float]) -> go.Figure:
        """ Elbow method plot """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(clusters_range),
                                 y=inertias,
                                 mode='lines+markers',
                                 name="Elbow Plot",
                                 hovertemplate="%{y}%"))

        fig.update_layout(title=dict(text="Elbow Plot"),
                          xaxis=dict(title="Cluster Size, K"),
                          yaxis=dict(title="Sum of Squared Distances", rangemode="tozero"),
                          hovermode="x",
                          showlegend=False)

        return fig

    def _plot_silhouette(self, clusters_range: range, inertias: List[float]) -> go.Figure:
        """ Silhouette method plot """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(clusters_range),
                                 y=inertias,
                                 mode='lines+markers',
                                 name="Silhouett Plot",
                                 hovertemplate="%{y}%"))

        fig.update_layout(title=dict(text="Silhouett Plot"),
                          xaxis=dict(title="Cluster Size, K"),
                          yaxis=dict(title="Silhouett's Score", rangemode="tozero"),
                          hovermode="x",
                          showlegend=False)

        return fig

    def _plot_scree(self, x: List[str], y: List[float]):
        """ Scree Plot """
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines+markers',
                                 name="Scree Plot",
                                 hovertemplate="%{y}%"))

        fig.update_layout(title=dict(text="Scree Plot"),
                          xaxis=dict(title="Principal Components"),
                          yaxis=dict(title="Proportion of Variance Explained", rangemode="tozero"),
                          hovermode="x",
                          showlegend=False)

        fig.add_shape(go.layout.Shape(type="line",
                                      xref="paper", yref="y",
                                      x0=0, x1=1, y0=1, y1=1,
                                      line=dict(color="red", dash="dash")))

        return fig

    def _plot_loading_heatmap(self, loadings: pd.DataFrame) ->  go.Figure:
        """
        PCA loading heatmap visualization

        :param loadings:
            loadings dataframe input. Can be generated from PCAnalyzer or FAnalyzer
        """
        df = loadings
        arr = df.T
        fig = go.Figure(ff.create_annotated_heatmap(z=arr.values,
                                                    x=arr.columns.tolist(),
                                                    y=arr.index.tolist(),
                                                    annotation_text=arr.values,
                                                    hoverinfo="text",
                                                    colorscale='RdBu',
                                                    showscale=True,
                                                    xgap=0.5,
                                                    ygap=0.5))
        fig.update_layout(title=dict(text="Loadings Heatmap"),
                          xaxis=dict(title="Columns Attribute", side="bottom", type="category"),
                          yaxis=dict(title="Principal Components", rangemode="tozero"),
                          hovermode="x",
                          showlegend=False)
        return fig

    def _generate_loadings_summary_table(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """ Get summary table for loadings DataFrame"""
        df = loadings.T
        mapper = {k: v for k, v in enumerate(df.columns)}
        df['observations > 0.55'] = df.apply(lambda x: ', '.join([mapper[i] for i, v in enumerate(x.values) if v > 0.55]) ,
                                      axis=1)
        df = df[df['observations > 0.55'] != ''][['observations > 0.55']]
        return df

    def _plot_scatter_3d(self, transformed_data: pd.DataFrame, cluster_labels: pd.Series, total_var: float) ->  go.Figure:
        """ 3D scatter plot for clustered data """
        fig = px.scatter_3d(
            transformed_data, x=0, y=1, z=2, color=cluster_labels,
            title=f'3D scatter plot - Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.update_layout(
            legend_title_text="Cluster",
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        return fig

    def _plot_scatter_2d(self, transformed_data: pd.DataFrame, cluster_labels: pd.Series, total_var: float) ->  go.Figure:
        """ 2D scatter plot for clustered data """
        fig = px.scatter(
            transformed_data, x=0, y=1, color=cluster_labels,
            title=f'2D scatter plot - Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2'}
        )
        fig.update_layout(
            legend_title_text="Cluster",
            legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
        return fig

    def _generate_scores_table(self, data: pd.DataFrame, clustered_lables: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary table for top 3 PCA scores according to clusters

        :param n_components:
            Number of principal components. If None will autogenerated using PCAnalyzer Library, else overwrite.
        """
        pr = data
        cl = clustered_lables
        assert len(pr) == len(cl), \
            "Mismatch clustered labels data length and principal component scores generated dataframe length!"
        df = pd.concat([pr, cl], axis=1)
        tmp = df.groupby('cluster')[[c for c in df.columns if c != 'cluster']].mean()
        return self._get_top_scores(tmp)

    @staticmethod
    def _validate_clusters_data(df: Union[List[Union[str, float]], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """ Check format for clusters data and convert to pandas DataFrame """
        if isinstance(df, pd.Series):
            assert len(df.shape) == 1, "please ensure that you only pass 1 column of pandas Series!"
            return df.to_frame().rename(columns={df.name: 'cluster'})
        if isinstance(df, list):
            return pd.DataFrame({"cluster": [str(int(x)) for x in df]})
        assert df.shape[1] == 1, "please ensure that you only pass 1 column of pandas DataFrame!"
        return df.rename(columns={df.columns[0]: 'cluster'})

    @staticmethod
    def _get_top_scores(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """
        :param df:
            pandas DataFrame grouped by clustered with mean of PCA scores.
        :param top_n:
            top n values of PCA as key and PCA scores as values

        Examples
        --------------
                                                          top 1                                    top 3
        cluster                                                     ...
        0        [(principal component 9, 0.21), (principal com...  ...   (principal component 17, 0.07)
        1        [(principal component 7, 11.79), (principal co...  ...     (principal component 9, 5.4)
        2        [(principal component 7, 1.56), (principal com...  ...   (principal component 10, 0.67)
        3        [(principal component 13, 22.17), (principal c...  ...  (principal component 10, 10.21)
        4        [(principal component 4, 3.76), (principal com...  ...   (principal component 19, 0.67)
        5        [(principal component 17, 6.75), (principal co...  ...    (principal component 7, 2.19)
        6        [(principal component 1, 2.6), (principal comp...  ...    (principal component 9, 0.16)
        7        [(principal component 2, 9.42), (principal com...  ...    (principal component 4, 0.48)
        8        [(principal component 12, 10.94), (principal c...  ...   (principal component 18, 1.94)
        9        [(principal component 3, 2.29), (principal com...  ...    (principal component 4, 1.29)
        10       [(principal component 14, 132.8), (principal c...  ...  (principal component 11, 28.56)
        11       [(principal component 11, 2.47), (principal co...  ...     (principal component 5, 1.2)
        [12 rows x 4 columns]
        """
        fn_top = lambda row, nlargest=top_n: sorted(pd.Series(zip(df.columns, round(row, 2))), key=lambda cv: -cv[1])[
                                                  :nlargest]
        res = pd.DataFrame(df.apply(fn_top, axis=1)).reset_index()
        res['cluster'] = res['cluster'].astype(int)
        res.sort_values(by=['cluster'], inplace=True)
        for i in range(0, top_n):
            res[f'top {i + 1}'] = res[0].apply(lambda x: x[i])
        return res.set_index('cluster')[[f'top {i+1}' for i in range(0, top_n)]]
