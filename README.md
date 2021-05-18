## This package uses Python 3.8

## Prometeus Library
Prometeus library is purely only for PCA(Principal Component Analysis) / FA (Factor Analysis) + provide 8 unique plots related to PCA for visualization 
and decision making purposes. Plotly graphs including: Cummulative Variance Plot, Elbow Plot, Screee Plot, 
PCA loadings Heatmap, 2D and 3D  clustered Scatter Plot; 2D and 3D bi-plots images being saved at same directory.

The library also automate PCA loadings and K-Mean clustering dataflow pipelines for ease of obtaining the results.

When using the library, please remember to normalize the dataset being utilized to obtain reasonable results.


## PCA Plots
#### Examples:
```python
from typing import List, Tuple

import pandas as pd

from prometeus.pca.plot import PCAPlot
from sklearn.preprocessing import StandardScaler

def preprocess_data_cluster(df: pd.DataFrame, clusters: List[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    df.dropna(how='any', inplace=True)
    if clusters is not None:
        df = df[df['Cluster'].isin(clusters)]
    clusters = df['Cluster'].astype(int).astype(str)
    if 'Unnamed: 0' not in df.columns:
        df.drop(columns=['SERVICE', 'MONTH_SQN', 'CUST_SQN', 'Cluster'], inplace=True)
    else:
        df.drop(columns=['Unnamed: 0', 'SERVICE', 'MONTH_SQN', 'CUST_SQN', 'Cluster'], inplace=True)
    return df, clusters

normalize_col = ['SH_CUST_RATING', 'CUST_LVL_MOB_ARPU_6M', 'TENURE_MOBILE', 'CUST_LVL_MOB_CNT', 'CONTRACT_DURATION', 'SUBS_TENURE',
                'LIFETIME_CONTRACTS_CNT', 'AUSAGE_SOCIALNET', 'AUSAGE_VIDEO', 'AUSAGE_COMMUNICATIONS', 'AUSAGE_NETFLIX',
                'AUSAGE_ECOMMERCE', 'AUSAGE_GAMES', 'AUSAGE_MUSIC', 'AUSAGE_MAIL','AUSAGE_LIFESTYLE', 'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION',
                'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION', 'USAGE_SOCIALNET', 'USAGE_STREAMINGVIDEO', 'USAGE_GAMES', 'USAGE_ENTERTAINMENT',
                'USAGE_MUSIC', 'USAGE_TRANSPORTATION', 'USAGE_SHOPPING', 'USAGE_ECOMMERCE', 'USAGE_SPORTS', 'USAGE_NEWS', 'USAGE_TRAVEL']

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df[normalize_col])
    scaled_target = df[normalize_col].copy()

    df[normalize_col]= scaler.transform(scaled_target)
    return df.fillna(0)

df = pd.read_csv('Data/clustered_data_20210414.csv')
df, clusters = preprocess_data_cluster(df)

PCAPlot(normalize(df), clusters).generate_graphs(n_components=22, is_filter=True)
```

## PCA Loadings Extraction
#### Examples:

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler
from prometeus.pca.analyzer import PCAnalyzer

normalize_col = ['SH_CUST_RATING', 'CUST_LVL_MOB_ARPU_6M', 'TENURE_MOBILE', 'CUST_LVL_MOB_CNT', 'CONTRACT_DURATION', 'SUBS_TENURE',
                'LIFETIME_CONTRACTS_CNT', 'AUSAGE_SOCIALNET', 'AUSAGE_VIDEO', 'AUSAGE_COMMUNICATIONS', 'AUSAGE_NETFLIX',
                'AUSAGE_ECOMMERCE', 'AUSAGE_GAMES', 'AUSAGE_MUSIC', 'AUSAGE_MAIL','AUSAGE_LIFESTYLE', 'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION',
                'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION', 'USAGE_SOCIALNET', 'USAGE_STREAMINGVIDEO', 'USAGE_GAMES', 'USAGE_ENTERTAINMENT',
                'USAGE_MUSIC', 'USAGE_TRANSPORTATION', 'USAGE_SHOPPING', 'USAGE_ECOMMERCE', 'USAGE_SPORTS', 'USAGE_NEWS', 'USAGE_TRAVEL']

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df[normalize_col])
    scaled_target = df[normalize_col].copy()

    df[normalize_col]= scaler.transform(scaled_target)
    return df.fillna(0)

df = pd.read_csv('Data/clustered_data_20210414.csv')

PCAnalyzer(normalize(df)).get_loadings(by="cum_var", threshold=0.8, n_components=None, is_filter=False)

```

## KMean Clustering Results
#### Examples:

```python
import pandas as pd

from sklearn.preprocessing import StandardScaler
from prometeus.pca.analyzer import PCAnalyzer

normalize_col = ['SH_CUST_RATING', 'CUST_LVL_MOB_ARPU_6M', 'TENURE_MOBILE', 'CUST_LVL_MOB_CNT', 'CONTRACT_DURATION', 'SUBS_TENURE',
                'LIFETIME_CONTRACTS_CNT', 'AUSAGE_SOCIALNET', 'AUSAGE_VIDEO', 'AUSAGE_COMMUNICATIONS', 'AUSAGE_NETFLIX',
                'AUSAGE_ECOMMERCE', 'AUSAGE_GAMES', 'AUSAGE_MUSIC', 'AUSAGE_MAIL','AUSAGE_LIFESTYLE', 'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION',
                'AUSAGE_NEWS', 'AUSAGE_TRANSPORTATION', 'USAGE_SOCIALNET', 'USAGE_STREAMINGVIDEO', 'USAGE_GAMES', 'USAGE_ENTERTAINMENT',
                'USAGE_MUSIC', 'USAGE_TRANSPORTATION', 'USAGE_SHOPPING', 'USAGE_ECOMMERCE', 'USAGE_SPORTS', 'USAGE_NEWS', 'USAGE_TRAVEL']

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df[normalize_col])
    scaled_target = df[normalize_col].copy()

    df[normalize_col]= scaler.transform(scaled_target)
    return df.fillna(0)

df = pd.read_csv('Data/clustered_data_20210414.csv')

PCAnalyzer(normalize(df)).get_clustering(by="cum_var", threshold=0.8, n_components=None, cluster_size=None)

```