"""
This script clusters freighting customers. 
To do this, it injests cleaned data, processes it, trains a k-means clustering algorithm, and visulaises the results. 

It two different clusters:
1. Current Business
2. Future Growth

Expected Inputs:
- Excel file ('carrier_dataset.xlsx') containing carrier performance data

Expected Output:
- CSV file ('output_labelled_data.csv') with clustered customer segments
- Evaluation metrics via visualizations and descriptive statistics

Dependencies:
- pandas, matplotlib, seaborn, scikit-learn
"""

import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any

# set styles and colours
plt.style.use('fivethirtyeight')

# Initialize an empty dictionary to store data
data: Dict[str, Any] = {
    'df_raw': pd.DataFrame,
    'df_scaled_crr_labled': pd.DataFrame,
    'df_scaled_ftr_labled': pd.DataFrame,
    'ls_feats_crr_rel': [],
    'ls_feats_ftr_gwth': [],
    'ls_mapping_cluster_crr': [],
    'ls_mapping_cluster_ftr': [],
    'scaler_crr': StandardScaler(),
    'scaler_ftr': StandardScaler()
}

data['ls_feats_crr_rel'] = [  # Define the features for current relationship clustering
    'total_volume',
    'automated_booked_volume',
    'n_corridors_served',
    'avg_gps_tracked_percent',
    'avg_on_time_performance'
]

data['ls_feats_ftr_gwth'] = [  # Define the features for future growth clustering
    'avg_tractor_count',
    'total_volume',
    'automated_booked_volume',
    'n_corridors_served',
    'avg_gps_tracked_percent',
    'avg_on_time_performance',
    'load_volume_growth_rate'
]

def clc_grwth_rate(
    summary_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame: 
    """ Calculate carrier growth rates from first to most recent """
    first_month = df.groupby('carrier_code')['year_month'].transform('min')
    last_month = df.groupby('carrier_code')['year_month'].transform('max')

    # Filter the data to include only the first and last months for each carrier
    first_month_data = df[df['year_month'] == first_month]
    last_month_data = df[df['year_month'] == last_month]

    # Merge the first and last month data to calculate load volume growth rate
    merged_data = pd.merge(
        first_month_data, 
        last_month_data, 
        on='carrier_code', 
        suffixes=('_first', '_last'))

    # Calculate load volume growth rate from the first to the last month for each carrier
    merged_data['load_volume_growth_rate'] = ((merged_data['load_volume_last'] - merged_data['load_volume_first']) /
                                            merged_data['load_volume_first']) * 100

    # Display carrier-wise load volume growth rates from the first to the last month
    carrier_growth_rates = merged_data[['carrier_code', 'load_volume_growth_rate']]
    
    # Fill in infintie or NaN
    carrier_growth_rates['load_volume_growth_rate'].replace([np.inf, -np.inf], 999, inplace=True)
    carrier_growth_rates['load_volume_growth_rate'].fillna(0, inplace=True)

    return pd.merge(summary_df, carrier_growth_rates, how='left')


def scale_features(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """Scales the features"""
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[list_of_columns])
    scaled_and_labelled_df = pd.DataFrame(scaled_df, columns=list_of_columns)
    return scaled_and_labelled_df, scaler


def scale_and_assess_feature_correlation(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """Once scaled, this assesses the correlation of the features"""
    scaled_and_labelled_df, scaler = scale_features(df, list_of_columns)
    correlation_matrix = scaled_and_labelled_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    return scaled_and_labelled_df, scaler


def assess_clstrs_elbw(df: pd.DataFrame) -> None:
    """Assesses n of K means using Elbow method"""
    inertia = []
    for k in range(1, 11):  # Testing different values of K from 1 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow curve
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()


def assess_clstrs_sil(df: pd.DataFrame) -> None:
    """Assesses n of K means using Silhoutte Score"""
    
    silhouette_scores = []
    for k in range(2, 11): 
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        silhouette_scores.append(silhouette_score(df, kmeans.labels_))

    # Plotting the Silhouette Scores
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')
    plt.show()


def compute_gap_statistic(data, k) -> None:
    """Computes Gap Statistics"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    inertia = kmeans.inertia_
    reference_inertias = []
    
    for _ in range(10):  # Generate reference null distribution (here, 10 reference datasets)
        reference_data = np.random.random_sample(data.shape)  # Generate random data with same shape as original data
        reference_kmeans = KMeans(n_clusters=k, random_state=42)
        reference_kmeans.fit(reference_data)
        reference_inertias.append(reference_kmeans.inertia_)
    
    gap = np.mean(np.log(reference_inertias)) - np.log(inertia)
    return gap


def cmpt_gp_sts_ks(df) -> None:
    """Assesses n of K means using calculated Gap Statistics"""
    gaps = []
    for k in range(1, 11):  # Testing different values of K from 1 to 10
        gap = compute_gap_statistic(df, k)
        gaps.append(gap)

    # Plotting the Gap Statistics
    plt.plot(range(1, 11), gaps, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistics for Optimal K')
    plt.show()
    

def assign_clusters(
    df: pd.DataFrame, crr=True) -> pd.DataFrame:
    """Assigns clusters based upon n given"""
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    return df['Cluster']

    
def visualise_permiations(
    column_list: list) -> None:
    """Generate all permutations of feature combinations, and visulises"""
    feature_combinations = list(itertools.permutations(column_list, 2)) 

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=len(feature_combinations)//3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (feat_x, feat_y) in enumerate(feature_combinations):
        sns.scatterplot(
            x=data['df_carrier_summary'][feat_x],
            y=data['df_carrier_summary'][feat_y],
            hue=data['df_carrier_summary']['cluster_crr'],
            palette='viridis',
            ax=axes[i]
        )
        axes[i].set_title(f"{feat_x} vs {feat_y}")
        axes[i].set_xlabel(feat_x)
        axes[i].set_ylabel(feat_y)

    plt.tight_layout()
    plt.show()
    

def plot_means_and_std(
    df: pd.DataFrame, column_names: list, cluster_column: str, title='') -> None:
    """Grouping the DataFrame by the cluster column, calculating means and std, and visualising"""
    grouped_stats = df.groupby(cluster_column)[column_names].agg(['mean', 'std'])

    # Creating subplots for each specified column
    num_plots = len(column_names)
    num_cols = 2  
    num_rows = (num_plots + 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Handling issues
    if num_plots == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = [axes]

    # Plotting the means and standard deviations for each specified column in separate subplots
    for i, column in enumerate(column_names):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row][col]

        # Plotting horizontal bar plots of means with error bars representing standard deviations
        y = grouped_stats.index
        x = grouped_stats[column, 'mean']
        xerr = grouped_stats[column, 'std']
        ax.barh(y, x, xerr=xerr, capsize=5, alpha=0.7, label=f'{column}')
        ax.set_title(f'Mean and SD of {column}')
        ax.set_xlabel(f'{column}')
        ax.legend()

    if title == 'current':
        plt.suptitle('Evalutation Metrics for Current Relationship', fontsize=28)
    else:
        plt.suptitle('Evaluation Metrics for Future Growth', fontsize=28)
    plt.tight_layout()
    plt.show()


# Section 1: Import data
data['df_raw'] = pd.read_excel("data/raw/carrier_dataset.xlsx")

# Set dtypes
data['df_raw']['year_month'] = pd.to_datetime(data['df_raw']['year_month'])

# Look
data['df_raw'].describe()
data['df_raw'].info()

# Sectiion 2: Process
data['df_carrier_summary'] = data['df_raw'].groupby('carrier_code').agg(
    avg_tractor_count=('tractor_count', 'mean'),
    total_volume=('load_volume', 'sum'),
    avg_automated_booked_ratio=('automated_booked_volume', 'mean'),  # Assuming this column exists
    n_corridors_served=('corridors', 'nunique'),
    avg_gps_tracked_percent=('gps_tracked_percent', 'mean'), 
    avg_on_time_performance=('on_time_performance', 'mean'),
    ).reset_index()

data['df_carrier_summary']['automated_booked_volume'] = data['df_carrier_summary']['total_volume'] * data['df_carrier_summary']['avg_automated_booked_ratio']
data['df_carrier_summary'] = clc_grwth_rate(data['df_carrier_summary'], data['df_raw'])


# Section 3: Standardise the features and check the correlation
data['df_scaled_crr_labled'], data['scaler_crr'] = scale_and_assess_feature_correlation(data['df_carrier_summary'], data['ls_feats_crr_rel']) # none of these are over 0.8 correlation 
data['df_scaled_ftr_labled'], data['scaler_crr'] = scale_and_assess_feature_correlation(data['df_carrier_summary'], data['ls_feats_ftr_gwth']) # none of these are over 0.8 correlation 


# Section 4: Assess the number of clusters
## Elbow method for K means optimisation Start ##
assess_clstrs_elbw(data['df_scaled_crr_labled']) # Best is 3 or 4
assess_clstrs_elbw(data['df_scaled_ftr_labled']) # Best is 4


cmpt_gp_sts_ks(data['df_scaled_crr_labled']) # Best is 3
cmpt_gp_sts_ks(data['df_scaled_ftr_labled']) # Best is 3


## Silhoutte Score for K means optimisation Start ##
assess_clstrs_sil(data['df_scaled_crr_labled'])  # Best is 4
assess_clstrs_sil(data['df_scaled_ftr_labled'])  # Best is 4  


# Section 5.1: Assign clusters using K-Means  
data['df_carrier_summary']['cluster_crr'] = assign_clusters(data['df_scaled_crr_labled'])
data['df_carrier_summary']['cluster_ftr'] = assign_clusters(data['df_scaled_ftr_labled'])


# Section 5.2: Visualiise the Current Relationships and Future Growth clusters on a few features
# # Create subplots for Current Relationships 
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

# # Plot 1: total_volume vs avg_gps_tracked_percent with cluster labels
# for cluster, color in zip(data['df_carrier_summary']['cluster_name_crr'].unique(), ['blue', 'orange', 'green']):
#     cluster_data = data['df_carrier_summary'][data['df_carrier_summary']['cluster_name_crr'] == cluster]
#     axes[0].scatter(cluster_data['total_volume'], cluster_data['avg_gps_tracked_percent'], label=cluster, color=color)

# axes[0].set_xlabel('Total Volume')
# axes[0].set_ylabel('Average GPS Tracked Percent')
# axes[0].set_title('Total Volume vs Average GPS Tracked Percent')
# axes[0].legend()

# # Plot 2: avg_on_time_performance vs avg_gps_tracked_percent with cluster labels
# for cluster, color in zip(data['df_carrier_summary']['cluster_name_crr'].unique(), ['blue', 'orange', 'green']):
#     cluster_data = data['df_carrier_summary'][data['df_carrier_summary']['cluster_name_crr'] == cluster]
#     axes[1].scatter(cluster_data['avg_on_time_performance'], cluster_data['avg_gps_tracked_percent'], label=cluster, color=color)

# axes[1].set_xlabel('Average On-Time Performance')
# axes[1].set_ylabel('Average GPS Tracked Percent')
# axes[1].set_title('Average On-Time Performance vs Average GPS Tracked Percent')
# axes[1].legend()

# plt.suptitle('Clusters for Current Relationship', fontsize=28)
# plt.tight_layout()
# plt.show()


# Section 6: Visualise and relabel the clusters  
visualise_permiations(data['ls_feats_crr_rel'])
visualise_permiations(data['ls_feats_ftr_gwth'])


# Relabel the clusters
data['ls_mapping_cluster_crr'] = {
    0: 'Saturated', #  high volume, high gps, mostly on time'
    1: 'Small, but could grow', # low volume, low gps, mixed on time'
    2: 'Big players', # high volume, high gps, mostly on time'
}

data['ls_mapping_cluster_ftr'] = {
    0: 'Small',
    1: 'Medium and growth potential',
    2: 'Big and automated', # Large fleet, total volumn, corridors
}

data['df_carrier_summary']['cluster_name_crr'] = data['df_carrier_summary']['cluster_crr'].map(data['ls_mapping_cluster_crr'])
data['df_carrier_summary']['cluster_name_ftr'] = data['df_carrier_summary']['cluster_ftr'].map(data['ls_mapping_cluster_ftr'])


# Section 7: Visualise to describe the clusters 
plot_means_and_std(data['df_carrier_summary'], data['ls_feats_crr_rel'], 'cluster_name_crr', 'current')
plot_means_and_std(data['df_carrier_summary'], data['ls_feats_ftr_gwth'], 'cluster_name_ftr')


# Section 8: Export labelled data 
data['df_carrier_summary'].to_csv('data/processed/output_labelled_data.csv')