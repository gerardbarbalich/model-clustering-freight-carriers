# Freight Customer Clustering Script

This Python script clusters freight customers based on their carrier performance data. It imports data, processes data, clusters customers, and visualizes the results.

## Script Overview

### Data Import

The script imports data from a CSV file (`'data/raw/carrier_dataset.xlsx'`) containing load volume, automated booked volume, number of corridors served, GPS tracking percentage, and on-time performance data for each freight customer.

### Data Processing

The script performs the following preprocessing steps:

1. **Extracts relevant features:** Extracts the specified features from the raw data.
2. **Calculates load volume growth rate:** Calculates the load volume growth rate for each customer between the first and the last month of their data.
3. **Standardizes features:** Standardizes the extracted features using `StandardScaler` to ensure a standard scale for the clustering algorithm.

### Clustering

The script clusters customers using the K-means clustering algorithm. It identifies the optimal number of clusters using elbow method, gap statistic, and silhouette score.

### Visualization

The script visualizes the clustered data:

1. **Visualizes relationships between features and clusters:** Creates scatter plots for each pair of features to examine the relationships between the features and the clusters.
2. **Visualizes means and standard deviations:** Creates bar plots for each feature to compare the means and standard deviations of the features across the clusters.

### Exporting Results

The script exports the processed data to a CSV file (`'data/processed/output_labelled_data.csv'`), along with the cluster labels for each customer.

## Usage

1. Ensure the configuration file (`config-preprocess-data.csv`) is available in the 'data/raw/' directory.
2. Update the file paths for input and output as needed.
3. Run the script to preprocess and cluster the data.

## Example Usage

```python
python freight_customer_clustering.py