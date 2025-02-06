# Clustering Analysis on Airline Passenger Satisfaction Dataset

## Overview
This project performs clustering analysis on the "Airline Passenger Satisfaction" dataset. It includes data cleaning, exploratory data analysis, dimensionality reduction using PCA, and clustering using K-Means and Agglomerative Clustering.

## Dataset
- Source: Kaggle ([teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction))
- The dataset contains passenger details and their satisfaction ratings.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SciPy
- pydot

## Steps in the Project
### 1. Data Cleaning and Exploration
- Load dataset using `kagglehub`
- Remove missing values and duplicates
- Drop unnecessary columns (`id`, `Unnamed: 0`)
- One-hot encode categorical variables
- Visualize feature correlations using a heatmap

### 2. Dimensionality Reduction
- Use Principal Component Analysis (PCA) to reduce feature dimensions before clustering

### 3. Clustering Methods
#### K-Means Clustering
- Use the Elbow Method to determine the optimal number of clusters
- Apply K-Means clustering with `k-means++` initialization
- Visualize clustered data with principal components

#### Agglomerative Clustering
- Take a random sample of 20,000 records for efficiency
- Apply PCA for dimensionality reduction
- Determine the number of clusters using a dendrogram
- Visualize clustering results

## How to Run the Code
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy kagglehub pydot
   ```
2. Run the script:
   ```bash
   python clustering.py
   ```

## Results
- K-Means successfully clustered the dataset into two distinct groups
- Agglomerative clustering confirmed the separation of clusters with hierarchical visualization

## Future Improvements
- Experiment with different clustering algorithms (DBSCAN, Gaussian Mixture Models)
- Optimize feature selection for better clustering performance
- Use t-SNE or UMAP for better visualization of high-dimensional data

