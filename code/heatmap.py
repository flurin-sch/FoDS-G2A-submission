import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
 
# sample_info = pd.read_csv('../data/sample_info.csv')
radiosensitivity = pd.read_csv('../data/radiosensitivity.csv')
expression = pd.read_csv('../data/expressionData.csv')
 
# Extract cell line names and set as index
expression.set_index("cell_line_name", inplace=True)
 
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(expression)
scaled_df = pd.DataFrame(scaled_data, columns=expression.columns, index=expression.index)
 
# Randomize the rows and columns
randomized_df = scaled_df.sample(frac=1, axis=0).sample(frac=1, axis=1)
 
# # Plot heatmap before clustering
# plt.figure(figsize=(12, 10))
# sns.heatmap(randomized_df, cmap='plasma', center=0, annot=False, cbar_kws={'label': 'Expression Level'})
# plt.title("Heatmap Before Clustering (Randomized)")
# plt.xlabel("Genes")
# plt.ylabel("Cell Lines")
# plt.tight_layout()
# plt.savefig("../output/gene_expression_heatmap_random.png")
 
# Perform hierarchical clustering on rows and columns separately
linkage_matrix_rows = linkage(randomized_df, method='ward')
linkage_matrix_cols = linkage(randomized_df.T, method='ward')
 
# Get the order of the rows and columns
dendro_rows = dendrogram(linkage_matrix_rows, no_plot=True)
dendro_cols = dendrogram(linkage_matrix_cols, no_plot=True)
ordered_df = randomized_df.iloc[dendro_rows['leaves'], :].iloc[:, dendro_cols['leaves']]
 
# Plot heatmap after clustering
plt.figure(figsize=(12, 10))
sns.heatmap(ordered_df, cmap='plasma', center=0,
            annot=False, cbar_kws={'label': 'Expression Level [z-score]'})
plt.title("Heatmap of Gene Expression After Hierarchical Clustering")
plt.xlabel("Genes")
plt.ylabel("Cell Lines")
plt.tight_layout()
plt.savefig("../output/gene_expression_heatmap_clustered.png")