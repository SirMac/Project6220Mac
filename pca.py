# INSE6220 Project
# Title: Water Potability Detection Using Principal Component Analysis And Machine Learning
# Data Source: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
# Author: Mac Gabriel Danyo
# Date: 2023-11-21



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv

# Allow printing of all data columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

sourceFile = open('printout.txt', 'w')
data = pd.read_csv("water_quality.csv")

# remove all the rows that contain null values
data = data.dropna()
# print(data.isnull().sum())
# print(data.head(), file=sourceFile)

# Visualize data
# Plot heat map of original data
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot= True, cmap='coolwarm')  #terrain
# plt.show()


# Pair plot
# plt.figure(figsize=(10, 8))
# sns.pairplot(data, hue="Potability")
# plt.show()



# Standardize data
x_std = StandardScaler().fit_transform(data)
print('Standardize data: \n',x_std[0:5, 0:10], file=sourceFile)

# Calculate covariance
mean_vec = np.mean(x_std, axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0]-1)
# print('Covariance: \n',cov_mat)

# Calculate eigenvalue and eigenvector
cov_mat = np.cov(x_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# print('Eigenvalues: \n', eig_vals)
# print('Eigenvector: \n', eig_vecs)

# sort eigenvalues: necessary?
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# for i in eig_pairs:
#     print(i[0])




# Plot varaince ratio
pca = PCA(n_components=2)
# pca.fit(x_std)
# pca = PCA(0.90)
x_pca = pca.fit_transform(x_std)

# pca = PCA().fit(data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_), '--bo')
# plt.gca().invert_yaxis()
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()



# Project original data to new axis
#Get the loadings of x and y axes
sns.set()
pca_df = pd.DataFrame(
    data=x_pca, 
    columns=['PC1', 'PC2']
)
pca_df_scaled = pca_df.copy()
features = list(data)
loadings = pca.components_
xs = loadings[0]
ys = loadings[1]

sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df_scaled, 
    fit_reg=False, 
)


# Plot the loadings on a scatterplot
# for i, varnames in enumerate(features):
#     plt.scatter(xs[i], ys[i], s=100)
#     plt.arrow(
#         0, 0, # coordinates of arrow base
#         xs[i], # length of the arrow along x
#         ys[i], # length of the arrow along y
#         color='r', 
#         head_width=0.06,
#         head_length=0.06, 
#         )
#     # plt.text(xs[i], ys[i], varnames)

# xticks = np.linspace(-0.8, 0.8, num=5)
# yticks = np.linspace(-0.8, 0.8, num=5)
# plt.xticks(xticks)
# plt.yticks(yticks)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('2D Biplot of projected data')
# plt.show()


# Plot scatter diagram
# plt.scatter(x_std[:, 0], x_std[:, 1], alpha=0.2)
# plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.8)
# plt.axis('equal')
# plt.show()