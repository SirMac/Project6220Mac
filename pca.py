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


data = pd.read_csv("water_quality.csv")
# print(data.head())


# remove all the rows that contain null values
data = data.dropna()
# print(data.isnull().sum())

# Visualize data
# Plot heat map of original data
# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot= True, cmap='coolwarm')  #terrain
# plt.show()


# Pair plot
# plt.figure(figsize=(10, 8))
# sns.pairplot(data, hue="Potability Pair Plot")
# plt.show()



# Standardize data
x_std = StandardScaler().fit_transform(data)
# print('Standardize data: \n',x_std)

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

pca = PCA().fit(data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_), '--bo')
# plt.gca().invert_yaxis()
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()


loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# fig = px.scatter(x_pca, x=0, y=1, color=data['Potability'])
# fig = px.scatter(x_pca)
# features = list(data)
# print(features)
X = np.arange(0,10,1)
Y = np.arange(0,10,1)
plt.scatter(x_std[:, 0], x_std[:, 1])
# plt.quiver(X,Y, eig_vecs[:,1], eig_vecs[:,0], zorder=11, width=0.01, scale=3)
# for i, feature in enumerate(features):
#     fig.add_annotation(
#         ax=0, ay=1,
#         axref="x", ayref="y",
#         x=loadings[i, 4],
#         y=loadings[i, 5],
#         showarrow=True,
#         arrowsize=2,
#         arrowhead=1,
#         xanchor="right",
#         yanchor="top"
#     )
# plt.show()



# Project original data to new axis
# def arrow(v1, v2, ax):
#     arrowprops=dict(arrowstyle='->',
#                    linewidth=2,
#                    shrinkA=0, shrinkB=0)
#     ax.annotate("", v2, v1, arrowprops=arrowprops)

# proj_data = eig_vecs.T.dot(x_std.T)
# print(proj_data)
# plt.scatter(proj_data[:, 0], proj_data[:, 1])
# fig, axes = plt.subplots(1,2, figsize=(12,4))
# axes[0].axis('equal')
# axes[0].scatter(x_std[:,0], x_std[:,1])
# for l, v in zip(pca.explained_variance_, pca.components_):
#     arrow([0,0], v*l*3, axes)
# plt.show()


# Plot vector
# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops=dict(arrowstyle='->',
#                     linewidth=2,
#                     shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)

# # plot data
# plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal')
# plt.show()



# Plot scatter diagram
# plt.scatter(x_std[:, 0], x_std[:, 1], alpha=0.2)
# plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.8)
# plt.axis('equal')
# plt.show()