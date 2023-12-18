# INSE6220 Project
# Title: Water Potability Detection Using Principal Component Analysis And Machine Learning
# Data Source: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability
# Author: Mac Gabriel Danyo
# Term: Fall 2023



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pycaret.classification import *

# Allow printing of all data columns
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)


sourceFile = open('printout.txt', 'w')
data = pd.read_csv("water_quality.csv")

# remove all the rows that contain null values
data = data.dropna()
print(data.head())
# print(data.isnull().sum())


# Distribution of Unsafe and Safe Water
ax = sns.countplot(x = "Potability",data= data, saturation=0.8)
plt.xticks(ticks=[0, 1], labels = ["Not Potable", "Potable"])
plt.show()

# Standardize data
x_std = StandardScaler().fit_transform(data)
print('Standardize data: \n',np.around(x_std[0:5, 0:10],4))

# Calculate covariance
mean_vec = np.mean(x_std, axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0]-1)
print('Covariance: \n',np.around(cov_mat,4))

# Calculate eigenvalue and eigenvector
cov_mat = np.cov(x_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvalues: \n', np.around(eig_vals,4))
print('Eigenvectors: \n', np.around(eig_vecs,4))


# PCA Explained Variance Scree Plot
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_std)

pca_fit = PCA().fit(data)
plt.plot(np.cumsum(pca_fit.explained_variance_ratio_), '--bo')
plt.gca().invert_yaxis()
plt.title('PCA Explained Variance Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance %')
plt.show()


# # Pareto chart of explained variance
pca = PCA(0.99)
x_pca = pca.fit_transform(x_std)
explained_varainces = pca.explained_variance_ratio_

# Compute the cumulative explained variance
print(explained_varainces)
cumulative_variances = np.cumsum(explained_varainces)

# Create the bar plot for individual variances
plt.figure(figsize=(12, 7))
bar = plt.bar(range(1, 11), explained_varainces, alpha=0.6, color='g', label='Individual Explained Variance')

# Create the line plot for cumulative variance
line = plt.plot(range(1, 11), cumulative_variances, marker='o', linestyle='-', color='r', 
                label='Cumulative Explained Variance')

# Adding percentage values on top of bars and dots
for i, (bar, cum_val) in enumerate(zip(bar, cumulative_variances)):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_varainces[i]*100:.1f}%', 
             ha='center', va='bottom')
    plt.text(i+1, cum_val, f'{cum_val*100:.1f}%', ha='center', va='bottom')

plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.xticks(range(1, 11))
plt.legend(loc='upper left')
plt.ylim(0, 1.1) 
plt.grid(True)
plt.show()



# Project original data to new axis
#Get the loadings of x and y axes
pca = PCA(n_components=2)   #reset pca
x_pca = pca.fit_transform(x_std)
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
for i, varnames in enumerate(features):
    plt.scatter(xs[i], ys[i], s=100)
    plt.arrow(
        0, 0, # coordinates of arrow base
        xs[i], # length of the arrow along x
        ys[i], # length of the arrow along y
        color='r', 
        head_width=0.06,
        head_length=0.06, 
        )
    # plt.text(xs[i], ys[i], varnames)

xticks = np.linspace(-0.5, 1, num=2)
yticks = np.linspace(-0.5, 1, num=2)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D Biplot of projected data')
plt.show()





#----------Machine Learning----------------

# Review Of Dataset Features
data.describe().T

# Visualize data
def visualize_data():
    # Plot heat map of original data
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot= True, cmap='coolwarm')  #terrain
    plt.show()


    # Pair plot
    plt.figure(figsize=(10, 8))
    sns.pairplot(data, hue="Potability")
    plt.show()
visualize_data()

# Checking for outliers (box plot)
df = data
fig, ax = plt.subplots(ncols = 5, nrows = 2, figsize = (20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad = 0.5, w_pad=0.7, h_pad=5.0)

# Checking for outliers (Histogram)
plt.rcParams['figure.figsize'] = [20,10]
df.hist()
plt.show()


# Selecting a Model
X = df.drop("Potability", axis=1).values
y = df["Potability"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# ------------Random Forest Algorithm---------------
# Acurracy of Random Forest Algorithm
classifier_RF = RandomForestClassifier()
classifier_RF = classifier_RF.fit(X_train, y_train)
y_pred_RF = classifier_RF.predict(X_test)
Accuracy_RF = accuracy_score(y_test,y_pred_RF)
# print("Model Accuracy of Random Forest Algorithm:",Accuracy_RF)


# Confusion Matrix for Random Forest Algorithm
result = confusion_matrix(y_test, y_pred_RF)
f, ax=plt.subplots(figsize=(5,5))
sns.heatmap(result,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Classification Report for Random Forest Algorithm
classification_rpt = classification_report(y_test, y_pred_RF)
print("Classification Report:",)
print (classification_rpt)


# ---------Quadratic Discriminant Analysis-------
# Acurracy of Quadratic Discriminant Analysis
classifier_QDA = QuadraticDiscriminantAnalysis()
classifier_QDA = classifier_QDA.fit(X_train, y_train)
y_pred_QDA = classifier_QDA.predict(X_test)
Accuracy_QDA = accuracy_score(y_test,y_pred_QDA)
# print("Model Accuracy of Random Forest Algorithm:",Accuracy_QDA)


# Confusion Quadratic Discriminant Analysis
result = confusion_matrix(y_test, y_pred_QDA)
f, ax=plt.subplots(figsize=(5,5))
sns.heatmap(result,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Classification Report for Quadratic Discriminant Analysis
classification_rpt = classification_report(y_test, y_pred_QDA)
print("Classification Report:",)
print (classification_rpt)




clf = setup(data, target = "Potability", session_id = 786)
compare_models()

model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()