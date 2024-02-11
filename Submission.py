#################################################
# Imports
#################################################
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from Decision_Trees import dt_generate_learning_curves, dt_generate_validation_curves, dt_analysis
from Neural_Networks import nn_generate_learning_curves, nn_generate_validation_curves, nn_analysis
from Boosted_Decision_Trees import bdt_generate_learning_curves, bdt_generate_validation_curves, bdt_analysis
from Support_Vector_Machines import svm_generate_learning_curves, svm_generate_validation_curves, svm_analysis
from K_Nearest_Neighbors import knn_generate_learning_curves, knn_generate_validation_curves, knn_analysis

#################################################
# Variables
#################################################
seed_value = None # Change to 'None' for random seed
random.seed(seed_value)
#################################################


#################################################
# Description of Classification Problems
#
# The following code will import the wine quality
# dataset, clean/preprocess the data, and then
# analyze the resulting dataset
#################################################
wine_quality = pd.read_csv(r"Datasets/wine_quality.csv")

# Clean data
wine_quality = wine_quality.drop_duplicates()   # Remove rows that contain duplicate data
wine_quality = wine_quality.dropna()            # Remove rows that contain missing values

# Examine the size/shape of the dataset
rows, cols = wine_quality.shape
print(f'There are {rows} rows and {cols} columns in Wine Quality Dataset.')

# Observe outliers
feature_columns = wine_quality.columns[:-1].tolist()

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))
axes = axes.flatten()

for i, column in enumerate(feature_columns):
    sns.boxplot(x=wine_quality[column], ax=axes[i])
    axes[i].set(xlabel=None)    # Remove x-label
    axes[i].set_xticks([])      # Remove existing x-axis ticks
    axes[i].set_xticklabels([]) # Remove existing x-axis tick labels
    axes[i].set_yticks([])      # Remove existing y-axis ticks
    axes[i].tick_params(axis='x', rotation=90)  # Set the rotation angle
    axes[i].set_title(column)
    axes[i].title.set_size(30)

fig.delaxes(axes[11])   # Remove empty graph

plt.subplots_adjust(hspace=0.5)
plt.savefig('Images/Dataset_Analysis/Wine_Quality_Outliers.png')
plt.close()

# Manually handle outliers
lower_limits = {'fixed acidity': 4.0,  'volatile acidity': 0.1, 'citric acid': 0.0, 'residual sugar': 0.0,  'chlorides': 0.0, 'free sulfur dioxide': 0,  'total sulfur dioxide': 0,   'density': 0.9, 'ph': 3.0, 'sulphates': 0.3, 'alcohol': 8.0}
upper_limits = {'fixed acidity': 15.0, 'volatile acidity': 1.1, 'citric acid': 0.8, 'residual sugar': 10.0, 'chlorides': 0.3, 'free sulfur dioxide': 60, 'total sulfur dioxide': 170, 'density': 1.1, 'ph': 4.0, 'sulphates': 1.5, 'alcohol': 13.6}

for column in wine_quality.columns:
        lower_limit = lower_limits.get(column, None)
        upper_limit = upper_limits.get(column, None)        
        if lower_limit is not None and upper_limit is not None:
            wine_quality[column] = np.where((wine_quality[column] < lower_limit) | (wine_quality[column] > upper_limit), wine_quality[column].mean(), wine_quality[column])

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))
axes = axes.flatten()

for i, column in enumerate(feature_columns):
    sns.boxplot(x=wine_quality[column], ax=axes[i])
    axes[i].set(xlabel=None)    # Remove x-label
    axes[i].set_xticks([])      # Remove existing x-axis ticks
    axes[i].set_xticklabels([]) # Remove existing x-axis tick labels
    axes[i].set_yticks([])      # Remove existing y-axis ticks
    axes[i].tick_params(axis='x', rotation=90)  # Set the rotation angle
    axes[i].set_title(column)
    axes[i].title.set_size(30)

fig.delaxes(axes[11])   # Remove empty graph

plt.subplots_adjust(hspace=0.5)
plt.savefig('Images/Dataset_Analysis/Wine_Quality_Outliers_Adjusted.png')
plt.close()

# Split the data into training and testing sets
x1 = wine_quality.iloc[:, 0:-1]
y1 = wine_quality.iloc[:, -1]
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20, stratify=y1)

# Scale features
scaler = StandardScaler()
x1_train_scaled = x1_train.copy()
x1_train_scaled[:] = scaler.fit_transform(x1_train)
x1_test_scaled = x1_test.copy()
x1_test_scaled[:] = scaler.transform(x1_test)

# Oversample minority class using SMOTE
smote1 = SMOTE()
x1_scaled_resampled, y1_scaled_resampled = smote1.fit_resample(x1_train_scaled, y1_train)
smote2 = SMOTE()
x1_resampled, y1_resampled = smote2.fit_resample(x1_train, y1_train)

# Create and save a bar graph showing the distribution of output labels
y1_train.value_counts().sort_index().plot(kind='bar')
plt.xlabel('Wine Quality')
plt.ylabel('Frequency')
plt.title('WQ - Distribution of Output Values')
plt.savefig('Images/Dataset_Analysis/Wine_Quality_Output_Distribution.png')
plt.close()

# Create and save a correlation matrix of the features
corr_matrix = x1_train.corr()
plt.figure(figsize=(10, 10))
corrMat = plt.matshow(corr_matrix, fignum = 1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.title(f'Correlation Matrix for Wine Quality', fontsize=15)
plt.savefig('Images/Dataset_Analysis/Wine_Quality_Correlation_Matrix.png')
plt.close()
#################################################


#################################################
# Training and Testing Error Rates
#
# The following code will analyze the training 
# and testing error rates obtained by the
# various algorithms for the Wine Quality dataset
#################################################

################# Decision Tree #################
dt_generate_learning_curves(x1_resampled, y1_resampled, True, seed_value)
dt_generate_validation_curves(x1_resampled, y1_resampled, True)
 
################ Neural Network #################
nn_generate_learning_curves(x1_train_scaled, x1_test_scaled, y1_train, y1_test, True, seed_value)
nn_generate_validation_curves(x1_train_scaled, y1_train, True)
 
############# Boosted Decision Tree #############
bdt_generate_learning_curves(x1_resampled, x1_test, y1_resampled, y1_test, True, seed_value)
bdt_generate_validation_curves(x1_resampled, y1_resampled, True)
 
###################### SVM ######################
svm_generate_learning_curves(x1_scaled_resampled, y1_scaled_resampled, True, seed_value)
svm_generate_validation_curves(x1_scaled_resampled, y1_scaled_resampled, True)

###################### KNN ######################
knn_generate_learning_curves(x1_scaled_resampled, y1_scaled_resampled, True, seed_value)
knn_generate_validation_curves(x1_scaled_resampled, y1_scaled_resampled, True)

#################################################


#################################################
# Analysis of Results
#
# The following code will provide an analysis 
# of the results of each algorithm for the Wine
# Quality dataset
#################################################

################# Decision Tree #################
params = {'criterion': 'gini', 
          'max_depth': 8,
          'ccp_alpha': 0.005,
          'min_samples_leaf': 20, 
          'max_leaf_nodes': 30}
dt_analysis(x1_train, x1_test, y1_train, y1_test, params, seed_value)

################ Neural Network #################
params = {'hidden_layer_sizes': (10,),
          'activation': 'tanh',
          'solver': 'adam',
          'learning_rate': 'adaptive',
          'max_iter': 100}
nn_analysis(x1_train_scaled, x1_test_scaled, y1_train, y1_test, params, seed_value)

############# Boosted Decision Tree #############
params = {'max_depth': 8,
          'learning_rate': 0.1,
          'min_child_weight': 2,
          'n_estimators': 1000}
bdt_analysis(x1_train, x1_test, y1_train, y1_test, params, seed_value)

###################### SVM ######################
params = {'kernel': 'rbf',
          'C': 30,
          'gamma': 0.1}
svm_analysis(x1_train_scaled, x1_test_scaled, y1_train, y1_test, params, seed_value)

###################### KNN ######################
params = {'n_neighbors': 5,
          'algorithm': 'kd_tree',
          'weights': 'uniform',
          'metric': 'chebyshev'}
knn_analysis(x1_train_scaled, x1_test_scaled, y1_train, y1_test, params, seed_value)

#################################################


#################################################
# Description of Classification Problems
#
# The following code will generate a dataset
# based on the user specifications and then
# analyze the resulting dataset
#################################################
x2, y2 = make_classification(n_samples=1500,
                             n_features=11,
                             n_informative=8,
                             n_redundant=3,
                             n_classes=5,
                             flip_y=0.1,
                             class_sep=2,
                             random_state=seed_value)

# Add noise to the dataset
x2 += np.random.normal(scale=0.5, size=x2.shape)

# Introduce outliers
outliers_indices = np.random.choice(range(len(x2)), size=50, replace=False)
x2[outliers_indices] += np.random.normal(loc=10, scale=5, size=x2[outliers_indices].shape)

# Convert to a pandas DataFrame
columns = [f'feature_{i}' for i in range(x2.shape[1])]
x2 = pd.DataFrame(x2, columns=columns)
y2 = pd.DataFrame(y2, columns=['target']).squeeze()

# Observe outliers
feature_columns = x2.columns.tolist()

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 15))
axes = axes.flatten()

for i, column in enumerate(feature_columns):
    sns.boxplot(x=x2[column], ax=axes[i])
    axes[i].set(xlabel=None)    # Remove x-label
    axes[i].set_xticks([])      # Remove existing x-axis ticks
    axes[i].set_xticklabels([]) # Remove existing x-axis tick labels
    axes[i].set_yticks([])      # Remove existing y-axis ticks
    axes[i].tick_params(axis='x', rotation=90)  # Set the rotation angle
    axes[i].set_title(column)
    axes[i].title.set_size(30)

fig.delaxes(axes[11])   # Remove empty graph

plt.subplots_adjust(hspace=0.5)
plt.savefig('Images/Dataset_Analysis/Generated_Dataset_Outliers.png')
plt.close()

# Split the data into training and testing sets
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.25)

# Scale features
scaler = StandardScaler()
x2_train_scaled = x2_train.copy()
x2_train_scaled[:] = scaler.fit_transform(x2_train)
x2_test_scaled = x2_test.copy()
x2_test_scaled[:] = scaler.transform(x2_test)

# Create and save a bar graph showing the distribution of output labels
y2_train.value_counts().sort_index().plot(kind='bar')
formatter = ticker.StrMethodFormatter("{x:,.0f}")
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=0)
plt.xlabel('Output Labels')
plt.ylabel('Frequency')
plt.title('Generated Dataset - Distribution of Output Values')
plt.savefig('Images/Dataset_Analysis/Generated_Dataset_Output_Distribution.png')
plt.close()

# Create and save a correlation matrix of the features
corr_matrix = x2_train.corr()
plt.figure(figsize=(10, 10))
corrMat = plt.matshow(corr_matrix, fignum = 1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.title(f'Correlation Matrix for Generated Dataset', fontsize=15)
plt.savefig('Images/Dataset_Analysis/Generated_Dataset_Correlation_Matrix.png')
plt.close()
#################################################


#################################################
# Training and Testing Error Rates
#
# The following code will analyze the training 
# and testing error rates obtained by the
# various algorithms for the generated dataset
#################################################

################# Decision Tree #################
dt_generate_learning_curves(x2_train, y2_train, False, seed_value)
dt_generate_validation_curves(x2_train, y2_train, False)

################ Neural Network #################
nn_generate_learning_curves(x2_train_scaled, x2_test_scaled, y2_train, y2_test, False, seed_value)
nn_generate_validation_curves(x2_train_scaled, y2_train, False)

############# Boosted Decision Tree #############
bdt_generate_learning_curves(x2_train, x2_test, y2_train, y2_test, False, seed_value)
bdt_generate_validation_curves(x2_train, y2_train, False)

###################### SVM ######################
svm_generate_learning_curves(x2_train_scaled, y2_train, False, seed_value)
svm_generate_validation_curves(x2_train_scaled, y2_train, False)

###################### KNN ######################
knn_generate_learning_curves(x2_train_scaled, y2_train, False, seed_value)
knn_generate_validation_curves(x2_train_scaled, y2_train, False)

#################################################


#################################################
# Analysis of Results
#
# The following code will provide an analysis 
# of the results of each algorithm for the
# generated dataset
#################################################

################# Decision Tree #################
params = {'criterion': 'gini', 
          'max_depth': 6,
          'ccp_alpha': 0.01,
          'min_samples_leaf': 20, 
          'max_leaf_nodes': 30}
dt_analysis(x2_train, x2_test, y2_train, y2_test, params, seed_value)

################ Neural Network #################
params = {'hidden_layer_sizes': (10,),
          'activation': 'identity',
          'solver': 'adam',
          'learning_rate': 'adaptive',
          'max_iter': 100}
nn_analysis(x2_train_scaled, x2_test_scaled, y2_train, y2_test, params, seed_value)

############# Boosted Decision Tree #############
params = {'max_depth': 4,
          'learning_rate': 0.01,
          'min_child_weight':6,
          'n_estimators': 500}
bdt_analysis(x2_train, x2_test, y2_train, y2_test, params, seed_value)

###################### SVM ######################
params = {'kernel': 'rbf',
          'C': 10,
          'gamma': 0.1}
svm_analysis(x2_train_scaled, x2_test_scaled, y2_train, y2_test, params, seed_value)

###################### KNN ######################
params = {'n_neighbors': 5,
          'algorithm': 'kd_tree',
          'weights': 'uniform',
          'metric': 'chebyshev'}
knn_analysis(x2_train_scaled, x2_test_scaled, y2_train, y2_test, params, seed_value)

#################################################
