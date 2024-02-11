import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def pooled_var(stds):
    return np.sqrt(sum((4)*(stds**2))/ len(stds)*(4))

def knn_generate_learning_curves(x_train, y_train, dataset, seed_value=None):
    train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(), X=x_train, y=y_train, cv=10, n_jobs=-1, train_sizes=np.arange(0.1, 1.1, 0.1).tolist())              
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot()
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="navy", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="darkorange", label="Cross-validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='navy', alpha=0.3)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='darkorange', alpha=0.3)
    plt.legend(loc="best")
    if dataset:
        plt.title('KNN Learning Curve - Wine Quality')
        plt.savefig('Images/KNN/Wine_Quality_Learning_Curve.png')
    else:
        plt.title('KNN Learning Curve - Generated Dataset')
        plt.savefig('Images/KNN/Generated_Dataset_Learning_Curve.png')
    plt.close()
    return

def knn_generate_validation_curves(x_train, y_train, dataset):
    # Define the parameters to optimize
    params = {'n_neighbors': list(range(1, 22, 1)),
              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
              'weights': ['uniform', 'distance'],
              'metric': ['minkowski', 'chebyshev']}

    # Conduct Grid Search over the parameter space to find the best parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1, return_train_score=True)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print('KNN Best Parameters: ' + str(best_params))

    # Generate Validation Curves using results from Grid Search
    df = pd.DataFrame(grid_search.cv_results_)
    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score', 
               'std_train_score']

    fig, axes = plt.subplots(1, len(params), figsize = (5*len(params), 7))
    axes[0].set_ylabel('Score', fontsize=25)

    for idx, (param_name, param_range) in enumerate(params.items()):
        
        grouped_df = df.groupby(f'param_{param_name}')[results].agg({'mean_test_score': 'mean',
                                                                    'mean_train_score': 'mean',
                                                                    'std_test_score': pooled_var,
                                                                    'std_train_score': pooled_var})
        axes[idx].set_xlabel(param_name, fontsize=30)
        if any(isinstance(element, str) for element in param_range):
            index = np.arange(len(param_range))

            axes[idx].bar(index, grouped_df['mean_train_score'], width=0.3)
            axes[idx].errorbar(index, 
                            grouped_df['mean_train_score'],
                            yerr=grouped_df['std_train_score'],
                            fmt='o',
                            color='r')
            axes[idx].bar(index+0.3, grouped_df['mean_test_score'], width=0.3)
            axes[idx].errorbar(index+0.3, 
                            grouped_df['mean_test_score'],
                            yerr=grouped_df['std_test_score'],
                            fmt='o',
                            color='r')
            axes[idx].set_xticks(index + 0.3 / 2)
            axes[idx].set_xticklabels(param_range)
        else:
            axes[idx].plot(param_range, 
                        grouped_df['mean_train_score'],
                        label='Training score',
                        lw=2)
            axes[idx].fill_between(param_range,
                        grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                        grouped_df['mean_train_score'] + grouped_df['std_train_score'],
                        alpha=0.2,
                        color='navy',
                        lw=2)
            axes[idx].plot(param_range,
                        grouped_df['mean_test_score'],
                        label='Cross-validation score',
                        lw=2)
            axes[idx].fill_between(param_range,
                            grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                            grouped_df['mean_test_score'] + grouped_df['std_test_score'],
                            alpha=0.2,
                            color='darkorange',
                            lw=2)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    if dataset:
        fig.suptitle('KNN Validation Curves - Wine Quality', fontsize=40)
        plt.savefig('Images/KNN/Wine_Quality_Validation_Curve.png')
    else:
        fig.suptitle('KNN Validation Curves - Generated Dataset', fontsize=40)
        plt.savefig('Images/KNN/Generated_Dataset_Validation_Curve.png')
    plt.close()
    return

def knn_analysis(x_train, x_test, y_train, y_test, params, seed_value=None):
    knn_clf = KNeighborsClassifier()
    knn_clf.set_params(**params)
    knn_clf.fit(x_train, y_train)
    y_pred = knn_clf.predict(x_test)
    print('KNN Accuracy Score: %.2f' % accuracy_score(y_test, y_pred))
    return
