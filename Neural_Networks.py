import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def pooled_var(stds):
    return np.sqrt(sum((4)*(stds**2))/ len(stds)*(4))

def nn_generate_learning_curves(x_train, x_test, y_train, y_test, dataset, seed_value=None):
    mlp = MLPClassifier(max_iter=100)

    # Train the classifier and store accuracy at each iteration
    train_accuracies = []
    test_accuracies = []

    for i in range(1, mlp.max_iter + 1):
        mlp.partial_fit(x_train, y_train, classes=np.unique(y_train))
        y_train_pred = mlp.predict(x_train)
        y_test_pred = mlp.predict(x_test)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    plt.plot()
    plt.plot(range(1, mlp.max_iter + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, mlp.max_iter + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    if dataset:
        plt.title('Neural Networks Learning Curve - Wine Quality')
        plt.savefig('Images/NN/Wine_Quality_Learning_Curve.png')
    else:
        plt.title('Neural Networks Learning Curve - Generated Dataset')
        plt.savefig('Images/NN/Generated_Dataset_Learning_Curve.png')
    plt.close()
    return

def nn_generate_validation_curves(x_train, y_train, dataset):
    # Define the parameters to optimize
    params = {'hidden_layer_sizes': [(50,), (20,), (10,)],
              'activation': ['tanh', 'relu', 'identity', 'logistic'],
              'solver': ['sgd', 'adam', 'lbfgs'],
              'learning_rate': ['constant','adaptive', 'invscaling'],
              'max_iter': list(range(1, 52, 10))}
    
    # Conduct Grid Search over the parameter space to find the best parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=params, cv=cv, n_jobs=-1, return_train_score=True)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print('NN Best Parameters: ' + str(best_params))
    
    # Generate Validation Curves using results from Grid Search
    df = pd.DataFrame(grid_search.cv_results_)
    results = ['mean_test_score',
            'mean_train_score',
            'std_test_score', 
            'std_train_score']

    fig, axes = plt.subplots(1, len(params), figsize = (5*len(params), 7))
    axes[0].set_ylabel('Score', fontsize=25)

    for idx, (param_name, param_range) in enumerate(params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results].agg({'mean_test_score' : 'mean',
                                                                     'mean_train_score': 'mean',
                                                                     'std_test_score'  : pooled_var,
                                                                     'std_train_score' : pooled_var})
        if param_name == 'hidden_layer_sizes':
            grouped_df = grouped_df.reset_index()
            grouped_df['param_hidden_layer_sizes'] = grouped_df['param_hidden_layer_sizes'].apply(
                lambda x: 'x'.join(map(str, x)))
            grouped_df.set_index('param_hidden_layer_sizes', inplace=True)
            param_range = ['x'.join(map(str, x)) for x in param_range]

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
            if param_name != 'index':
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
        fig.suptitle('Neural Networks Validation Curves - Wine Quality', fontsize=40)
        plt.savefig('Images/NN/Wine_Quality_Validation_Curve.png')
    else:
        fig.suptitle('Neural Networks Validation Curves - Generated Dataset', fontsize=40)
        plt.savefig('Images/NN/Generated_Dataset_Validation_Curve.png')
    plt.close()
    return

def nn_analysis(x_train, x_test, y_train, y_test, params, seed_value=None):
    nn_clf = MLPClassifier()
    nn_clf.set_params(**params)
    nn_clf.fit(x_train, y_train)
    y_pred = nn_clf.predict(x_test)
    print('NN Accuracy Score: %.2f' % accuracy_score(y_test, y_pred))
    return