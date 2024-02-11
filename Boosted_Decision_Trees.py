import numpy as np
import pandas as pd
import xgboost as xgb 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def pooled_var(stds):
    return np.sqrt(sum((4)*(stds**2))/ len(stds)*(4))

def bdt_generate_learning_curves(x_train, x_test, y_train, y_test, dataset, seed_value=None):
    # Define and fit model to training data
    bdt_clf = xgb.XGBClassifier(n_estimators=100, eval_metric=['merror', 'mlogloss'])
    le_train = LabelEncoder()
    le_test = LabelEncoder()
    y_train_encoded = le_train.fit_transform(y_train)
    y_test_encoded = le_test.fit_transform(y_test)
    eval_set = [(x_train, y_train_encoded), (x_test, y_test_encoded)]
    bdt_clf.fit(x_train, y_train_encoded, eval_set=eval_set, verbose=False)
    
    # Make predictions for test data and evaluate accuracy
    y_pred = bdt_clf.predict(x_test)
    y_pred_decoded = le_train.inverse_transform(y_pred)
    print('BDT Accuracy Score: %.2f' % accuracy_score(y_test, y_pred_decoded))
    
    # Retrieve and graph performance metrics
    results = bdt_clf.evals_result()
    x_axis = range(0, len(results['validation_0']['merror']))

    plt.plot()
    plt.plot(x_axis, results['validation_0']['merror'], label='Train')
    plt.plot(x_axis, results['validation_1']['merror'], label='Test')
    plt.legend()
    plt.ylabel('Classification Error')
    plt.xlabel('Number of Iterations')
    plt.title('XGBoost Classification Error')
    if dataset:
        plt.savefig('Images/BDT/Wine_Quality_Learning_Curve_Error.png')
    else:
        plt.savefig('Images/BDT/Generated_Dataset_Learning_Curve_Error.png')
    plt.close()
    return

def bdt_generate_validation_curves(x_train, y_train, dataset):
    # Define the parameters to optimize
    params = {'max_depth': list(range(3, 11)),
              'learning_rate': [0.001, 0.01, 0.1],
              'min_child_weight': list(range(1, 11, 2)),
              'n_estimators': list(range(100, 1001, 100))}
    
    # Conduct Grid Search over the parameter space to find the best parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=params, cv=cv, n_jobs=-1, scoring='accuracy', return_train_score=True)
    le_train = LabelEncoder()
    y_train_encoded = le_train.fit_transform(y_train)
    grid_search.fit(x_train, y_train_encoded, verbose=False)
    best_params = grid_search.best_params_
    print('BDT Best Parameters: ' + str(best_params))

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
        fig.suptitle('Boosted Decision Tree Validation Curves - Wine Quality', fontsize=40)
        plt.savefig('Images/BDT/Wine_Quality_Validation_Curve.png')
    else:
        fig.suptitle('Boosted Decision Tree Validation Curves - Generated Dataset', fontsize=40)
        plt.savefig('Images/BDT/Generated_Dataset_Validation_Curve.png')
    plt.close()

    return

def bdt_analysis(x_train, x_test, y_train, y_test, params, seed_value=None):
    bdt_clf = xgb.XGBClassifier(**params)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    bdt_clf.fit(x_train, y_train)
    y_pred = bdt_clf.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    print('BDT Accuracy Score: %.2f' % accuracy_score(y_test, y_pred))
    return
