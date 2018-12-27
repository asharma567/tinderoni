
'''
when addressing the imbalance of dataset the predicted probability of a example can be perturbed but the rank-ordering of all the examples remain... How can I set up the voting committee to capture the most ppopular rank ordering.

- try NOT correcting for imbalance in the dataset using the voting committee
'''

full_param_grid =  {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'n_estimators':[10,500,800,1000],
            # positives/negs
            'scale_pos_weight' : [1]
        }


def random_grid_searching_model(model,  X, Y, folds=5, params=full_param_grid, param_comb=5):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        n_iter=param_comb, 
        scoring='roc_auc', 
        n_jobs=-1, 
        cv=skf.split(X,Y), 
        verbose=3, 
        random_state=1001 
    )

    # Here we go
    start_time = _timer(None)
    random_search.fit(X, Y)
    print (_timer(start_time)) 

    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best score:')
    print(random_search.best_score_ )
    print('\n Best hyperparameters:')
    print(random_search.best_params_)

    return random_search


def _timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



def full_gridsearch(model,  X, Y,  param_grid, folds=5):
    
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", n_jobs=-1, cv=skf.split(X,Y))
    grid_result = grid_search.fit(X, Y)
    
    
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result


if __name__ == '__main__':
    
    # Tune learning_rate
    from numpy import loadtxt
    from xgboost import XGBClassifier
    
    # load data
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
    
    # split data into X and y
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # grid search
    # model = XGBClassifier()
    model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        silent=True, nthread=1)

    positives = sum(Y)
    negs = Y.shape[0]

    print(random_grid_searching_model(model, X, Y,params=full_param_grid))
    
