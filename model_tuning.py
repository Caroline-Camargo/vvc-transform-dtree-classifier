import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from logger import log_message
from config import RANDOM_STATE

def grid_search(X_train, y_train):
    log_message("Starting randomized hyperparameter search...")

    param_dist = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': np.arange(2, 100, 10),
        'min_samples_leaf': np.arange(2, 200, 10),
        'max_leaf_nodes': np.arange(10, 500, 25),
        'max_depth': np.arange(1, 30, 1),
    }

    dt = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        random_state=RANDOM_STATE,
        scoring='accuracy'
    )

    random_search.fit(X_train, y_train)
    log_message(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_params_
