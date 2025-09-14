import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from logger import log_message
from config import RANDOM_STATE

def recursive_feature_elimination_cv(X_train, y_train):
    start_time = time.time()
    log_message("Running RFECV to find optimal number of features...")

    clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
    rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
    rfecv.fit(X_train, y_train)

    log_message(f"RFECV completed. Optimal number of features: {rfecv.n_features_}")
    log_message(f"Selected features (bool): {rfecv.support_}")
    log_message(f"Total time: {time.time() - start_time:.2f} seconds")
    return rfecv.support_
