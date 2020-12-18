from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import pickle
import os
import numpy as np
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


def dim_reduction(feat_mat, remain_dims=500):
    """
    PCA dimension reduction for high dimension sparse feature matrix

    params:
        feat_mat:   csr matrix
        remain_dim: int

    return:
        reduced_feat_mat: np.ndarray
    """
    svd = TruncatedSVD(n_components=remain_dims, n_iter=7, random_state=42)
    svd.fit(feat_mat)

    reduced_feat_mat = svd.transform(feat_mat)
    print("old shape is: ", feat_mat.shape)
    print("new shape is: ", reduced_feat_mat.shape)
    return reduced_feat_mat


def mutual_info_feat_select(X, y, num_feats=500):
    """
    mututal information feature selection for high dimension sparse feature matrix

    params:
        X:  csr matrix
        y:  np.ndarray

    return:
        X_new:  np.ndarray
    """
    selection = SelectKBest(mutual_info_classif, k=num_feats)
    selection.fit(X, y)
    X_new = selection.transform(X)
    print("new shape of mututal infor selection: ", X_new.shape)

    return X_new, selection


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def run_models(model_type, feat_mat, labels, best_model_file, numfolds=5):
    """
    cross validation for a chosen model and report the several evaluation metrics

    params:
        model_type: str
        feat_mat:   csr matrix
        labels:     np.ndarray
        best_model_file:    str
        numfolds:   int

    return:
        clf.best_estimator_:    model_object
        clf:    gridsearch object
    """
    skf = StratifiedKFold(n_splits=numfolds)
    scoring_dict = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
            'fp': make_scorer(fp), 'fn': make_scorer(fn), 'prec': 'precision', 'rec': 'recall', 'f1_s': 'f1', 'auc_s': 'roc_auc', 'acc': 'accuracy'}

    model = None
    search_params = {}
    if model_type == "lr":
        search_params = {"penalty": ['l1', 'l2'], "C": [1.0, 10.0]}
        model = LogisticRegression(random_state=0, solver='liblinear')
    elif model_type == "svc":
        search_params = {"C": [0.1, 1.0, 10.0, 50.0], "kernel": ['linear'], "degree": [1,2,3]}
        model = SVC(gamma='auto')
    elif model_type == "dt":
        search_params = {"criterion": ['gini', 'entropy']}
        model = DecisionTreeClassifier(random_state=0)
    elif model_type == "gbdt":
        search_params = {"learning_rate": [0.1], "n_estimators": [50, 100, 200], "max_depth": [3, 4]}
        model = GradientBoostingClassifier(random_state=0)

    clf = GridSearchCV(estimator=model, param_grid=search_params, cv=skf, refit='f1_s', n_jobs=15, verbose=1, scoring=scoring_dict)
    clf.fit(feat_mat, labels)
    print(clf.cv_results_)
    print('the best model: ', clf.best_estimator_)
    print('the best model f1: ', clf.best_score_)
    print('the best model param: ', clf.best_params_)
    print('best model training time: ', clf.refit_time_)

    with open(best_model_file, 'wb') as b_model_f:
        pickle.dump(clf.best_estimator_, b_model_f)

    return clf.best_estimator_, clf
