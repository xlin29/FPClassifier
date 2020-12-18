from scipy import sparse
import pickle
import os
import numpy as np
import copy
from run_models import *


def get_bestmodel_diff(best_model, feat_mat, labels, dict_vectorizer, url_list, fpjs2_sim, important_feat_file, diff_file):
    """
    get the best model prediction inconsistency with the ground truth

    params:
        best_mode:  model_object
        feat_mat:   csr matrix
        labels:     np.ndarray
        dict_vectorizer:    object
        url_list:   list
        fpjs2_sim:  np.ndarray
        important_feat_file:    str
        diff_file:  str
    """
    predictions = best_model.predict_proba(feat_mat)
    true_label_prob = predictions[:, 1]

    incorrect_predictions = {}
    ind = 0
    for pred, gt in zip(true_label_prob, labels):
        if (pred >= 0.5 and gt < 1) or (pred < 0.5 and gt > 0):
            incorrect_predictions[url_list[ind]] = (pred, gt, ind)
        ind += 1

    feat_importances = best_model.feature_importances_
    max_feat = np.argsort(-feat_importances)[:500]
    with open(important_feat_file, 'w') as f:
        for ind in range(len(max_feat)):
            if max_feat[ind] != len(dict_vectorizer.feature_names_):
                f.write(str(dict_vectorizer.feature_names_[max_feat[ind]]) + '\t' + str(feat_importances[max_feat[ind]]) + '\n')
            else:
                f.write('fpjs2_sim\t' + str(feat_importances[max_feat[ind]]) + '\n')

    with open(diff_file, 'w') as f:
        for url_pair, scores in incorrect_predictions.items():
            f.write(url_pair[0] + '\t' + url_pair[1] + '\t' + str(scores[0]) + '\t' + str(scores[1]) + '\n')


def main():
    feat_mat_file = '../features_v2/features_mat_round0.npz'
    feat_mat_nextround_file = '../features_v2/features_mat_round1.npz'
    labels_file = '../features_v2/label_round0.npy'
    labels_nextround_file = '../features_v2/label_round1.npy'
    url_list_file = '../features_v2/url_list_round0.txt'
    url_list_nextround_file = '../features_v2/url_list_round1.txt'
    dict_vectorizer_file = '../features_v2/dict_vectorizer.pkl'
    fpjs2_sim_file = '../features_v2/fpjs2_sim_round0.npy'
    fpjs2_sim_nextround_file = '../features_v2/fpjs2_sim_round1.npy'
    important_feat_file = '../results/dt_important_features_round1.txt'
    diff_file = '../results/dt_diff_file_round1.txt'
    
    #Data loading and preparation
    feat_mat = sparse.load_npz(feat_mat_file)
    fpjs2_sim = np.load(fpjs2_sim_file)
    labels = np.load(labels_file)
    print("labels shape: ", labels.shape)
    feat_mat_with_sim = sparse.hstack((feat_mat, fpjs2_sim.T))
    features = feat_mat_with_sim
    print("feat_mat shape: ", features.shape)
    
    #grid search on decision tree model
    best_dt, gridsearch = run_models('dt', features, labels, '../saved_models/best_dt_updated_round1_10fold_withsim.pickle', numfolds=10)
    with open('../save_models/gridsearch_results_dt_updated_round1_10fold_withsim.pickle', 'wb') as f:
        pickle.dump(gridsearch, f)
    
    #load url list
    url_list = []
    with open(url_list_file, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip('\n').split('\t')
            url_list.append((parts[0], parts[1]))
    print('url_list loaded')
    
    #load best model
    best_model_file = '../saved_models/best_dt_updated_round1_10fold_withsim.pickle'
    with open(best_model_file, 'rb') as f:
        best_model = pickle.load(f)
    print('best model loaded')
    with open(dict_vectorizer_file, 'rb') as f:
        dict_vectorizer = pickle.load(f)
    
    #find out the prediction inconsistency
    get_bestmodel_diff(best_model, features, labels, dict_vectorizer, url_list, fpjs2_sim, important_feat_file, diff_file)
