from scipy import sparse
import pickle
import os
import numpy as np
import copy
from run_models import *


def get_bestmodel_diff(best_model, feat_mat, labels, dict_vectorizer, url_list, fpjs2_sim, important_feat_file, diff_file):
    """
    get the best model prediction inconsistency with the ground truth and remove the incorrectly labeled ground truth

    params:
        best_mode:  model_object
        feat_mat:   csr matrix
        labels:     np.ndarray
        dict_vectorizer:    object
        url_list:   list
        fpjs2_sim:  np.ndarray
        important_feat_file:    str
        diff_file:  str

    return:
        feat_mat_nextround: csr matrix
        labels_nextround:   np.ndarray
        fpjs2_sim_nextround:    np.ndarray
        url_list_nextround: list
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

    feat_mat_nextround = copy.deepcopy(feat_mat)
    labels_nextround = copy.deepcopy(labels)
    indicators = np.ones_like(labels, dtype=bool)
    empty_urlset = set()
    ext_urlset = set()
    with open(diff_file, 'w') as f:
        for url_pair, scores in incorrect_predictions.items():
            f.write(url_pair[0] + '\t' + url_pair[1] + '\t' + str(scores[0]) + '\t' + str(scores[1]) + '\n')

            if url_pair[1] == "":
                indicators[scores[2]] = False
                empty_urlset.add(url_pair[0])
            if url_pair[0] == "https___www.fbi.gov_" and url_pair[1] == "https://gateway.foresee.com/code/19.3.3-v.3/fs.utils.js?v=esk7s2a":
                labels_nextround[scores[2]] = 1

    for ind in range(len(url_list)):
        if url_list[ind][1].startswith('extensions') or url_list[ind][1] == "":
            indicators[ind] = False
            ext_urlset.add(url_list[ind][1])

    print("num_empyt: , num_exts: ", len(empty_urlset), len(ext_urlset))
    feat_mat_nextround = feat_mat_nextround[indicators, :]
    labels_nextround = labels_nextround[indicators]
    fpjs2_sim_nextround = fpjs2_sim[:, indicators]
    url_list_nextround = []
    for ind in range(len(indicators)):
        if indicators[ind]:
            url_list_nextround.append(url_list[ind])

    print("new shape of feat_mat: ", feat_mat_nextround.shape)
    print("new shape of labels: ", labels_nextround.shape)
    print("new fpjs2 sim shape: ", fpjs2_sim_nextround.shape)
    print("new url list len: ", len(url_list_nextround))

    return feat_mat_nextround, labels_nextround, fpjs2_sim_nextround, url_list_nextround


def main():
    feat_mat_file = '../features_v2/features_mat.npz'
    feat_mat_nextround_file = '../features_v2/features_mat_round0.npz'
    labels_file = '../features_v2/label.npy'
    labels_nextround_file = '../features_v2/label_round0.npy'
    url_list_file = '../features_v2/url_list.txt'
    url_list_nextround_file = '../features_v2/url_list_round0.txt'
    dict_vectorizer_file = '../features_v2/dict_vectorizer.pkl'
    fpjs2_sim_file = '../features_v2/fpjs2_sim.npy'
    fpjs2_sim_nextround_file = '../features_v2/fpjs2_sim_round0.npy'
    important_feat_file = '../results/dt_important_features_round0.txt'
    diff_file = '../results/dt_diff_file_round0.txt'

    #Data loading and preparation
    feat_mat = sparse.load_npz(feat_mat_file)
    fpjs2_sim = np.load(fpjs2_sim_file)
    labels = np.load(labels_file)
    print("labels shape: ", labels.shape)
    feat_mat_with_sim = sparse.hstack((feat_mat, fpjs2_sim.T))
    features = feat_mat_with_sim
    print("feat_mat shape: ", features.shape)
    
    #grid search on decision tree model
    best_dt, gridsearch = run_models('dt', features, labels, '../saved_models/best_dt_updated_round0_10fold_withsim.pickle', numfolds=10)
    with open('../save_models/gridsearch_results_dt_updated_round0_10fold_withsim.pickle', 'wb') as f:
        pickle.dump(gridsearch, f)
    
    #load url list
    url_list = []
    with open(url_list_file, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip('\n').split('\t')
            url_list.append((parts[0], parts[1]))
    print('url_list loaded')
   
    #load best model
    best_model_file = '../saved_models/best_dt_updated_round0_10fold_withsim.pickle'
    with open(best_model_file, 'rb') as f:
        best_model = pickle.load(f)
    print('best model loaded')
    with open(dict_vectorizer_file, 'rb') as f:
        dict_vectorizer = pickle.load(f)
    
    #find out the prediction inconsistency
    feat_mat_nextround, labels_nextround, fpjs2_sim_nextround, url_list_nextround = get_bestmodel_diff(best_model, features, labels, dict_vectorizer, url_list, fpjs2_sim, important_feat_file, diff_file)

    #save the updated version of data for next round
    sparse.save_npz(feat_mat_nextround_file, feat_mat_nextround)
    np.save(labels_nextround_file, labels_nextround)
    np.save(fpjs2_sim_nextround_file, fpjs2_sim_nextround)
    with open(url_list_nextround_file, 'w') as f:
        for eachurl in url_list_nextround:
            f.write(eachurl[0] + '\t' + eachurl[1] + '\n')


if __name__ == "__main__":
    main()
