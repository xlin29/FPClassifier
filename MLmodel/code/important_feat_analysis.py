from scipy import sparse
import pickle
import os
import numpy as np

important_feat_list = ['{CanvasRenderingContext2D}:"fillText"', '{BatteryManager}:"chargingTime"', '%open:{XMLHttpRequest}="GET":"https\://c.adsco.re/":#T', '{CanvasRenderingContext2D}:"strokeText"', '{HTMLCanvasElement}:"toDataURL"', '{RTCPeerConnection}:"createOffer"', '%open:{XMLHttpRequest}="GET":"http\://c.adsco.re/":#T', '%setAttribute:{HTMLIFrameElement}="src":"javascript\:no"', '{Performance}:"timing"', '{AudioParam}:"setValueAtTime"']

dict_vectorizer_file = '../features_v2/dict_vectorizer.pkl'
feat_mat_nextround_file = '../features_v2/features_mat_round0.npz'
fpjs2_sim_nextround_file = '../features_v2/fpjs2_sim_round0.npy'
labels_nextround_file = '../features_v2/label_round0.npy'


#data loading and preparation
with open(dict_vectorizer_file, 'rb') as f:
    dict_vectorizer = pickle.load(f)
feat_mat = sparse.load_npz(feat_mat_nextround_file)
fpjs2_sim = np.load(fpjs2_sim_nextround_file)
feat_mat_with_sim = sparse.hstack((feat_mat, fpjs2_sim.T)).tocsr()

#separate positive instances and negative instances
labels = np.load(labels_nextround_file)
pos_indicators = labels > 0
neg_indicators = labels < 1
feat_mat_pos = feat_mat_with_sim[pos_indicators, :]
feat_mat_neg = feat_mat_with_sim[neg_indicators, :]


#count the ratio of positive and negative for each significant feature
count = {}
for eachfeat in important_feat_list:
    feat_ind = dict_vectorizer.feature_names_.index(eachfeat)
    print(feat_ind)

    pos_count = np.sum(feat_mat_pos[:, feat_ind] > 0)
    neg_count = np.sum(feat_mat_neg[:, feat_ind] > 0)
    count[eachfeat] = [pos_count, neg_count]

print(count)
