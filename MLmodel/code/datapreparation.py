import json
import sys
from collections import defaultdict
from scipy import sparse
import numpy as np

def load_label_data(label_data_file):
    """
    load the labeled url from file

    params:
        label_data_file:    str

    return:
        pos_site_script_list:   list
    """
    pos_site_script_list = []
    with open(label_data_file, 'r') as f:
        for eachline in f:
            one_site = json.loads(eachline.rstrip())
            site_url = list(one_site.keys())[0]
            script_list = list(one_site.values())[0]
            if len(script_list) > 0:
                for eachscript_dict in script_list:
                    script_url = list(eachscript_dict.keys())[0]
                    fp_prop_list = list(eachscript_dict.values())[0]
                    if len(fp_prop_list) > 0:
                        pos_site_script_list.append((site_url, script_url))

    return pos_site_script_list

def load_url_list(url_list_file):
    """
    load all script url list

    params:
        url_list_file: str

    return:
        url_list:   list
    """
    url_list = []
    with open(url_list_file, 'r') as f:
        for eachline in f:
            eachline = eachline.rstrip('\n')
            parts = eachline.split('\t')
            domain, script_url = parts
            url_list.append((domain, script_url))

    return url_list

def assign_labels(feat_npz, ind_url_list, pos_site_script_list):
    """
    assign the labels to correct url and use a ndarray to save the labels

    params:
        feat_npz:   csr matrix
        ind_url_list:   list
        pos_site_script_list:   list

    return:
        labels: np.ndarray
    """
    num_instances = feat_npz.shape[0]
    labels = np.zeros(num_instances)

    all_url_group = defaultdict(list)
    for i, x in enumerate(ind_url_list):
        all_url_group[x].append(i)

    pos_group = {}
    for i in pos_site_script_list:
        pos_group[i] = all_url_group[i]

    for each_domain_script, indlist in pos_group.items():
        for pos_ind in indlist:
            labels[pos_ind] = 1

    return labels


if __name__ == '__main__':
    label_data_file = '../features_v2/result_fp.json'
    feat_npz_file = '../features_v2/features_mat.npz'
    url_list_file = '../features_v2/url_list.txt'
    labels_file = '../features_v2/label'

    url_list = load_url_list(url_list_file)
    print(url_list[:5])
    feat_npz = sparse.load_npz(feat_npz_file)

    pos_site_script_list = load_label_data(label_data_file)
    print(pos_site_script_list[:5])

    labels = assign_labels(feat_npz, url_list, pos_site_script_list)
    np.save(labels_file, labels)
