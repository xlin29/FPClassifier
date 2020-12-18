import re
import os
import json
from tld import get_fld
import math
import copy
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import scipy
import pickle


class FEATURE:
    """
    parse the logs of one site and then convert the logs into a feature dictionary
    """

    def __init__(self, log_file):
        self.log = []
        self.output = []
        self.script = {}
        self.js = {}
        with open(log_file, 'r', encoding='latin1') as f:
            self.log = f.readlines()
        self.domain = log_file.replace('http___', '').replace('_', '')


    def identify_script(self):
        """
        group the log events into different scripts based on their URLs and the associated script IDs
        """
        for i, line in enumerate(self.log):
            id_pa = re.compile('(?<=^\$)\d+')
            url_pa = re.compile('(^\$\d+:")(.*?(?="))')
            child_id_pa = re.compile('(^\$\d+:)(\d+(?=:))')

            id_found = re.search(id_pa, line)
            if id_found:
                line_id = id_found.group(0)
                url_found = re.search(url_pa, line)
                child_id_found = re.search(child_id_pa, line)

                if url_found:
                    url = url_found.group(2).replace('https\\', 'https')
                    url = url.replace('http\\', 'http')
                    self.script[line_id] = url

                if child_id_found: #child id should be mapped to the parent id
                    self.script[line_id] = self.script[child_id_found.group(2)]

    def split_log(self):
        """
        split the logs with retrieved ids
        the split logs are saved in self.js
        """
        exe_id = 0
        id_pa = re.compile(r'(?<=!)\d+')
        for line in self.log:
            if line.startswith('!'):
                id_found = re.search(id_pa, line)
                if id_found:
                    exe_id = id_found.group(0)
                    if exe_id not in self.js.keys():
                        self.js[exe_id] = []
            if exe_id and not line.startswith('!') and not line.startswith('$'):
                self.js[exe_id].append(line)


    def extract_features(self, output_file=None):
        """
        feature extraction function
        """
        g_pat = re.compile(r'(?<=:){.*?}:.*')
        c_pat = re.compile(r'(?<=:).*')
        c_pat_first = re.compile(r'(.*?:.*?)(:)(.*)')
        self.identify_script()
        self.split_log()

        fp_key_list_file = "easyprivacy.txt"
        fp_key_list = []
        with open(fp_key_list_file, 'r') as f:
            for eachline in f:
                fp_key_list.append(eachline.rstrip('\n'))
        
        url_script_dict = dict()
        for k, v in self.script.items():
            if any(word in v for word in
                   ["chrome-search\://", "https://www.gstatic.com", "https://www.googletagmanager.com",
                    "https://www.google-analytics.com/", "https://www.google.com/recaptcha"]):
                continue
            if v in url_script_dict:
                url_script_dict[v].extend(self.js.get(k, []))
            else:
                url_script_dict[v] = self.js.get(k, [])

        allfeatures = []
        allfeatures_withurl = []
        for url, lines in url_script_dict.items():
            features = {}

            features['domain'] = self.domain
            features['url'] = url
            try:
                features['fld'] = get_fld(features['url'], fix_protocol=True)
            except Exception as e:
                print(e)
                pass
            if any(onefpkey in url for onefpkey in fp_key_list):# feature indicating if url contains fingerprinting keywords
                features['fpurl'] = 1

            for each in lines:
                if each.startswith('g'): #property get
                    call = re.search(g_pat, each).group(0)
                    if call in features.keys():
                        features[call] += 1
                    else:
                        features[call] = 1

                if each.startswith('c'): #function call
                    access = re.search(c_pat, each).group(0)
                    if access.count(':') > 1:
                        c_access = re.search(c_pat_first, access)
                        call = c_access.group(1)
                        argu = c_access.group(3)
                        if call in features.keys() and argu == features[call]:
                            continue
                        else:
                            features[call] = argu
            compare = copy.deepcopy(features)
            del compare['domain']
            del compare['url']
            if "fld" in compare:
                del compare['fld']

            if output_file:
                output_file.write(json.dumps(features) + '\n')
            allfeatures.append(compare)
            allfeatures_withurl.append(features)
        return allfeatures, allfeatures_withurl


class DataSet:
    """
    store all feature dictionary and transform the list of feature dictionary into a feature matrix
    most features are categorical features, therefore, most are sparse features with one-hot indexing transformation
    """

    def __init__(self, features):
        self.raw_features = features
        self.dict_vectorizer = None

    def to_sparsemat(self):
        """
        tranform the list of feature dictionary to a sparse feature matrix
        """
        self.dict_vectorizer = DictVectorizer()
        sparse_feat = self.dict_vectorizer.fit_transform(self.raw_features)
        sparse_feat.data = np.nan_to_num(sparse_feat.data, copy=False)
        sparse_feat.eliminate_zeros()
        return sparse_feat


if __name__ == "__main__":
    if not os.path.isdir("../MLmodel/features_v2"):
        os.mkdir("../MLmodel/features_v2")
    output_file = open("../MLmodel/features_v2/features_new.json", 'a', encoding="utf-8")
    all_features = []
    domain_script_url_list = []
    for sub_dir in os.listdir('../logs'):
        result = {}
        if not sub_dir.startswith('.'):
            for file in os.listdir('../logs/' + sub_dir):
                try:
                    if not file.startswith('.'):
                        label = FEATURE('../logs/' + sub_dir + '/' + file)

                        features, features_withurl = label.extract_features(output_file)
                        all_features.extend(features)
                        url_in_one_domain_list = [(sub_dir, feat_dict['url']) for feat_dict in features_withurl]
                        domain_script_url_list.extend(url_in_one_domain_list)
                except Exception as e:
                    print(file)
                    print('error--', e)
                    continue
    dataset = DataSet(all_features)
    data_sp_mat = dataset.to_sparsemat()
    print(data_sp_mat.shape)

    scipy.sparse.save_npz('../MLmodel/features_v2/features_mat.npz', data_sp_mat)

    with open('../MLmodel/features_v2/dict_vectorizer.pkl', 'wb') as f:
        pickle.dump(dataset.dict_vectorizer, f)

    with open('../MLmodel/features_v2/url_list.txt', 'w') as f:
        for eachurl in domain_script_url_list:
            f.write(eachurl[0] + '\t' + eachurl[1] + '\n')
