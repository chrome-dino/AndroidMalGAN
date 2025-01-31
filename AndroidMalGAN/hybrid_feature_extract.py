import os
import pandas as pd
import re
from other_apk_feature_extract import labeled_perm_data, labeled_api_data, labeled_intent_data
from opcode_ngram_feature_extract import labeled_data as labeled_ngram_data
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

MAX_COLLECT = 5
LIMIT = False


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def feature_reduction(n_count=5):
    ord_enc = OrdinalEncoder()
    dfs = []

    # Read each CSV file and append to list
    for file in [f"malware_hybrid_{str(n_count)}.csv", f"benign_hybrid_{str(n_count)}.csv"]:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all dataframes
    train_df = pd.concat(dfs, ignore_index=True)
    # data_benign = np.loadtxt(BENIGN_CSV, delimiter=',', skiprows=1)
    train_df = (train_df.astype(np.bool_)).astype(float)

    train = train_df.fillna(0)
    train1 = train[:, 1:]
    labels = train[:, 0]
    train1[[column for column in train1.columns]] = ord_enc.fit_transform(train1[[column for column in train1.columns]])
    var_thr = VarianceThreshold(threshold=0.1)
    var_thr.fit(train1)
    concol = [column for column in train1.columns
              if column not in train1.columns[var_thr.get_support()]]

    train1.drop(concol, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(train1, labels, test_size=0.33, random_state=1)
    ord_enc.fit(X_train)
    X_train_enc = ord_enc.transform(X_train)
    X_test_enc = ord_enc.transform(X_test)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()

    for file in [f"malware_hybrid_{str(n_count)}.csv", f"malware_hybrid_{str(n_count)}.csv"]:
        df = pd.read_csv(file)
        df[[column for column in df.columns]] = ord_enc.fit_transform(
            df[[column for column in df.columns]])
        concol = [column for column in df.columns
                  if column not in df.columns[X_test_fs.get_support()]]
        df.drop(concol, axis=1)
        df.to_csv(file.split('.')[0] + '_feature_reduced.csv', index=False)
    return


def labeled_hybrid_data(root_dir='.', malware=False, n_count=3, single_file=False):
    sample_md5s = []
    if not single_file:
        if malware:
            if os.path.isfile(f'malware_hybrid.csv'):
                os.remove(f'malware_hybrid.csv')
        else:
            if os.path.isfile(f'benign_hybrid.csv'):
                os.remove(f'benign_hybrid.csv')

    count = 0
    with open("hybrid_samples.txt", "w") as f:
        for item in os.walk(root_dir):
            sub_dir = item[0]
            if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
                if md5_hash[0] not in sample_md5s:
                    if not os.listdir(sub_dir):
                        continue
                    sample_md5s.append(md5_hash[0])
                    f.write(sub_dir + '\n')
                    count += 1
                    if LIMIT and count >= MAX_COLLECT:
                        break

    count = 0
    with open(f'ngram_features_{str(n_count)}.txt', 'r') as file:
        ngram_features = file.read()
        ngram_features = ngram_features.split('\n')
        ngram_features = ngram_features[:100]
    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')
        intent_features = intent_features[:100]
    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')
        api_features = api_features[:100]
    with open('perm_features.txt', 'r') as file:
        perm_features = file.read()
        perm_features = perm_features.split('\n')
        perm_features = perm_features[:100]
    with open("hybrid_samples.txt") as samples:
        for s in samples:
            s = s.rstrip()
            file_ngrams = labeled_ngram_data(root_dir=s, ngram_features=ngram_features, malware=True, n_count=n_count,
                                             single_file=True)
            file_apis = labeled_api_data(root_dir=s, api_features=api_features, malware=api_features,
                                         single_file=True)

            file_permissions = labeled_perm_data(root_dir=s, perm_features=perm_features, single_file=True)
            file_intents = labeled_intent_data(root_dir=s, intent_features=intent_features,
                                               single_file=True)

            row = file_intents | file_permissions | file_apis | file_ngrams
            if not single_file:
                if malware:
                    if os.path.isfile(f'malware_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame([row])
                        df.to_csv(f'malware_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame([row])
                        df.to_csv(f'malware_hybrid_{str(n_count)}.csv')
                else:
                    if os.path.isfile(f'benign_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame([row])
                        df.to_csv(f'benign_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame([row])
                        df.to_csv(f'benign_hybrid_{str(n_count)}.csv')
            else:
                return [row]

            count += 1
            if count % 100 == 0:
                print(f'finished labelling {str(count)} files')

    return


def extract():
    print(f'extracting malware hybrid data...')
    labeled_hybrid_data(root_dir='./samples/malware_samples/decompiled', malware=True, n_count=5)
    print(f'extracting benign hybrid data...')
    labeled_hybrid_data(root_dir='./samples/benign_samples/decompiled', malware=False, n_count=5)
    print(f'finished extracting hybrid data...')
    # print('running feature reduction...')
    # feature_reduction()
    print('Done!')