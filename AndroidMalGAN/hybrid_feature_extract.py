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
    if not single_file:
        if malware:
            if os.path.isfile(f'malware_hybrid.csv'):
                os.remove(f'malware_hybrid.csv')
        else:
            if os.path.isfile(f'benign_hybrid.csv'):
                os.remove(f'benign_hybrid.csv')

    count = 0
    sample_md5s = []
    with open("hybrid_samples.txt", "w") as f:
        for (root, dirs, files) in os.walk(root_dir):
            for file in files:
                sub_dir = root
                if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
                    if md5_hash[0] not in sample_md5s:
                        sample_md5s.append(md5_hash[0])
                        f.write(sub_dir + '\n')
                        count += 1
                        if LIMIT and count >= MAX_COLLECT:
                            break


    count = 0
    with open(f'malware_features_{str(n_count)}.txt', 'r') as file:
        mal_ngrams = file.read()
        mal_ngrams = mal_ngrams.split('\n')
        mal_ngrams = mal_ngrams[:100]
    with open(f'benign_features_{str(n_count)}.txt', 'r') as file:
        ben_ngrams = file.read()
        ben_ngrams = ben_ngrams.split('\n')
        ben_ngrams = ben_ngrams[:15]
    ngram_features = list(set(mal_ngrams + ben_ngrams))
    with open('malware_intents.txt', 'r') as file:
        mal_intents = file.read()
        mal_intents = mal_intents.split('\n')
        mal_intents = mal_intents[:100]
    with open('benign_intents.txt', 'r') as file:
        ben_intents = file.read()
        ben_intents = ben_intents.split('\n')
        ben_intents = ben_intents[:15]
    intent_features = list(set(mal_intents + ben_intents))
    with open('malware_apis.txt', 'r') as file:
        mal_apis = file.read()
        mal_apis = mal_apis.split('\n')
        mal_apis = mal_apis[:100]
    with open('benign_apis.txt', 'r') as file:
        ben_apis = file.read()
        ben_apis = ben_apis.split('\n')
        ben_apis = ben_apis[:15]
    api_features = list(set(mal_apis + ben_apis))
    with open('malware_perms.txt', 'r') as file:
        mal_perms = file.read()
        mal_perms = mal_perms.split('\n')
        mal_perms = mal_perms[:100]
    with open('benign_perms.txt', 'r') as file:
        ben_perms = file.read()
        ben_perms = ben_perms.split('\n')
        ben_perms = ben_perms[:15]
    perm_features = list(set(mal_perms + ben_perms))
    with open("hybrid_samples.txt") as samples:
        for s in samples:
            s = s.rstrip()
            file_ngrams = labeled_ngram_data(root_dir=s, ngram_features=ngram_features, malware=malware, n_count=n_count,
                                             single_file=True)[0]
            file_apis = labeled_api_data(root_dir=s, api_features=api_features, malware=malware,
                                         single_file=True)[0]

            file_permissions = labeled_perm_data(root_dir=s, perm_features=perm_features, malware=malware, single_file=True)[0]
            file_intents = labeled_intent_data(root_dir=s, intent_features=intent_features, malware=malware,
                                               single_file=True)[0]
            if file_ngrams is None or file_apis is None or file_permissions is None or file_intents is None:
                continue
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
    labeled_hybrid_data(root_dir='../samples/malware_samples/decompiled', malware=True, n_count=5)
    print(f'extracting benign hybrid data...')
    labeled_hybrid_data(root_dir='../samples/benign_samples/decompiled', malware=False, n_count=5)
    print(f'finished extracting hybrid data...')
    # print('running feature reduction...')
    # feature_reduction()
    print('Done!')