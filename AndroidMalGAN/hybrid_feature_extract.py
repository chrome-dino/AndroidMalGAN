import os
import pandas as pd
import re
from other_apk_feature_extract import labeled_permissions_data, labeled_api_data, labeled_intent_data
from opcode_ngram_feature_extract import labeled_data as labeled_ngram_data

MAX_COLLECT = 5
LIMIT = False
FEATURE_REDUCE = False


def feature_reduction(n_count=5):
    with open(f'ngram_features_{str(n_count)}.txt', 'r') as file:
        ngram_features = file.read()
        ngram_features = ngram_features.split('\n')
    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')
    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')
    permissions = []
    with open('permissions.txt', 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            permissions.append(line.rstrip())
    reduced_features = ngram_features + intent_features + api_features + permissions
    return reduced_features


def labeled_hybrid_data(root_dir='.', malware=False, n_count=3, single_file=False, white_list=None):
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
        if FEATURE_REDUCE:
            ngram_features = [i for i in ngram_features if i not in white_list]
    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')
        if FEATURE_REDUCE:
            intent_features = [i for i in intent_features if i not in white_list]
    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')
        if FEATURE_REDUCE:
            api_features = [i for i in api_features if i not in white_list]

    with open("hybrid_samples.txt") as samples:
        for s in samples:
            s = s.rstrip()
            file_ngrams = labeled_ngram_data(root_dir=s, ngram_features=ngram_features, malware=True, n_count=n_count,
                                             single_file=True)
            file_apis = labeled_api_data(root_dir=s, api_features=api_features, malware=api_features,
                                         single_file=True)
            if FEATURE_REDUCE:
                file_permissions = labeled_permissions_data(root_dir=s, permissions_file='reduced_permissions.txt')
            else:
                file_permissions = labeled_permissions_data(root_dir=s)
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
    if FEATURE_REDUCE:
        white_list = feature_reduction()
    else:
        white_list = []
    print(f'extracting malware hybrid dsample_md5sata...')
    labeled_hybrid_data(root_dir='./samples/malware_samples/decompiled', malware=True, n_count=5, white_list=white_list)
    print(f'extracting benign hybrid data...')
    labeled_hybrid_data(root_dir='./samples/benign_samples/decompiled', malware=False, n_count=5, white_list=white_list)
    print(f'finished extracting hybrid data...')
