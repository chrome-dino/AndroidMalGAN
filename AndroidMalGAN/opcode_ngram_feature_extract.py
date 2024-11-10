import os
import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import other_apk_feature_extract
import logging
import gc

MAX_COLLECT = 5
LIMIT = False
std_codes = {}  # empty list which will later contain all the standard op-codes read from the ops.txt file
NGRAM_STORE = './ngram_store'
NGRAM_STORE_LIMIT = 50


def merge_ngram_dict(ngram_dict, ngram_file_name, exclude):
    ngram_dict_file = {}
    with open(ngram_file_name, 'r') as ngram_file:
        # ngrams = ngram_file.read().splitlines()
        # del ngrams[0]

        for n in ngram_file:
            n = n.rstrip()
            if exclude and n in exclude:
                continue
            if n in ngram_dict_file.keys():
                ngram_dict_file[n] += 1
            else:
                ngram_dict_file[n] = 1
    scaler = MinMaxScaler()
    scaled = np.array(list(ngram_dict_file.items()), dtype=object)
    scaled[:, 1] = scaler.fit_transform(scaled[:, -1].reshape(1, scaled.shape[0]).T).T
    ngram_dict_file = dict(scaled)
    ngram_dict_file = {key: val for key, val in ngram_dict_file.items() if val != 0}
    # x = pd.concat([ngram_dict_file, x], axis=0, ignore_index=True)

    # ngram_dict = pd.read_csv(csv_filename, index_col=0, header=None, squeeze=True).to_dict()
    ngram_dict = {x: ngram_dict.get(x, 0) + ngram_dict_file.get(x, 0) for x in
                  set(ngram_dict).union(ngram_dict_file)}

    # ngram_dict = pd.DataFrame([ngram_dict])
    # ngram_dict.to_csv(csv_filename)
    return ngram_dict


def labeled_data(root_dir='.', ngram_features=None, malware=False, n_count=3):
    # print('obtaining labelled data')
    # final = []
    sample_md5s = []
    # dirs = [item[0] for item in os.walk(root_dir)]
    if malware:
        if os.path.isfile('malware.csv'):
            os.remove('malware.csv')
    else:
        if os.path.isfile('benign.csv'):
            os.remove('benign.csv')
    # print('getting files...')
    count = 0
    with open("samples.txt", "w") as f:
        for item in os.walk(root_dir):
            sub_dir = item[0]
        # for sub_dir in dirs:
            if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
                if md5_hash[0] not in sample_md5s:
                    if not os.listdir(sub_dir):
                        continue
                    sample_md5s.append(md5_hash[0])
                    f.write(sub_dir + '\n')
                    count += 1
                    if LIMIT and count >= MAX_COLLECT:
                        break
    # print('finished collecting files...')
    # print('Analyzing files...')
    count = 0
    with open("samples.txt") as samples:
        for s in samples:
            s = s.rstrip()
            smali_files = []
            count += 1
            # print(s + ' ' + str(count))
            if count % 100 == 0:
                print(f'finished labelling {str(count)} files')
            for root, dirs, files in os.walk(s):
                for file in files:
                    if file.endswith(".smali"):
                        file_dest = os.path.join(root, file)
                        with open(file_dest, 'r') as fp:
                            smali_content = fp.readlines()
                            smali_content = [line.rstrip('\n').split(" ") for line in
                                             smali_content]  # store the contents of a file
                            opcodes = ''
                            for line in smali_content:
                                for opcode in line:
                                    if opcode == '':
                                        continue
                                    if opcode in std_codes.keys():
                                        opcodes += (std_codes[opcode])
                            smali_files.append(opcodes)

            ngrams = []
            for each_file in smali_files:
                file_chunks = [each_file[i:i + 2] for i in range(0, len(each_file), 2)]
                for i in range(len(file_chunks)):
                    ngram = ''
                    try:
                        for n in range(n_count):
                            ngram += (file_chunks[i + n])
                        ngrams.append(ngram)
                    except IndexError as e:
                        break
            row = dict.fromkeys(ngram_features, 0)
            for n in ngrams:
                if n in row.keys():
                    row[n] += 1
            # row['malware'] = malware
            # row['md5'] = file_dest.split('\\')[-2]
            if malware:
                if os.path.isfile(f'malware_ngram_{str(n_count)}.csv'):
                    df = pd.DataFrame([row])
                    df.to_csv(f'malware_ngram_{str(n_count)}.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame([row])
                    df.to_csv(f'malware_ngram_{str(n_count)}.csv')
            else:
                if os.path.isfile(f'benign_ngram_{str(n_count)}.csv'):
                    df = pd.DataFrame([row])
                    df.to_csv(f'benign_ngram_{str(n_count)}.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame([row])
                    df.to_csv(f'benign_ngram_{str(n_count)}.csv')
    return


def extract_ngram_features(root_dir='./samples', feature_count=300, exclude=None, n_count=3):
    # get_opcodes
    # smali_files = []
    ngram_file_name = 'tmp_ngrams.txt'

    if os.path.isfile(ngram_file_name):
        os.remove(ngram_file_name)

    filelist = [f for f in os.listdir(NGRAM_STORE) if f.endswith(".json")]
    for f in filelist:
        os.remove(os.path.join(NGRAM_STORE, f))
    count = 0
    ngram_store_count = 0
    current_hash = ''
    ngram_dict = {}
    first = True
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
        for file in files:
            if file.endswith(".smali"):
                file_dest = os.path.join(root, file)
                md5_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]
                if first:
                    current_hash = md5_hash
                    first = False
                if md5_hash != current_hash:
                    if os.stat(ngram_file_name).st_size == 0:
                        current_hash = md5_hash
                        continue
                    ngram_dict = merge_ngram_dict(ngram_dict, ngram_file_name, exclude)

                    count += 1
                    current_hash = md5_hash
                    open(ngram_file_name, 'w').close()
                    # print(current_hash + ' ' + str(count))
                    if count % 100 == 0:
                        print(f'finished extracting {str(count)} files')
                    if count % NGRAM_STORE_LIMIT == 0:
                        with open(f'./{NGRAM_STORE}/ngram_store_{str(ngram_store_count)}.json', 'w') as f:
                            json.dump(ngram_dict, f)
                        ngram_dict = {}
                        ngram_store_count += 1

                ####################################################################################################
                #
                # if LIMIT:
                #     if count >= MAX_COLLECT:
                #         break
                # print('extracting from file: ' + file_dest)
                with open(file_dest, 'r') as fp:
                    smali_content = fp.readlines()
                    smali_content = [line.rstrip('\n').split(" ") for line in
                                     smali_content]  # store the contents of a file
                    opcodes = ''
                    for line in smali_content:
                        for opcode in line:
                            if opcode == '':
                                continue
                            if opcode in std_codes.keys():
                                opcodes += (std_codes[opcode])
                    # smali_files.append(opcodes)
                    # opcodes to ngrams
                    file_chunks = [opcodes[i:i + 2] for i in range(0, len(opcodes), 2)]
                    with open(ngram_file_name, 'a') as ngram_file:
                        for i in range(len(file_chunks)):
                            ngram = ''
                            if i + n_count > len(file_chunks):
                                break
                            for n in range(n_count):
                                ngram += (file_chunks[i + n])
                            ngram_file.write(ngram + '\n')

            if LIMIT:
                if count >= MAX_COLLECT:
                    break

    # def keyfunc(k):
    #     return ngram_dict[k]
    if ngram_dict:
        ngram_dict = merge_ngram_dict(ngram_dict, ngram_file_name, exclude)
        with open(f'./{NGRAM_STORE}/ngram_store_{str(ngram_store_count)}.json', 'w') as f:
            json.dump(ngram_dict, f)
    ngram_dict = {}
    for f in os.listdir(NGRAM_STORE):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(NGRAM_STORE, f)) as json_file:
            data = json.load(json_file)
            ngram_dict = {x: ngram_dict.get(x, 0) + data.get(x, 0) for x in set(ngram_dict).union(data)}
    filtered_ngrams = Counter(ngram_dict).most_common(feature_count)
    # print('finished extracting ngrams')
    return [filtered_ngrams[n][0] for n in range(len(filtered_ngrams))]


def std_opcodes():
    with open('AndroidMalGAN/std_codes.txt', 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            opline = line.split("\n")[0].split(' ')
            opcode = opline[0]
            ophex = opline[1]
            std_codes[opcode] = ophex


def extract():
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler('ngram_extract.log')
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)
    std_opcodes()
########################################################################################################################
    for n in range(3, 11):
        print(f'extracting malware {str(n)} gram features...')
        # extract_ngram_features(root_dir='./samples/malware_samples/decompiled', feature_count=300, malware=True)
        # print('extracting malware ngrams')
        malware_ngrams = extract_ngram_features(root_dir='./samples/malware_samples/decompiled', feature_count=300,
                                                n_count=n)
        # gc.collect()
        if os.path.isfile(f'malware_features_{str(n)}.txt'):
            os.remove(f'malware_features_{str(n)}.txt')
        malware_ngrams = "\n".join(malware_ngrams)
        with open(f'malware_features_{str(n)}.txt', 'w') as file:
            file.write(malware_ngrams)

    ########################################################################################################################

        print(f'extracting benign {str(n)} gram features...')
        # extract_ngram_features(root_dir='./samples/benign_samples/decompiled', feature_count=50, malware=False)

        with open(f'malware_features_{str(n)}.txt', 'r') as file:
            malware_ngrams = file.read()
            malware_ngrams = malware_ngrams.split('\n')
        benign_ngrams = extract_ngram_features(root_dir='./samples/benign_samples/decompiled', feature_count=50,
                                               exclude=malware_ngrams, n_count=n)

        # gc.collect()
        if os.path.isfile(f'benign_features_{str(n)}.txt'):
            os.remove(f'benign_features_{str(n)}.txt')
        benign_ngrams = "\n".join(benign_ngrams)
        with open(f'benign_features_{str(n)}.txt', 'w') as file:
            file.write(benign_ngrams)

    ########################################################################################################################

        with open(f'benign_features_{str(n)}.txt', 'r') as file:
            benign_ngrams = file.read()
            benign_ngrams = benign_ngrams.split('\n')

        with open(f'malware_features_{str(n)}.txt', 'r') as file:
            malware_ngrams = file.read()
            malware_ngrams = malware_ngrams.split('\n')

        ngram_features = list(set(malware_ngrams + benign_ngrams))
        ngram_features = "\n".join(ngram_features)
        print('\n\n\n')
        print('ngram features')
        print('----------------------------------')
        print(ngram_features)
        with open(f'ngram_features_{str(n)}.txt', 'w') as file:
            file.write(ngram_features)

        print(f'finished extracting {str(n)} gram features...')

    ########################################################################################################################

        print(f'extracting malware {str(n)} gram data...')
        with open(f'ngram_features_{str(n)}.txt', 'r') as file:
            ngram_features = file.read()
            ngram_features = ngram_features.split('\n')
        labeled_data(root_dir='./samples/malware_samples/decompiled', ngram_features=ngram_features, malware=True,
                     n_count=n)
        # gc.collect()
        # df = pd.DataFrame(malware_data)
        # df.to_csv('malware.csv')
        # gc.collect()

    ########################################################################################################################
        print(f'extracting benign {str(n)} gram data...')
        labeled_data(root_dir='./samples/benign_samples/decompiled', ngram_features=ngram_features, malware=False,
                     n_count=n)
        # gc.collect()
        # df = pd.DataFrame(benign_data)
        # df.to_csv('benign.csv')
        # gc.collect()
        # print('finished')
        print(f'finished extracting {str(n)} gram data...')

        other_apk_feature_extract.get_features()
