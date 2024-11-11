import os
import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import gc

N_COUNT = 5
MAX_COLLECT = 5
PHASE = 1
LIMIT = False
std_codes = {}  # empty list which will later contain all the standard op-codes read from the ops.txt file


def labeled_data(root_dir='.', ngram_features=None, malware=False):
    print('obtaining labelled data')
    samples = []
    # final = []
    sample_md5s = []
    dirs = [item[0] for item in os.walk(root_dir)]
    if malware:
        if os.path.isfile('malware_ngram.csv'):
            os.remove('malware_ngram.csv')
    else:
        if os.path.isfile('benign_ngram.csv'):
            os.remove('benign_ngram.csv')
    print('getting files...')
    for sub_dir in dirs:
        if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
            if md5_hash[0] not in sample_md5s:
                if not os.listdir(sub_dir):
                    continue
                sample_md5s.append(md5_hash[0])
                samples.append(sub_dir)
                if LIMIT and len(samples) >= MAX_COLLECT:
                    break
    print('finished collecting files...')
    print('Analyzing files...')
    count = 0
    for s in samples:
        smali_files = []
        count += 1
        print(s + ' ' + str(count))
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
                                for each_code in std_codes.keys():
                                    if opcode == each_code:
                                        opcodes += (std_codes[each_code])
                                        break
                        smali_files.append(opcodes)

        ngrams = []
        for each_file in smali_files:
            file_chunks = [each_file[i:i + 2] for i in range(0, len(each_file), 2)]
            for i in range(len(file_chunks)):
                ngram = ''
                try:
                    for n in range(N_COUNT):
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
            if os.path.isfile('malware_ngram.csv'):
                df = pd.DataFrame([row])
                df.to_csv('malware_ngram.csv', mode='a', header=False)
            else:
                df = pd.DataFrame([row])
                df.to_csv('malware_ngram.csv')
        else:
            if os.path.isfile('benign_ngram.csv'):
                df = pd.DataFrame([row])
                df.to_csv('benign_ngram.csv', mode='a', header=False)
            else:
                df = pd.DataFrame([row])
                df.to_csv('benign_ngram.csv')
    return


def extract_ngram_features(root_dir='./samples', feature_count=300, exclude=None, malware=False, collect_phase=False):
    # get_opcodes
    # smali_files = []
    if malware:
        ngram_file_name = 'tmp_ngrams_malware.txt'
    else:
        ngram_file_name = 'tmp_ngrams_benign.txt'

    if collect_phase:
        if os.path.isfile(ngram_file_name):
            os.remove(ngram_file_name)
        count = 0
        current_hash = ''
        for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
            for file in files:
                if file.endswith(".smali"):
                    file_dest = os.path.join(root, file)
                    md5_hash = re.findall(r"([a-fA-F\d]{32})", file_dest)[0]

                    if md5_hash != current_hash:
                        with open(ngram_file_name, 'a') as ngram_file:
                            ngram_file.write('###' + '\n')
                        count += 1
                        current_hash = md5_hash
                        print(current_hash + ' ' + str(count))
                    if LIMIT:
                        if count >= MAX_COLLECT:
                            break
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
                                for each_code in std_codes.keys():
                                    if opcode == each_code:
                                        opcodes += (std_codes[each_code])
                                        break
                        # smali_files.append(opcodes)
                        # opcodes to ngrams
                        file_chunks = [opcodes[i:i + 2] for i in range(0, len(opcodes), 2)]
                        with open(ngram_file_name, 'a') as ngram_file:
                            for i in range(len(file_chunks)):
                                ngram = ''
                                # try:
                                if i + 5 > len(file_chunks):
                                    break
                                for n in range(N_COUNT):
                                    ngram += (file_chunks[i + n])
                                ngram_file.write(ngram + '\n')
                                    # ngrams.append(ngram)
                                # except IndexError as e:
                                #     break
            if LIMIT:
                if count >= MAX_COLLECT:
                    break
        print('finished extracting ngrams')
        return
    else:
        # opcodes to ngrams
        # ngrams = []
        # for each_file in smali_files:
        #     file_chunks = [each_file[i:i + 2] for i in range(0, len(each_file), 2)]
        #     for i in range(len(file_chunks)):
        #         ngram = ''
        #         try:
        #             for n in range(N_COUNT):
        #                 ngram += (file_chunks[i + n])
        #             ngrams.append(ngram)
        #         except IndexError as e:
        #             break

        # feature selection
        first = True
        with open(ngram_file_name, 'r') as ngram_file:
            # ngrams = ngram_file.read().splitlines()
        # del ngrams[0]

            ngram_dict = {}
            ngram_dict_file = {}
            for n in ngram_file:
                n = n.rstrip()
                if n == '###':
                    if first:
                        first = False
                        continue
                    scaler = MinMaxScaler()
                    scaled = np.array(list(ngram_dict_file.items()), dtype=object)
                    scaled[:, 1] = scaler.fit_transform(scaled[:, -1].reshape(1, scaled.shape[0]).T).T
                    ngram_dict_file = dict(scaled)
                    ngram_dict = {x: ngram_dict.get(x, 0) + ngram_dict_file.get(x, 0) for x in set(ngram_dict).union(ngram_dict_file)}
                    ngram_dict_file = {}
                    continue
                if exclude and n in exclude:
                    continue
                if n in ngram_dict_file.keys():
                    ngram_dict_file[n] += 1
                else:
                    ngram_dict_file[n] = 1

            # def keyfunc(k):
            #     return ngram_dict[k]
            filtered_ngrams = Counter(ngram_dict).most_common(feature_count)
            # filtered_ngrams = []
            # for key in sorted(ngram_dict, key=keyfunc, reverse=True)[:feature_count]:
            #     filtered_ngrams.append(key)
            # print([filtered_ngrams[n][0] for n in range(len(filtered_ngrams))])
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

########################################################################################################################
    if PHASE == 1:
        print('collecting malware ngrams')
        std_opcodes()
        extract_ngram_features(root_dir='./samples/malware_samples/decompiled', feature_count=300, malware=True,
                               collect_phase=True)
    if PHASE == 2:
        print('extracting malware ngrams')
        std_opcodes()
        malware_ngrams = extract_ngram_features(root_dir='./samples/malware_samples/decompiled', feature_count=300,
                                                malware=True, collect_phase=False)
        # gc.collect()
        if os.path.isfile('malware_features.txt'):
            os.remove('malware_features.txt')
        malware_ngrams = "\n".join(malware_ngrams)
        with open('malware_features.txt', 'w') as file:
            file.write(malware_ngrams)
        return

########################################################################################################################
    if PHASE == 3:


        print('extracting benign ngrams')
        std_opcodes()
        extract_ngram_features(root_dir='./samples/benign_samples/decompiled', feature_count=50,
                               collect_phase=True, malware=False)
        return

    if PHASE == 4:
        with open('malware_features.txt', 'r') as file:
            malware_ngrams = file.read()
            malware_ngrams = malware_ngrams.split('\n')
        std_opcodes()
        benign_ngrams = extract_ngram_features(root_dir='./samples/benign_samples/decompiled', feature_count=50,
                                               exclude=malware_ngrams, collect_phase=False, malware=False)

        # gc.collect()
        if os.path.isfile('benign_features.txt'):
            os.remove('benign_features.txt')
        benign_ngrams = "\n".join(benign_ngrams)
        with open('benign_features.txt', 'w') as file:
            file.write(benign_ngrams)
        return

########################################################################################################################
    if PHASE == 5:
        with open('benign_features.txt', 'r') as file:
            benign_ngrams = file.read()
            benign_ngrams = benign_ngrams.split('\n')

        with open('malware_features.txt', 'r') as file:
            malware_ngrams = file.read()
            malware_ngrams = malware_ngrams.split('\n')

        ngram_features = list(set(malware_ngrams + benign_ngrams))
        ngram_features = "\n".join(ngram_features)
        print('\n\n\n')
        print('ngram features')
        print('----------------------------------')
        print(ngram_features)
        with open('ngram_features.txt', 'w') as file:
            file.write(ngram_features)
        return


########################################################################################################################
    if PHASE == 6:
        with open('ngram_features.txt', 'r') as file:
            ngram_features = file.read()
            ngram_features = ngram_features.split('\n')
        std_opcodes()
        labeled_data(root_dir='./samples/malware_samples/decompiled', ngram_features=ngram_features, malware=True)
        return
        # gc.collect()
        # df = pd.DataFrame(malware_data)
        # df.to_csv('malware_ngram.csv')
        # gc.collect()

########################################################################################################################
    if PHASE == 7:
        with open('ngram_features.txt', 'r') as file:
            ngram_features = file.read()
            ngram_features = ngram_features.split('\n')
        std_opcodes()
        labeled_data(root_dir='./samples/benign_samples/decompiled', ngram_features=ngram_features, malware=False)
        return
        # gc.collect()
        # df = pd.DataFrame(benign_data)
        # df.to_csv('benign_ngram.csv')
        # gc.collect()
        # print('finished')