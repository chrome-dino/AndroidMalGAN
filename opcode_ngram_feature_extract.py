import os
import pandas as pd
from collections import Counter
import re

N_COUNT = 3
std_codes = {}  # empty list which will later contain all the standard op-codes read from the ops.txt file


def labeled_data(root_dir='.', ngram_features=None):
    samples = []
    final = []
    for root, dirs, files in os.walk('.'):
        for name in dirs:
            if re.search("^[0-9a-fA-F]{32}$", name):
                file_dest = os.path.join(root, name)
                samples.append(file_dest)

    for s in samples:
        for root, dirs, files in os.walk(s):
            for file in files:
                if file.endswith(".smali"):
                    file_dest = os.path.join(root, file)
                    smali_files = []
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
                        final.append(row)
    return final


def extract_ngram_features(root_dir='.', feature_count=300):
    # get_opcodes
    smali_files = []
    for root, dirs, files in os.walk(root_dir):  # Scanning through each file in each subdirectory
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

    # opcodes to ngrams
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

    # feature selection
    ngram_dict = {}
    for n in ngrams:
        if n in ngram_dict.keys():
            ngram_dict[n] += 1
        else:
            ngram_dict[n] = 1

    filtered_ngrams = Counter(ngram_dict).most_common(feature_count)
    return [filtered_ngrams[n][0] for n in range(len(filtered_ngrams))]


with open('std_codes.txt', 'r') as fp:
    read_lines = fp.readlines()
    for line in read_lines:
        opline = line.split("\n")[0].split(' ')
        opcode = opline[0]
        ophex = opline[1]
        std_codes[opcode] = ophex

malware_ngrams = extract_ngram_features(root_dir='malware_samples', feature_count=300)
benign_ngrams = extract_ngram_features(root_dir='benign_samples', feature_count=300)

ngram_features = list(set(malware_ngrams + benign_ngrams))

malware_data = labeled_data(root_dir='malware_samples', ngram_features=ngram_features)
df = pd.DataFrame(malware_data)
df.to_csv('malware.csv')
benign_data = labeled_data(root_dir='benign_samples', ngram_features=ngram_features)
df = pd.DataFrame(benign_data)
df.to_csv('benign.csv')
