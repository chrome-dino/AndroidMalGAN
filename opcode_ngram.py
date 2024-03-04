import os
import pandas as pd

smali_content = []
smali_files = []
N_COUNT = 3

std_codes = {}  # empty list which will later contain all the standard op-codes read from the ops.txt file

with open('std_codes.txt', 'r') as fp:
    read_lines = fp.readlines()
    for line in read_lines:
        opline = line.split("\n")[0].split(' ')
        opcode = opline[0]
        ophex = opline[1]
        std_codes[opcode] = ophex

for root, dirs, files in os.walk('.'):  # Scanning through each file in each subdirectory
    for file in files:
        if file.endswith(".smali"):
            file_dest = os.path.join(root, file)
            with open(file_dest, 'r') as fp:
                smali_content = fp.readlines()
                smali_content = [line.rstrip('\n').split(" ") for line in smali_content]  # store the contents of a file
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
                ngram += (file_chunks[i+n])
            ngrams.append(ngram)
        except IndexError as e:
            break

print(ngrams)


# feature selection

    # ngram frequency


# feature