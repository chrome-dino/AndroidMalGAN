import os
import subprocess
import random
from opcode_ngram_feature_extract import labeled_data
from opcode_ngram_model import NgramGenerator
import torch
import numpy as np
import pandas as pd

SAVED_MODEL_PATH = './opcode_ngram_malgan.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gen_code(opcode):
    idx = int(opcode, 16)
    with open('AndroidMalGAN/inject_code.txt') as f:
        inject_list = f.read()
    inject_list = inject_list.split('###')
    del inject_list[0]
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


def inject(input_file):
    os.system('rm -rf temp_file_dir')
    ngram_generator = NgramGenerator()
    ngram_generator.load_state_dict(torch.load(SAVED_MODEL_PATH))
    ngram_generator.eval()

    with open('ngram_features.txt', 'r') as file:
        ngram_features = file.read()
        ngram_features = ngram_features.split('\n')

    filename = os.path.basename(input_file).split('.')[0]
    print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o {filename}')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename}'
    command = command.split()
    subprocess.run(command)

    data_malware = labeled_data(root_dir='temp_file_dir', ngram_features=ngram_features)
    # df = pd.DataFrame(data_malware)
    # df.to_csv('temp_file_dir/malware.csv')
    # data_malware = np.loadtxt('temp_file_dir/malware.csv', delimiter=',')
    # print(data_malware)
    # labels_malware = data_malware[:, 0]
    # data_malware = data_malware[:, 1:]
    labels_malware = list(data_malware[0].keys())
    del labels_malware[-1]
    data_malware = [data_malware[0][k] for k in labels_malware]
    # dataNorm_malware = data_malware / np.max(data_malware)
    # dataNorm_malware = 2 * dataNorm_malware - 1
    # convert to tensor
    data_tensor_malware = torch.tensor(data_malware).float()

    # noise = torch.as_tensor(np.random.uniform(0, 1, (1, ngram_generator.noise_dims)))
    # malware_noise = torch.cat((data_tensor_malware, noise), 1)
    # data_tensor_malware = data_tensor_malware.to(DEVICE)
    # gen_malware = ngram_generator(data_tensor_malware.to(DEVICE)).cpu()
    gen_malware = ngram_generator(data_tensor_malware)
    gen_malware = gen_malware[0]

    final = {}
    for i in range(len(data_tensor_malware)):
        diff = gen_malware - data_tensor_malware[i]
        final[labels_malware[i]] = diff
    print(final)
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

    smali_dir = f'temp_file_dir/{filename}/smali'
    # smali_files = [f for f in os.listdir(smali_dir) if os.path.isfile(os.path.join(smali_dir, f) and f.endswith('.smali'))]
    smali_files = []
    for root, subdir, files in os.walk(smali_dir):
        for name in files:
            smali_file = os.path.join(root, name)
            if os.path.isfile(smali_file) and name.endswith('.smali'):
                smali_files.append(smali_file)

    inject_file = random.choice(smali_files)
    print(f'injecting into file: {inject_file}')
    with open(inject_file, 'a') as file:
        file.write(smali_inject)

    print(f'Compiling file: {filename} with command: apktool b {input_file}')
    command = f'apktool b {input_file}'
    command = command.split()
    subprocess.run(command)
    print(f'Finished!')
