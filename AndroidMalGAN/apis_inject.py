import os
import subprocess
import random
from other_apk_feature_extract import labeled_api_data
from apis_model import ApisGenerator
import torch
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

SAVED_MODEL_PATH = '../apis_'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inject(input_file, copy_file=False, blackbox=''):
    os.system('rm -rf temp_file_dir/*')
    with open(f'../config_apis_{blackbox}_malgan.json') as f:
        g = json.load(f)
    api_generator = ApisGenerator(noise_dims=g['g_noise'], input_layers=350, l2=g['g_1'], l3=g['g_2'],
                                                 l4=g['g_3'])
    api_generator.load_state_dict(torch.load(SAVED_MODEL_PATH + blackbox + '.pth', weights_only=True))
    api_generator = api_generator.to(DEVICE)
    api_generator.eval()

    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')

    filename = os.path.basename(input_file).split('.', -1)[0]
    # print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o temp_file_dir')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename} -q -b'
    command = command.split()
    process = subprocess.Popen(command)
    process.wait()
    data_malware = labeled_api_data(root_dir='temp_file_dir', api_features=api_features,
                                    single_file=True)

    manifest = os.path.join('temp_file_dir', filename, 'AndroidManifest.xml')
    tree = ET.parse(manifest)
    root = tree.getroot()
    for application in root.findall('application'):
        attributes = application.attrib
        remove = []
        for attribute in attributes:
            if 'qihoo' in attribute:
                remove.append(attribute)
        for attribute in remove:
            del application.attrib[attribute]
    tree.write(manifest, encoding='utf-8', xml_declaration=True)

    # df = pd.DataFrame(data_malware)
    # df.to_csv('temp_file_dir/malware_ngram.csv')
    # data_malware = np.loadtxt('temp_file_dir/malware_ngram.csv', delimiter=',')
    # print(data_malware)
    # labels_malware = data_malware[:, 0]
    # data_malware = data_malware[:, 1:]
    labels_malware = list(data_malware[0].keys())
    # del labels_malware[-1]
    data_malware = [data_malware[0][k] for k in labels_malware]
    # dataNorm_malware = data_malware / np.max(data_malware)
    # dataNorm_malware = 2 * dataNorm_malware - 1
    # convert to tensor
    data_tensor_malware = torch.tensor([data_malware]).float()
    data_tensor_malware = data_tensor_malware.to(DEVICE)

    # noise = torch.as_tensor(np.random.uniform(0, 1, (1, ngram_generator.noise_dims)))
    # malware_noise = torch.cat((data_tensor_malware, noise), 1)
    # data_tensor_malware = data_tensor_malware.to(DEVICE)
    # gen_malware = ngram_generator(data_tensor_malware.to(DEVICE)).cpu()
    gen_malware = api_generator(data_tensor_malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(data_tensor_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or
    gen_malware = gen_malware[0]

    final = {}
    for i in range(len(data_tensor_malware[0])):
        diff = gen_malware[i] - data_tensor_malware[0][i]
        final[labels_malware[i]] = diff.item()

    smali_inject = ''
    function_start = '''.method private throw2()V
        .locals 3
        .prologue
        .line 31
    '''
    function_end = '.end method\n'
    for api in final:
        if final[api] < 1.0:
            continue
        smali_inject += function_start
        smali_inject += 'invoke-virtual {}, ' + api + '(Landroid/app/Activity;Landroid/content/Intent;ILandroid/os/Bundle;)V\n'
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
    # print(f'injecting into file: {inject_file}')
    with open(inject_file, 'a') as file:
        file.write(smali_inject)

    # print(f'Compiling file: {filename} with command: apktool b temp_file_dir/{filename}')
    command = f'apktool b temp_file_dir/{filename} -q -b'
    command = command.split()
    process = subprocess.Popen(command)
    process.wait()
    if copy_file:
        path, name = os.path.split(input_file)
        name = f'modified_{name}'
        copy_path = os.path.join(path, name)
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {copy_path}'
        command = command.split()
        process = subprocess.Popen(command)
        process.wait()
    else:
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {input_file}'
        command = command.split()
        process = subprocess.Popen(command)
        process.wait()
    # print(f'Finished!')
