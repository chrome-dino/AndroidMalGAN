import os
import subprocess
import xml.etree.ElementTree as ET
from other_apk_feature_extract import labeled_perm_data
from permissions_model import PermissionsGenerator
import torch
import json

SAVED_MODEL_PATH = '../permissions_'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inject(input_file, copy_file=False, blackbox=''):
    os.system('rm -rf temp_file_dir/*')
    with open(f'../config_permissions_{blackbox}_malgan.json') as f:
        g = json.load(f)
    permissions_generator = PermissionsGenerator(noise_dims=g['g_noise'], input_layers=350, l2=g['g_1'], l3=g['g_2'], l4=g['g_3'])
    permissions_generator.load_state_dict(torch.load(SAVED_MODEL_PATH + blackbox + '.pth', weights_only=True))
    permissions_generator = permissions_generator.to(DEVICE)
    permissions_generator.eval()

    filename = os.path.basename(input_file).split('.', -1)[0]
    # print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o temp_file_dir')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename} -q -b'
    command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if err:
        print(err)
        raise

    with open('perm_features.txt', 'r') as file:
        perm_features = file.read()
        perm_features = perm_features.split('\n')

    data_malware = labeled_perm_data(root_dir='temp_file_dir', perm_features=perm_features, single_file=True)
    labels_malware = list(data_malware[0].keys())
    # del labels_malware[-1]
    data_malware = [data_malware[0][k] for k in labels_malware]

    data_tensor_malware = torch.tensor([data_malware]).float()
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    gen_malware = permissions_generator(data_tensor_malware)

    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(data_tensor_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or

    gen_malware = gen_malware[0]

    final = {}
    for i in range(len(data_tensor_malware[0])):
        diff = gen_malware[i] - data_tensor_malware[0][i]
        final[labels_malware[i]] = diff.item()
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
    for permission in final:
        if final[permission] < 1.0:
            continue
        namespaces = dict([node for _, node in ET.iterparse(manifest, events=['start-ns'])])
        for namespace in namespaces.keys():
            ET.register_namespace(namespace, namespaces[namespace])

        # Parse the XML file
        tree = ET.parse(manifest)
        root = tree.getroot()

        # Add a new element
        new_element = ET.Element('uses-permission')
        new_element.set('android:name', permission)
        root.append(new_element)
        # Write the modified XML back to the file
        tree.write(manifest, encoding='utf-8', xml_declaration=True)

    # print(f'Compiling file: {filename} with command: apktool b temp_file_dir/{filename}')
    command = f'apktool b temp_file_dir/{filename} -q -b'
    command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if err:
        print(err)
        raise

    if copy_file:
        path, name = os.path.split(input_file)
        name = f'modified_{name}'
        copy_path = os.path.join(path, name)
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {copy_path}'
        command = command.split()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            print(err)
            raise
    else:
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {input_file}'
        command = command.split()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            print(err)
            raise
    # print(f'Finished!')
