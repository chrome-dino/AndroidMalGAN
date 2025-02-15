import os
import subprocess
import xml.etree.ElementTree as ET
from other_apk_feature_extract import labeled_intent_data
from intents_model import IntentsGenerator
import json
import torch

SAVED_MODEL_PATH = '../intents_'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inject(input_file, copy_file=False, blackbox=''):
    os.system('rm -rf temp_file_dir/*')
    with open(f'../config_intents_{blackbox}_malgan.json') as f:
        g = json.load(f)
    intents_generator = IntentsGenerator(noise_dims=g['g_noise'], input_layers=350, l2=g['g_1'], l3=g['g_2'], l4=g['g_3'])
    intents_generator.load_state_dict(torch.load(SAVED_MODEL_PATH + blackbox + '.pth', weights_only=True))
    intents_generator = intents_generator.to(DEVICE)
    intents_generator.eval()

    filename = os.path.basename(input_file).split('.', -1)[0]
    print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o temp_file_dir')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename}'
    command = command.split()
    subprocess.run(command)

    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')

    data_malware = labeled_intent_data(root_dir='temp_file_dir', intent_features=intent_features, single_file=True)
    labels_malware = list(data_malware[0].keys())
    # del labels_malware[-1]
    data_malware = [data_malware[0][k] for k in labels_malware]

    data_tensor_malware = torch.tensor([data_malware]).float()
    data_tensor_malware = data_tensor_malware.to(DEVICE)

    gen_malware = intents_generator(data_tensor_malware)
    gen_malware = gen_malware[0]

    final = {}
    for i in range(len(data_tensor_malware)):
        diff = gen_malware - data_tensor_malware[i]
        final[labels_malware[i]] = diff
    print(final)
    manifest = os.path.join('temp_file_dir', filename, 'AndroidManifest.xml')
    for intent in final:
        namespaces = dict([node for _, node in ET.iterparse(manifest, events=['start-ns'])])
        for namespace in namespaces.keys():
            ET.register_namespace(namespace, namespaces[namespace])

        # Parse the XML file
        tree = ET.parse(manifest)
        root = tree.getroot()
        # for application in root.findall('application'):
        #     for activity in application.findall('activity'):
        #         for intent in activity.findall('intent-filter'):
        #             actions = intent.findall('action')
        # Add a new element
        activity = root.findall('application')[0].find('activity')
        intent_filter = ET.Element('intent-filter')

        new_element_act = ET.Element('action')
        new_element_cat = ET.Element('category')

        new_element_act.set('android:name', intent)
        new_element_cat.set('android:name', 'android.intent.category.DEFAULT')
        intent_filter.append(new_element_act)
        activity.append(intent_filter)
        # Write the modified XML back to the file
        tree.write(manifest, encoding='utf-8', xml_declaration=True)

    print(f'Compiling file: {filename} with command: apktool b temp_file_dir/{filename}')
    command = f'apktool b temp_file_dir/{filename}'
    command = command.split()
    subprocess.run(command)

    if copy_file:
        path, name = os.path.split(input_file)
        name = f'modified_{name}'
        copy_path = os.path.join(path, name)
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {copy_path}'
        command = command.split()
        subprocess.run(command)
    else:
        command = f'mv -f temp_file_dir/{filename}/dist/{filename}.apk {input_file}'
        command = command.split()
        subprocess.run(command)
    print(f'Finished!')
