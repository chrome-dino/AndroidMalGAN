import os
import subprocess
import xml.etree.ElementTree as ET
from other_apk_feature_extract import labeled_intent_data
from permissions_model import PermissionsGenerator
import torch

SAVED_MODEL_PATH = './intents_malgan.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inject(input_file):
    os.system('rm -rf temp_file_dir')
    permissions_generator = PermissionsGenerator()
    permissions_generator.load_state_dict(torch.load(SAVED_MODEL_PATH))
    permissions_generator.eval()

    filename = os.path.basename(input_file).split('.')[0]
    print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o {filename}')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename}'
    command = command.split()
    subprocess.run(command)

    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')

    data_malware = labeled_intent_data(root_dir='temp_file_dir', intent_features=intent_features, single_file=True)
    labels_malware = list(data_malware[0].keys())
    del labels_malware[-1]
    data_malware = [data_malware[0][k] for k in labels_malware]

    data_tensor_malware = torch.tensor(data_malware).float()

    gen_malware = permissions_generator(data_tensor_malware)
    gen_malware = gen_malware[0]

    final = {}
    for i in range(len(data_tensor_malware)):
        diff = gen_malware - data_tensor_malware[i]
        final[labels_malware[i]] = diff
    print(final)

    for intent in final:
        namespaces = dict([node for _, node in ET.iterparse('AndroidManifest.xml', events=['start-ns'])])
        for namespace in namespaces.keys():
            ET.register_namespace(namespace, namespaces[namespace])

        # Parse the XML file
        tree = ET.parse('AndroidManifest.xml')
        root = tree.getroot()
        # for application in root.findall('application'):
        #     for activity in application.findall('activity'):
        #         for intent in activity.findall('intent-filter'):
        #             actions = intent.findall('action')
        # Add a new element
        activity = root.findall('application')[0].find('activity')
        intent_filter = ET.Element('intent-filter')
        if 'action' in intent:
            new_element = ET.Element('action')
        elif 'category' in intent:
            new_element = ET.Element('category')
        else:
            continue
        new_element.set('android:name', intent)
        intent_filter.append(new_element)
        activity.append(intent_filter)
        # Write the modified XML back to the file
        tree.write('AndroidManifest.xml', encoding='utf-8', xml_declaration=True)

    print(f'Compiling file: {filename} with command: apktool b {input_file}')
    command = f'apktool b {input_file}'
    command = command.split()
    subprocess.run(command)
    print(f'Finished!')
