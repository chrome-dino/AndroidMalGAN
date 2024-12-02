import os
import subprocess
import random
from hybrid_feature_extract import labeled_hybrid_data
from hybrid_model import HybridGenerator
import torch
import xml.etree.ElementTree as ET

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


def inject(input_file, copy_file=False, n_count=5):
    os.system('rm -rf temp_file_dir')
    hybrid_generator = HybridGenerator()
    hybrid_generator.load_state_dict(torch.load(SAVED_MODEL_PATH)).to(DEVICE)
    hybrid_generator.eval()

    filename = os.path.basename(input_file).split('.')[0]
    print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o {filename}')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename}'
    command = command.split()
    subprocess.run(command)

    data_malware = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
    # df = pd.DataFrame(data_malware)
    # df.to_csv('temp_file_dir/malware_ngram.csv')
    # data_malware = np.loadtxt('temp_file_dir/malware_ngram.csv', delimiter=',')
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
    gen_malware = hybrid_generator(data_tensor_malware)
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
    with open(f'ngram_features_{str(n_count)}.txt', 'r') as file:
        ngram_features = file.read()
        ngram_features = ngram_features.split('\n')
    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')
    permissions = []
    # https://github.com/mrcse/Android-Permissions/blob/main/xml_permissions.txt
    with open('permissions.txt', 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            permissions.append(line.rstrip())
    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')

    for ngrams in final:
        if ngrams in ngram_features:
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

    for api in final:
        if api in api_features:
            smali_inject += function_start
            smali_inject += 'invoke-virtual {}, ' + api + '\n'
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

    for permission in final:
        if permission in permissions:
            namespaces = dict([node for _, node in ET.iterparse('AndroidManifest.xml', events=['start-ns'])])
            for namespace in namespaces.keys():
                ET.register_namespace(namespace, namespaces[namespace])

            # Parse the XML file
            tree = ET.parse('AndroidManifest.xml')
            root = tree.getroot()

            # Add a new element
            new_element = ET.Element('uses-permission')
            new_element.set('android:name', permission)
            root.append(new_element)
            # Write the modified XML back to the file
            tree.write('AndroidManifest.xml', encoding='utf-8', xml_declaration=True)

    for intent in final:
        if intent in intent_features:
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

    if copy_file:
        print(f'Compiling file: {filename} with command: apktool b modified_{input_file}')
        command = f'apktool b {input_file}'
    else:
        print(f'Compiling file: {filename} with command: apktool b {input_file}')
        command = f'apktool b {input_file}'
    command = command.split()
    subprocess.run(command)
    print(f'Finished!')
