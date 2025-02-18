from intents_inject import inject as intent_inject
from permissions_inject import inject as permission_inject
from ngram_inject import inject as ngrams_inject
from apis_inject import inject as api_inject
from hybrid_inject import inject as hybrid_inject
from hybrid_feature_extract import labeled_hybrid_data
from other_apk_feature_extract import labeled_intent_data, labeled_api_data, labeled_perm_data
from opcode_ngram_feature_extract import labeled_data
from train_blackbox import train_blackbox, Classifier2
from daisy_chained_malgan import daisy_chain_attack
from ensemble_blackbox import hybrid_ensemble_detector, ensemble_detector
import pandas as pd
import numpy as np
import subprocess
import os
import torch
import re
from sklearn.model_selection import train_test_split
from apis_model import ApisGenerator
from intents_model import IntentsGenerator
from permissions_model import PermissionsGenerator
from opcode_ngram_model import NgramGenerator
from hybrid_model import HybridGenerator
import json
from train_blackbox import train_blackbox_retrain


DOWNLOAD_TEST_SET = False
LABEL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = '../hybrid_5_'
DATADIR = os.path.join('DATA')
MALWAREDIR = 'DATA/MALWARE'
DEVICE_CPU = torch.device('cpu')

def load_blackbox(name, path, size=350):
    if name == 'ensemble':
        bb = None
    else:
        if name != 'mlp':
            bb = torch.load(path)
            bb = bb.to(DEVICE_CPU)
        else:
            load_model = torch.load(path)
            input_len = size
            bb = Classifier2(d_input_dim=input_len, l1=len(load_model['input.weight']),
                                   l2=len(load_model['fc1.weight']),
                                   l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))

            bb.load_state_dict(torch.load(path))
            bb = bb.to(DEVICE_CPU)
            bb.eval()
    return bb


def get_bb_models(model_name):
    bb_models = [{'name': 'dt', 'path': f'../dt_{model_name}_model.pth'}, {'name': 'rf', 'path': f'../rf_{model_name}_model.pth'},
             {'name': 'svm', 'path': f'../svm_{model_name}_model.pth'}, {'name': 'knn', 'path': f'../knn_{model_name}_model.pth'},
             {'name': 'gnb', 'path': f'../gnb_{model_name}_model.pth'}, {'name': 'lr', 'path': f'../lr_{model_name}_model.pth'},
             {'name': 'mlp', 'path': f'../{model_name}_mlp_model.pth'}]
    loaded_models = []
    for bb_model in bb_models:
        if 'hybrid' in bb_model["path"]:
            i_size = 460
        else:
            i_size = 350
        bb = load_blackbox(bb_model["name"], bb_model['path'], size=i_size)
        loaded_models.append({'name': bb_model["name"], 'model': bb})
    return loaded_models


def blackbox_test(test_data=None, blackbox=None, bb_name='', model_type=''):
    if bb_name == 'ensemble':
        results = ensemble_detector(model_type=model_type, test_data=test_data)
        results = np.array([[row[0]] for row in results])
    else:
        if bb_name == 'mlp':
            results = blackbox(test_data)
        else:
            results = blackbox.predict_proba(test_data)
    return results


def download_samples():
        # download 1000 malicious files
        # https://github.com/tarquinb/AZoo
        subprocess.call(['python', 'azoo.py', 'update'])
        subprocess.call(['python', 'azoo.py', 'download', '0', '1000'])

def sample_list():
    sample_md5s = []
    with open("hybrid_samples_test.txt", "w") as f:
        for (root, dirs, files) in os.walk(MALWAREDIR):
            for file in files:
                sub_dir = str(os.path.join(root, file))
                if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
                    if md5_hash[0] not in sample_md5s:
                        sample_md5s.append(md5_hash[0])
                        f.write(sub_dir + '\n')

def test_data(n_count=5):
    with open('intent_features.txt', 'r') as file:
        features = file.read()
        intent_features = features.split('\n')
    with open('perm_features.txt', 'r') as file:
        features = file.read()
        perm_features = features.split('\n')
    with open('api_features.txt', 'r') as file:
        features = file.read()
        api_features = features.split('\n')
    with open(f'ngram_features_{str(n_count)}.txt', 'r') as file:
        features = file.read()
        ngram_features = features.split('\n')
    with open("hybrid_samples_test.txt") as samples:
        hybrid_bb_models = get_bb_models('hybrid_5')
        for bb_model in hybrid_bb_models:
            print(f'Labelling {bb_model["name"]} test suite...')
            count = 0
            for s in samples:
                try:
                    s = s.rstrip()
                    s_list = list(os.path.split(s))
                    f_name = s_list[-1]
                    s_list[-1] = 'modified_' + f_name
                    s_mod = os.path.join(*s_list)
                    count += 1
                    if count > 125:
                        break
                    print(f'{f_name} {str(count)}')
                    # print('####################intent_inject####################')
                    intent_inject(s, copy_file=True, blackbox=bb_model["name"])
                    with open(f"test_suite/intent_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    intent_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    intent_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/intent_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(intent_hybrid_row)
                        df.to_csv(f'test_suite/intent_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(intent_hybrid_row)
                        df.to_csv(f'test_suite/intent_{bb_model["name"]}_hybrid_{str(n_count)}.csv')

                    intent_row = labeled_intent_data(root_dir='temp_file_dir', malware=False, single_file=True, intent_features=intent_features)
                    intent_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/malware_intent_{bb_model["name"]}_modified.csv'):
                        df = pd.DataFrame(intent_row)
                        df.to_csv(f'test_suite/malware_intent_{bb_model["name"]}_modified.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(intent_row)
                        df.to_csv(f'test_suite/malware_intent_{bb_model["name"]}_modified.csv')
                    # print('####################permission_inject####################')
                    permission_inject(s, copy_file=True, blackbox=bb_model["name"])
                    with open(f"test_suite/permission_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    permission_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    permission_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/permission_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(permission_hybrid_row)
                        df.to_csv(f'test_suite/permission_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(permission_hybrid_row)
                        df.to_csv(f'test_suite/permission_{bb_model["name"]}_hybrid_{str(n_count)}.csv')

                    permission_row = labeled_perm_data(root_dir='temp_file_dir', malware=False, single_file=True, perm_features=perm_features)
                    permission_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/malware_permission_{bb_model["name"]}_modified.csv'):
                        df = pd.DataFrame(permission_row)
                        df.to_csv(f'test_suite/malware_permission_{bb_model["name"]}_modified.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(permission_row)
                        df.to_csv(f'test_suite/malware_permission_{bb_model["name"]}_modified.csv')
                    # print('#####################api_inject####################')
                    api_inject(s, copy_file=True, blackbox=bb_model["name"])
                    with open(f"test_suite/api_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    api_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    api_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/api_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(api_hybrid_row)
                        df.to_csv(f'test_suite/api_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(api_hybrid_row)
                        df.to_csv(f'test_suite/api_{bb_model["name"]}_hybrid_{str(n_count)}.csv')

                    api_row = labeled_api_data(root_dir='temp_file_dir', malware=False, single_file=True, api_features=api_features)
                    api_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/malware_api_{bb_model["name"]}_modified.csv'):
                        df = pd.DataFrame(api_row)
                        df.to_csv(f'test_suite/malware_api_{bb_model["name"]}_modified.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(api_row)
                        df.to_csv(f'test_suite/malware_api_{bb_model["name"]}_modified.csv')
                    # print('#####################ngram_inject####################')
                    ngrams_inject(s, copy_file=True, n_count=n_count, blackbox=bb_model["name"])
                    with open(f"test_suite/ngram_{str(n_count)}_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    ngram_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    ngram_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/ngram_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(ngram_hybrid_row)
                        df.to_csv(f'test_suite/ngram_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(ngram_hybrid_row)
                        df.to_csv(f'test_suite/ngram_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv')

                    ngram_row = labeled_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True, ngram_features=ngram_features)
                    ngram_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/malware_ngram_{str(n_count)}_{bb_model["name"]}_modified.csv'):
                        df = pd.DataFrame(ngram_row)
                        df.to_csv(f'test_suite/malware_ngram_{str(n_count)}_{bb_model["name"]}_modified.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(ngram_row)
                        df.to_csv(f'test_suite/malware_ngram_{str(n_count)}_{bb_model["name"]}_modified.csv')
                    # print('#####################daisy_inject####################')
                    daisy_chain_attack(file_path=s, n_count=n_count, blackbox=bb_model["name"])
                    with open(f"test_suite/daisy_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    daisy_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    daisy_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/daisy_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(daisy_hybrid_row)
                        df.to_csv(f'test_suite/daisy_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(daisy_hybrid_row)
                        df.to_csv(f'test_suite/daisy_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv')
                    # print('#####################hybrid_inject####################')
                    hybrid_inject(s, copy_file=True, n_count=n_count, blackbox=bb_model["name"])
                    with open(f"test_suite/hybrid_ensemble_{bb_model['name']}_results.txt", "a") as f:
                        result = hybrid_ensemble_detector(bb_type=bb_model["name"], input_file=s_mod, n_count=n_count)
                        f.write(str(result) + f' {f_name}\n')

                    hybrid_hybrid_row = labeled_hybrid_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True)
                    hybrid_hybrid_row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv'):
                        df = pd.DataFrame(hybrid_hybrid_row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(hybrid_hybrid_row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv')

                    row = labeled_intent_data(root_dir='temp_file_dir', malware=False, single_file=True, intent_features=intent_features)
                    row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_intent.csv'):
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_intent.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_intent.csv')
                    row = labeled_api_data(root_dir='temp_file_dir', malware=False, single_file=True, api_features=api_features)
                    row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_api.csv'):
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_api.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_api.csv')
                    row = labeled_perm_data(root_dir='temp_file_dir', malware=False, single_file=True, perm_features=perm_features)
                    row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_permission.csv'):
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_permission.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_permission.csv')
                    row = labeled_data(root_dir='temp_file_dir', malware=False, n_count=n_count, single_file=True, ngram_features=ngram_features)
                    row[0]['file'] = f_name
                    if os.path.isfile(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_ngram_{str(n_count)}.csv'):
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_ngram_{str(n_count)}.csv', mode='a', header=False)
                    else:
                        df = pd.DataFrame(row)
                        df.to_csv(f'test_suite/hybrid_{str(n_count)}_{bb_model["name"]}_ngram_{str(n_count)}.csv')
                except Exception as e:
                    print(e)
                    count = count - 1
                    with open(f"test_suite/failed_{bb_model['name']}.txt", "a") as f:
                        f.write(f'{f_name}\n')
    return


def run_tests(n_count=5):
    if os.path.exists('test_suite_results.txt'):
        os.remove('test_suite_results.txt')

    hybrid_bb_models = get_bb_models('hybrid_5')
    for bb_model in hybrid_bb_models:
        blackbox = bb_model['model']

        final = []
        with open(f"test_suite/intent_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        intent_ensemble_results = final
        final = []
        with open(f"test_suite/api_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        api_ensemble_results = final
        final = []
        with open(f"test_suite/permission_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        permission_ensemble_results = final
        final =[]
        with open(f"test_suite/ngram_{str(n_count)}_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        ngram_ensemble_results = final
        final = []
        with open(f"test_suite/hybrid_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        hybrid_results = final
        final = []
        with open(f"test_suite/daisy_ensemble_{bb_model['name']}_results.txt") as test_results:
            for result_str in test_results:
                result_read = result_str.rsplit(' ', 1)[0].replace('[', '').replace(']', '').split(', ')
                result_row = [float(result_read[0]), float(result_read[1])]
                final.append(result_row)
        daisy_ensemble_results = final


        malware_hybrid = np.loadtxt(f'test_suite/intent_{bb_model["name"]}_hybrid_{str(n_count)}.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        intent_hybrid = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/permission_{bb_model["name"]}_hybrid_{str(n_count)}.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        permission_hybrid = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/api_{bb_model["name"]}_hybrid_{str(n_count)}.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        api_hybrid = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/ngram_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        ngram_hybrid = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/daisy_{str(n_count)}_{bb_model["name"]}_hybrid_{str(n_count)}.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        daisychain_hybrid = torch.tensor(malware_hybrid).float()

        intent_results = blackbox_test(test_data=intent_hybrid, blackbox=blackbox, bb_name=bb_model["name"], model_type='hybrid_5')
        permission_results = blackbox_test(test_data=permission_hybrid, blackbox=blackbox, bb_name=bb_model["name"], model_type='hybrid_5')
        api_results = blackbox_test(test_data=api_hybrid, blackbox=blackbox, bb_name=bb_model["name"], model_type='hybrid_5')
        ngram_results = blackbox_test(test_data=ngram_hybrid, blackbox=blackbox, bb_name=bb_model["name"], model_type='hybrid_5')
        daisychain_results = blackbox_test(test_data=daisychain_hybrid, blackbox=blackbox, bb_name=bb_model["name"], model_type='hybrid_5')

        if bb_model["name"] == 'svm':
            intent_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in intent_results]
            permission_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in permission_results]
            api_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in api_results]
            ngram_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in ngram_results]
            daisychain_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in daisychain_results]
        # results = torch.where(results > 0.5, True, False)
        mal = 0
        ben = 0
        for result in intent_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'hybrid detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} intent malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        mal = 0
        ben = 0
        for result in permission_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'hybrid detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} permission malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        mal = 0
        ben = 0
        for result in api_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'hybrid detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} api malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        mal = 0
        ben = 0
        for result in ngram_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'hybrid detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} ngram malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in daisychain_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'hybrid detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} daisy chain attack'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in hybrid_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in intent_ensemble_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} intent feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in permission_ensemble_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} permissions feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in api_ensemble_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} api feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in ngram_ensemble_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} ngram {str(n_count)} feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        mal = 0
        ben = 0
        for result in daisy_ensemble_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} daisy chain attack'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score*100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

        malware_hybrid = np.loadtxt(f'test_suite/malware_intent_{bb_model["name"]}_modified.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        labeled_intents = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/malware_permission_{bb_model["name"]}_modified.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        labeled_perms = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/malware_api_{bb_model["name"]}_modified.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        labeled_apis = torch.tensor(malware_hybrid).float()
        malware_hybrid = np.loadtxt(f'test_suite/malware_ngram_{str(n_count)}_{bb_model["name"]}_modified.csv', delimiter=',', skiprows=1)
        malware_hybrid = (malware_hybrid.astype(np.bool_)).astype(float)
        malware_hybrid = malware_hybrid[:, 1:]
        labeled_ngrams = torch.tensor(malware_hybrid).float()
        if bb_model["name"] == 'mlp':
            bb = load_blackbox(bb_model["name"], f'../intents_{bb_model["name"]}_model.pth')
        else:
            bb = load_blackbox(bb_model["name"], f'../{bb_model["name"]}_intents_model.pth')
        intent_results = blackbox_test(test_data=labeled_intents, blackbox=bb, bb_name=bb_model["name"], model_type='intents')
        if bb_model["name"] == 'svm':
            intent_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in intent_results]
        mal = 0
        ben = 0
        for result in intent_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'intents detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        if bb_model["name"] == 'mlp':
            bb = load_blackbox(bb_model["name"], f'../permissions_{bb_model["name"]}_model.pth')
        else:
            bb = load_blackbox(bb_model["name"], f'../{bb_model["name"]}_permissions_model.pth')
        permissions_results = blackbox_test(test_data=labeled_perms, blackbox=bb, bb_name=bb_model["name"], model_type='permissions')
        if bb_model["name"] == 'svm':
            permissions_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in permissions_results]
        mal = 0
        ben = 0
        for result in permissions_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'permissions detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        if bb_model["name"] == 'mlp':
            bb = load_blackbox(bb_model["name"], f'../apis_{bb_model["name"]}_model.pth')
        else:
            bb = load_blackbox(bb_model["name"], f'../{bb_model["name"]}_apis_model.pth')
        api_results = blackbox_test(test_data=labeled_apis, blackbox=bb, bb_name=bb_model["name"], model_type='apis')
        if bb_model["name"] == 'svm':
            api_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in api_results]
        mal = 0
        ben = 0
        for result in api_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'apis detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        if bb_model["name"] == 'mlp':
            bb = load_blackbox(bb_model["name"], f'../ngram_{str(n_count)}_{bb_model["name"]}_model.pth')
        else:
            bb = load_blackbox(bb_model["name"], f'../{bb_model["name"]}_ngram_{str(n_count)}_model.pth')
        ngram_results = blackbox_test(test_data=labeled_ngrams, blackbox=bb, bb_name=bb_model["name"], model_type=f'ngram_{str(n_count)}')
        if bb_model["name"] == 'svm':
            ngram_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in ngram_results]
        mal = 0
        ben = 0
        for result in ngram_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ngram {str(n_count)} detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

    model_types = ['permissions', 'intents', 'apis', f'ngram_{str(n_count)}', 'hybrid_5']
    for model_type in model_types:
        retrain(model_type=model_type)
    return


def retrain(model_type=''):
    # train black box on modified train set and save to new pth file

    data_malware = np.loadtxt(f'../malware_{model_type}.csv', delimiter=',', skiprows=1)
    data_malware = (data_malware.astype(np.bool_)).astype(float)
    data_malware = data_malware[:, 1:]
    labels_malware = data_malware[:, 0:]
    data_malware = np.array(data_malware)
    data_tensor_malware = torch.tensor(data_malware).float()
    train_data_malware = data_tensor_malware.to(DEVICE_CPU)

    bb_models = [{'name': 'mlp', 'path': f'{model_type}_retrain_mlp_model.pth'},
                {'name': 'rf', 'path': f'rf_retrain_{model_type}_model.pth'},
                 {'name': 'dt', 'path': f'dt_retrain_{model_type}_model.pth'},
                 {'name': 'svm', 'path': f'svm_retrain_{model_type}_model.pth'},
                 {'name': 'knn', 'path': f'knn_retrain_{model_type}_model.pth'},
                 {'name': 'gnb', 'path': f'gnb_retrain_{model_type}_model.pth'},
                 {'name': 'lr', 'path': f'lr_retrain_{model_type}_model.pth'},
                 ]

    # test modified test set on new black box

    # labeled_unmodified = pd.read_csv(f'../malware_{model_type}.csv')
    labeled_unmodified = np.loadtxt(f'../malware_{model_type}.csv', delimiter=',', skiprows=1)
    labeled_unmodified = (labeled_unmodified.astype(np.bool_)).astype(float)
    labeled_unmodified = labeled_unmodified[:, 1:]
    labeled_unmodified = torch.tensor(labeled_unmodified).float()
    data_benign = np.loadtxt(f'../benign_{model_type}.csv', delimiter=',', skiprows=1)
    data_benign = (data_benign.astype(np.bool_)).astype(float)
    data_benign = data_benign[:, 1:]
    labels_benign = data_benign[:, 0:]
    data_benign = np.array(data_benign)
    data_tensor_benign = torch.tensor(data_benign).float()
    train_data_benign = data_tensor_benign.to(DEVICE_CPU)

    partition = [0.6, 0.4]
    labeled_benign, data_tensor_benign, train_labels_benign, test_labels_benign = train_test_split(
        train_data_benign, labels_benign, test_size=partition[0], random_state=42)
    input_num = 350
    if model_type == 'apis':
        Generator = ApisGenerator
    elif model_type == 'intents':
        Generator = IntentsGenerator
    elif model_type == 'permissions':
        Generator = PermissionsGenerator
    elif 'ngram' in model_type:
        Generator = NgramGenerator
    else:
        Generator = HybridGenerator
        input_num = 460

    for bb in bb_models:
        with open(f'../config_{model_type}_{bb["name"]}_malgan.json') as f:
            g = json.load(f)

        generator = Generator(noise_dims=g['g_noise'], input_layers=input_num, l2=g['g_1'], l3=g['g_2'], l4=g['g_3'])
        if 'ngram' not in model_type:
            generator.load_state_dict(torch.load(f'../{model_type}_' + bb['name'] + '.pth', weights_only=True))
        else:
            generator.load_state_dict(torch.load(f'../opcode_ngram_' + bb['name'] + '_5_final.pth', weights_only=True))
        generator = generator.to(DEVICE)
        train_data_malware = train_data_malware.to(DEVICE)
        generator.eval()
        gen_malware = generator(train_data_malware)
        binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
        binarized_gen_malware_logical_or = torch.logical_or(train_data_malware, binarized_gen_malware).float()
        gen_malware = binarized_gen_malware_logical_or.to(DEVICE_CPU)
        df = pd.DataFrame(gen_malware)
        df.to_csv(f'../malware_{model_type}_modified.csv')

        train_blackbox_retrain(f'../malware_{model_type}_modified.csv', f'../benign_{model_type}.csv', f'retrain_{model_type}', bb['name'])
        labeled_modified = np.loadtxt(f'../malware_{model_type}_modified.csv', delimiter=',', skiprows=1)
        labeled_modified = (labeled_modified.astype(np.bool_)).astype(float)
        labeled_modified = labeled_modified[:, 1:]
        labeled_modified = torch.tensor(labeled_modified).float()

        if 'hybrid' in model_type:
            i_size = 460
        else:
            i_size = 350
        blackbox = load_blackbox(bb["name"], '../retrain_model.pth', size=i_size)

        labeled_modified, data_tensor_malware, train_labels_malware, test_labels_malware = train_test_split(
            labeled_modified, labels_malware, test_size=partition[0], random_state=42)

        results_modified = blackbox_test(test_data=labeled_modified, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]

        mal = 0
        ben = 0
        for result in results_modified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (modified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} retrained black box'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
    # test unmodified test set on new black box
        results_unmodified = blackbox_test(test_data=labeled_unmodified, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        mal = 0
        ben = 0
        for result in results_unmodified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (unmodified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} retrained black box'
        score = mal / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% correctly identified malicious files'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
    # test benign test set on new black box
        results_benign = blackbox_test(test_data=labeled_benign, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        mal = 0
        ben = 0
        for result in results_benign:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (benign files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% correctly identified benign files'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

     # train black box on half modified set and half unmodified set
        df1 = pd.read_csv(f'../malware_{model_type}_modified.csv', skiprows=1, nrows=4000, header=None)
        df2 = pd.read_csv(f'../malware_{model_type}.csv', skiprows=1, nrows=4000, header=None)
        df1 = df1.drop(df2.columns[[0]], axis=1)
        df2 = df2.drop(df2.columns[[0]], axis=1)
        df_merged = pd.concat([df1, df2], axis=0, ignore_index=True)
        df_merged.to_csv(f'../malware_{model_type}_merged.csv')

        train_blackbox_retrain(f'../malware_{model_type}_merged.csv', f'../benign_{model_type}.csv', model_type, bb['name'], split_data=False)
        # labeled_modified = pd.read_csv(f'../malware_{model_type}_modified.csv')

        if 'hybrid' in model_type:
            i_size = 460
        else:
            i_size = 350
        blackbox = load_blackbox(bb["name"], '../retrain_model.pth', size=i_size)
        labeled_modified = np.loadtxt(f'../malware_{model_type}_modified.csv', delimiter=',', skiprows=4001)
        labeled_modified = (labeled_modified.astype(np.bool_)).astype(float)
        labeled_modified = labeled_modified[:, 1:]
        labeled_modified = torch.tensor(labeled_modified).float()
        labeled_unmodified = np.loadtxt(f'../malware_{model_type}.csv', delimiter=',', skiprows=4001)
        labeled_unmodified = (labeled_unmodified.astype(np.bool_)).astype(float)
        labeled_unmodified = labeled_unmodified[:, 1:]
        labeled_unmodified = torch.tensor(labeled_unmodified).float()

        results_modified = blackbox_test(test_data=labeled_modified, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]
        mal = 0
        ben = 0
        for result in results_modified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (modified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% bypass'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        # test unmodified test set on new black box
        results_unmodified = blackbox_test(test_data=labeled_unmodified, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        mal = 0
        ben = 0
        for result in results_unmodified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (unmodified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        score = mal / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% correctly identified malicious files'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        # test benign test set on new black box
        results_benign = blackbox_test(test_data=labeled_benign, blackbox=blackbox, bb_name=bb["name"], model_type=model_type)
        if bb["name"] == 'svm':
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        if bb['name'] == 'knn':
            # results_modified = results_modified[:8000]
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        mal = 0
        ben = 0
        for result in results_benign:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (benign files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        score = ben / (ben + mal)
        result_str = result_str + f' | {str(score * 100)}% correctly identified benign files'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

# sample_list()
# test_data(n_count=5)
# retrain('ngram_5')
run_tests(n_count=5)