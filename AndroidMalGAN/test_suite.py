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
from ensemble_blackbox import hybrid_ensemble_detector
import pandas as pd
import subprocess
import os
import torch
import re


DOWNLOAD_TEST_SET = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/hybrid_'
DATADIR = os.path.join('DATA')
MALWAREDIR = DATADIR + '/MALWARE'


def load_blackbox(name, path):
    if name != 'mlp':
        bb = torch.load(path)
    else:
        # {'name': 'mlp', 'path': 'mlp_ngram_model.pth'}
        load_model = torch.load(path)
        bb = Classifier2(d_input_dim=350, l1=len(load_model['input.weight']),
                         l2=len(load_model['fc1.weight']),
                         l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))
        bb.load_state_dict(load_model)
    return bb


def get_bb_models(model_name):
    bb_models = [{'name': 'rf', 'path': f'rf_{model_name}_model.pth'}, {'name': 'dt', 'path': f'dt_{model_name}_model.pth'},
             {'name': 'svm', 'path': f'svm_{model_name}_model.pth'}, {'name': 'knn', 'path': f'knn_{model_name}_model.pth'},
             {'name': 'gnb', 'path': f'gnb_{model_name}_model.pth'}, {'name': 'lr', 'path': f'lr_{model_name}_model.pth'},
             {'name': 'mlp', 'path': f'{model_name}_mlp.pth'}]
    loaded_models = []
    for bb_model in bb_models:
        bb = load_blackbox(bb_model['name'], bb_model['path'])
        loaded_models.append({'name': bb_model['name'], 'model': bb})
    return loaded_models


def blackbox_test(test_data=None, blackbox=None, bb_name=''):
    if bb_name == 'mlp':
        results = blackbox(test_data)
    else:
        results = blackbox.predict_proba(test_data)
    return results


def run_tests(n_count=5):
    if os.path.exists('test_suite_results.txt'):
        os.remove('test_suite_results.txt')
    if DOWNLOAD_TEST_SET:
        # download 1000 malicious files
        # https://github.com/tarquinb/AZoo
        subprocess.call(['python', 'azoo.py', 'update'])
        subprocess.call(['python', 'azoo.py', 'download', '0', '1000'])

    sample_md5s = []
    with open("hybrid_samples.txt", "w") as f:
        for item in os.walk(MALWAREDIR):
            sub_dir = item[0]
            if md5_hash := re.findall(r"([a-fA-F\d]{32})", sub_dir):
                if md5_hash[0] not in sample_md5s:
                    if not os.listdir(sub_dir):
                        continue
                    sample_md5s.append(md5_hash[0])
                    f.write(sub_dir + '\n')

    intent_hybrid = []
    permission_hybrid = []
    api_hybrid = []
    ngram_hybrid = []
    daisychain_hybrid = []
    hybrid_results = []
    intent_ensemble_results = []
    permission_ensemble_results = []
    api_ensemble_results = []
    ngram_ensemble_results = []
    daisy_ensemble_results = []
    with open("hybrid_samples.txt") as samples:
        hybrid_bb_models = get_bb_models('hybrid')
        for bb_model in hybrid_bb_models:
            blackbox = bb_model['model']
            for s in samples:
                s = s.rstrip()
                s_list = list(os.path.split(s))
                f_name = s_list[-1]
                s_list[-1] = 'modified_' + f_name
                s_mod = os.path.join(*s_list)

                intent_inject(s, copy_file=True, blackbox=bb_model['name'])
                intent_ensemble_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))
                intent_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])
                intent_row = labeled_intent_data(root_dir=s_mod, malware=False, single_file=True)
                if os.path.isfile(f'malware_intent_modified.csv'):
                    df = pd.DataFrame(intent_row)
                    df.to_csv(f'malware_intent_modified.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame(intent_row)
                    df.to_csv(f'malware_intent_modified.csv')

                permission_inject(s, copy_file=True, blackbox=bb_model['name'])
                permission_ensemble_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))
                permission_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])
                permission_row = labeled_perm_data(root_dir=s_mod, malware=False, single_file=True)
                if os.path.isfile(f'malware_permission_modified.csv'):
                    df = pd.DataFrame(permission_row)
                    df.to_csv(f'malware_permission_modified.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame(permission_row)
                    df.to_csv(f'malware_permission_modified.csv')

                api_inject(s, copy_file=True, blackbox=bb_model['name'])
                api_ensemble_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))
                api_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])
                api_row = labeled_api_data(root_dir=s_mod, malware=False, single_file=True)
                if os.path.isfile(f'malware_api_modified.csv'):
                    df = pd.DataFrame(api_row)
                    df.to_csv(f'malware_api_modified.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame(api_row)
                    df.to_csv(f'malware_api_modified.csv')

                ngrams_inject(s, copy_file=True, n_count=n_count, blackbox=bb_model['name'])
                ngram_ensemble_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))
                ngram_row = labeled_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)
                ngram_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])
                if os.path.isfile(f'malware_ngram_{str(n_count)}_modified.csv'):
                    df = pd.DataFrame(ngram_row)
                    df.to_csv(f'malware_ngram_{str(n_count)}_modified.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame([ngram_row])
                    df.to_csv(f'malware_ngram_{str(n_count)}_modified.csv')
                daisy_chain_attack(file_path=s, n_count=n_count, blackbox=bb_model['name'])
                daisy_ensemble_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))
                daisychain_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

                hybrid_inject(s, copy_file=True, n_count=n_count, blackbox=bb_model['name'])
                hybrid_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod, n_count=n_count))

            intent_results = blackbox_test(test_data=intent_hybrid, blackbox=blackbox, bb_name=bb_model['name'])
            permission_results = blackbox_test(test_data=permission_hybrid, blackbox=blackbox, bb_name=bb_model['name'])
            api_results = blackbox_test(test_data=api_hybrid, blackbox=blackbox, bb_name=bb_model['name'])
            ngram_results = blackbox_test(test_data=ngram_hybrid, blackbox=blackbox, bb_name=bb_model['name'])
            daisychain_results = blackbox_test(test_data=daisychain_hybrid, blackbox=blackbox, bb_name=bb_model['name'])

            if bb_model['name'] == 'svm':
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
            print(result_str)
            with open('test_suite_results.txt', 'a') as f:
                f.write(result_str + '\n')

            labeled_intents = []
            labeled_perms = []
            labeled_apis = []
            labeled_ngrams = []
            for s in samples:
                s = s.rstrip()
                s_list = list(os.path.split(s))
                f_name = s_list[-1]
                s_list[-1] = 'modified_' + f_name
                s_mod = os.path.join(*s_list)

                hybrid_inject(s, copy_file=True, n_count=n_count, blackbox=bb_model['name'])
                hybrid_row = labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)
                if os.path.isfile(f'malware_hybrid_modified.csv'):
                    df = pd.DataFrame(hybrid_row)
                    df.to_csv(f'malware_hybrid_modified.csv', mode='a', header=False)
                else:
                    df = pd.DataFrame(hybrid_row)
                    df.to_csv(f'malware_hybrid_modified.csv')
                labeled_intents.append(labeled_intent_data(root_dir=s_mod, malware=False, single_file=True)[0])
                labeled_apis.append(labeled_api_data(root_dir=s_mod, malware=False, single_file=True)[0])
                labeled_perms.append(labeled_perm_data(root_dir=s_mod, malware=False, single_file=True)[0])
                labeled_ngrams.append(labeled_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

            bb = load_blackbox(bb_model['name'], f'{bb_model["name"]}_intents_model.pth')
            intent_results = blackbox_test(test_data=labeled_intents, blackbox=bb, bb_name=bb_model['name'])
            if bb['name'] == 'svm':
                intent_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in intent_results]
            mal = 0
            ben = 0
            for result in intent_results:
                if result[0] < 0.5:
                    ben += 1
                else:
                    mal += 1
            result_str = f'intents detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
            print(result_str)
            with open('test_suite_results.txt', 'a') as f:
                f.write(result_str + '\n')

            bb = load_blackbox(bb_model['name'], f'{bb_model["name"]}_permissions_model.pth')
            permissions_results = blackbox_test(test_data=labeled_perms, blackbox=bb, bb_name=bb_model["name"])
            if bb['name'] == 'svm':
                permissions_results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in permissions_results]
            mal = 0
            ben = 0
            for result in permissions_results:
                if result[0] < 0.5:
                    ben += 1
                else:
                    mal += 1
            result_str = f'permissions detector {bb_model["name"]} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_model["name"]} hybrid feature set malgan'
            print(result_str)
            with open('test_suite_results.txt', 'a') as f:
                f.write(result_str + '\n')

            bb = load_blackbox(bb_model['name'], f'{bb_model["name"]}_apis_model.pth')
            api_results = blackbox_test(test_data=labeled_apis, blackbox=bb, bb_name=bb_model["name"])
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
            print(result_str)
            with open('test_suite_results.txt', 'a') as f:
                f.write(result_str + '\n')

            bb = load_blackbox(bb_model['name'], f'{bb_model["name"]}_ngram_{str(n_count)}_model.pth')
            ngram_results = blackbox_test(test_data=labeled_ngrams, blackbox=bb, bb_name=bb_model["name"])
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
            print(result_str)
            with open('test_suite_results.txt', 'a') as f:
                f.write(result_str + '\n')

    model_types = ['permissions', 'intents', 'apis', f'ngram_{str(n_count)}', 'hybrid']
    for model_type in model_types:
        retrain(model_type=model_type)
    return

def retrain(model_type=''):
    # train black box on modified train set and save to new pth file
    train_blackbox(f'../malware_{model_type}_modified.csv', f'../benign_{model_type}.csv', model_type, split_data=True)
    bb_models = [{'name': 'rf', 'path': f'rf_{model_type}_model_retrain.pth'},
                 {'name': 'dt', 'path': f'dt_{model_type}_model_retrain.pth'},
                 {'name': 'svm', 'path': f'svm_{model_type}_model_retrain.pth'},
                 {'name': 'knn', 'path': f'knn_{model_type}_model_retrain.pth'},
                 {'name': 'gnb', 'path': f'gnb_{model_type}_model_retrain.pth'},
                 {'name': 'lr', 'path': f'lr_{model_type}_model_modified.pth'},
                 {'name': 'mlp', 'path': f'{model_type}_mlp_modified.pth'}]

    # test modified test set on new black box
    labeled_modified = pd.read_csv(f'../malware_{model_type}_modified.csv', nrows=600)
    labeled_unmodified = pd.read_csv(f'../malware_{model_type}.csv', nrows=600)
    labeled_benign = pd.read_csv(f'../benign_{model_type}.csv', nrows=600)
    for bb in bb_models:
        blackbox = load_blackbox(bb['name'], bb['path'])
        results_modified = blackbox_test(test_data=labeled_modified, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]
        mal = 0
        ben = 0
        for result in results_modified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (modified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
    # test unmodified test set on new black box
        results_unmodified = blackbox_test(test_data=labeled_unmodified, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        mal = 0
        ben = 0
        for result in results_unmodified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (unmodified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
    # test benign test set on new black box
        results_benign = blackbox_test(test_data=labeled_benign, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        mal = 0
        ben = 0
        for result in results_benign:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (benign files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

    # train black box on half modified set and half unmodified set
    df1 = pd.read_csv(f'../malware_{model_type}_modified.csv', nrows=500)
    df2 = pd.read_csv(f'../malware_{model_type}.csv', nrows=500)
    df_merged = df1.append(df2, ignore_index=True)
    df_merged.to_csv('../malware_{model_type}_modified.csv')
    train_blackbox(f'../malware_{model_type}_modified.csv', f'../benign_{model_type}.csv', model_type,
                   split_data=True)
    bb_models = [{'name': 'rf', 'path': f'rf_{model_type}_model_merged.pth'},
                 {'name': 'dt', 'path': f'dt_{model_type}_model_merged.pth'},
                 {'name': 'svm', 'path': f'svm_{model_type}_model_merged.pth'},
                 {'name': 'knn', 'path': f'knn_{model_type}_model_merged.pth'},
                 {'name': 'gnb', 'path': f'gnb_{model_type}_model_merged.pth'},
                 {'name': 'lr', 'path': f'lr_{model_type}_model_merged.pth'},
                 {'name': 'mlp', 'path': f'{model_type}_mlp_merged.pth'}]
    # test modified test set on new black box
    for bb in bb_models:
        blackbox = load_blackbox(bb['name'], bb['path'])
        results_modified = blackbox_test(test_data=labeled_modified, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_modified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_modified]
        mal = 0
        ben = 0
        for result in results_modified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (modified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        # test unmodified test set on new black box
        results_unmodified = blackbox_test(test_data=labeled_unmodified, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_unmodified = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_unmodified]
        mal = 0
        ben = 0
        for result in results_unmodified:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (unmodified malware files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')
        # test benign test set on new black box
        results_benign = blackbox_test(test_data=labeled_benign, blackbox=blackbox, bb_name=bb["name"])
        if bb['name'] == 'svm':
            results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
        mal = 0
        ben = 0
        for result in results_benign:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{model_type} detector {bb["name"]} predicted (benign files): {str(ben)} benign files and {str(mal)} malicious files on {bb["name"]} {model_type} mixed retrained black box'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')