from intents_inject import inject as intent_inject
from permissions_inject import inject as permission_inject
from ngram_inject import inject as ngrams_inject
from apis_inject import inject as api_inject
from hybrid_inject import inject as hybrid_inject
from hybrid_feature_extract import labeled_hybrid_data
from train_blackbox import Classifier
from daisy_chained_malgan import daisy_chain_attack
from ensemble_blackbox import hybrid_ensemble_detector
import subprocess
import os
import torch
import re


DOWNLOAD_TEST_SET = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/hybrid_'
BB_MODELS = [{'name': 'rf', 'path': 'rf_hybrid_model.pth'}, {'name': 'dt', 'path': 'dt_hybrid_model.pth'},
             {'name': 'svm', 'path': 'svm_hybrid_model.pth'}, {'name': 'knn', 'path': 'knn_hybrid_model.pth'},
             {'name': 'gnb', 'path': 'gnb_hybrid_model.pth'}, {'name': 'lr', 'path': 'lr_hybrid_model.pth'},
             {'name': 'mlp', 'path': 'mlp_hybrid_model.pth'}]
DATADIR = os.path.join('DATA')
MALWAREDIR = DATADIR + '/MALWARE'


def blackbox_test(test_data=None, blackbox=None, bb_name=''):
    if bb_name == 'mlp':
        results = blackbox(test_data)
    else:
        results = blackbox.predict_proba(test_data)
    return results


def run_tests(bb_name, n_count=5):
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
    with open("hybrid_samples.txt") as samples:
        for s in samples:
            s = s.rstrip()
            s_list = list(os.path.split(s))
            f_name = s_list[-1]
            s_list[-1] = 'modified_' + f_name
            s_mod = os.path.join(*s_list)

            intent_inject(s, copy_file=True)
            intent_hybrid.append(labeled_hybrid_data(root_dir=s, malware=False, n_count=n_count, single_file=True)[0])

            permission_inject(s, copy_file=True)
            permission_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

            api_inject(s, copy_file=True)
            api_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

            ngrams_inject(s, copy_file=True, n_count=n_count)
            ngram_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

            daisy_chain_attack(file_path=s, n_count=n_count)
            daisychain_hybrid.append(labeled_hybrid_data(root_dir=s_mod, malware=False, n_count=n_count, single_file=True)[0])

    for bb_model in BB_MODELS:
        if bb_model['name'] != 'mlp':
            blackbox = torch.load(bb_model['path'])
            blackbox = blackbox.to(DEVICE)
        else:
            blackbox = Classifier()
            blackbox.load_state_dict(torch.load(SAVED_MODEL_PATH + f'{str(n_count)}_mlp.pth'))
            blackbox = blackbox.to(DEVICE)
            blackbox.eval()

        intent_results = blackbox_test(test_data=intent_hybrid, blackbox=blackbox, bb_name=bb_name)
        permission_results = blackbox_test(test_data=permission_hybrid, blackbox=blackbox, bb_name=bb_name)
        api_results = blackbox_test(test_data=api_hybrid, blackbox=blackbox, bb_name=bb_name)
        ngram_results = blackbox_test(test_data=ngram_hybrid, blackbox=blackbox, bb_name=bb_name)
        daisychain_results = blackbox_test(test_data=daisychain_hybrid, blackbox=blackbox, bb_name=bb_name)

        if bb_name == 'svm':
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
        result_str = f'hybrid detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} intent malgan'
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
        result_str = f'hybrid detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} permission malgan'
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
        result_str = f'hybrid detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} api malgan'
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
        result_str = f'hybrid detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} ngram malgan'
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
        result_str = f'hybrid detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} daisy chain attack'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

    hybrid_results = []
    for bb_model in BB_MODELS:
        with open("hybrid_samples.txt") as samples:
            for s in samples:
                s = s.rstrip()
                s_list = list(os.path.split(s))
                f_name = s_list[-1]
                s_list[-1] = 'modified_' + f_name
                s_mod = os.path.join(*s_list)

                hybrid_inject(s, copy_file=True, n_count=n_count)
                hybrid_results.append(hybrid_ensemble_detector(bb_type=bb_model['name'], input_file=s_mod))
        mal = 0
        ben = 0
        for result in hybrid_results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'ensemble detector {bb_name} predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name} hybrid feature set malgan'
        print(result_str)
        with open('test_suite_results.txt', 'a') as f:
            f.write(result_str + '\n')

    return
