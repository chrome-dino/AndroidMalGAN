import numpy as np
import torch
from train_blackbox import Classifier2
import os
import pandas as pd
import subprocess
from opcode_ngram_feature_extract import labeled_data
from other_apk_feature_extract import labeled_api_data
from other_apk_feature_extract import labeled_intent_data
from other_apk_feature_extract import labeled_perm_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/'


def load_mlp(path):
    load_model = torch.load(path)
    bb = Classifier2(d_input_dim=350, l1=len(load_model['input.weight']),
                     l2=len(load_model['fc1.weight']),
                     l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))
    bb.load_state_dict(load_model)
    return bb


def hybrid_ensemble_detector(bb_type='', input_file='', n_count=3):

    combined = []

    os.system('rm -rf temp_file_dir')

    with open('api_features.txt', 'r') as file:
        api_features = file.read()
        api_features = api_features.split('\n')

    with open('intent_features.txt', 'r') as file:
        intent_features = file.read()
        intent_features = intent_features.split('\n')

    with open('perm_features.txt', 'r') as file:
        perm_features = file.read()
        perm_features = perm_features.split('\n')

    with open(f'ngram_{str(n_count)}_features.txt', 'r') as file:
        ngram_features = file.read()
        ngram_features = ngram_features.split('\n')

    filename = os.path.basename(input_file).split('.')[0]
    print(f'decompiling file: {input_file} with command: apktool d -f {input_file} -o {filename}')
    command = f'apktool d -f {input_file} -o temp_file_dir/{filename} -q -b'
    command = command.split()
    process = subprocess.Popen(command)
    process.wait()

    intent_data_malware = labeled_intent_data(root_dir='temp_file_dir', intent_features=intent_features,
                                             single_file=True)
    permission_data_malware = labeled_perm_data(root_dir='temp_file_dir', perm_features=perm_features, single_file=True)
    api_data_malware = labeled_api_data(root_dir='temp_file_dir', api_features=api_features, single_file=True)
    ngram_data_malware = labeled_data(root_dir='temp_file_dir', ngram_features=ngram_features, n_count=n_count, single_file=True)

    combined_results = []
    if bb_type == 'mlp':
        blackbox = load_mlp(SAVED_MODEL_PATH + f'{bb_type}_intents_model.pth')
        blackbox = blackbox.to(DEVICE)
        blackbox.eval()
        combined_results.append(blackbox(intent_data_malware))
        blackbox = load_mlp(SAVED_MODEL_PATH + f'{bb_type}_apis_model.pth')
        blackbox = blackbox.to(DEVICE)
        blackbox.eval()
        combined_results.append(blackbox(api_data_malware))
        blackbox = load_mlp(SAVED_MODEL_PATH + f'{bb_type}_permissions_model.pth')
        blackbox = blackbox.to(DEVICE)
        blackbox.eval()
        combined_results.append(blackbox(permission_data_malware))
        blackbox = load_mlp(SAVED_MODEL_PATH + f'{bb_type}_ngrams_{str(n_count)}_model.pth')
        blackbox = blackbox.to(DEVICE)
        blackbox.eval()
        combined_results.append(blackbox(ngram_data_malware))
    else:
        blackbox = torch.load(f'{bb_type}_intents_model.pth')
        blackbox = blackbox.to(DEVICE)
        combined_results.append(blackbox.predict_proba(intent_data_malware))
        blackbox = torch.load(f'{bb_type}_apis_model.pth')
        blackbox = blackbox.to(DEVICE)
        combined_results.append(blackbox.predict_proba(api_data_malware))
        blackbox = torch.load(f'{bb_type}_permissions_model.pth')
        blackbox = blackbox.to(DEVICE)
        combined_results.append(blackbox.predict_proba(permission_data_malware))
        blackbox = torch.load(f'{bb_type}_ngrams_{str(n_count)}_model.pth')
        blackbox = blackbox.to(DEVICE)
        combined_results.append(blackbox.predict_proba(ngram_data_malware))

    if bb_type == 'mlp':
        combined_results_2 = []
        for results in combined_results:
            combined_results_2.append([[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results])
        combined_results = combined_results_2
    if bb_type == 'knn':
        combined_results_2 = []
        for results in combined_results:
            combined_results_2.append(results[:4])
        combined_results = combined_results_2
    if bb_type == 'svm':
        combined_results_2 = []
        for results in combined_results:
            combined_results_2.append([[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results])
        combined_results = combined_results_2

    mal = False
    for results in combined_results:
        if results[0][0] > 0.5:
            mal = True
    if not mal:
        return [0.0, 1.0]
    else:
        return [1.0, 0.0]


def ensemble_detector(model_type='', test_data=None):
    blackboxes = [{'name': 'rf', 'path': f'../rf_{model_type}_model.pth'}, {'name': 'dt', 'path': f'../dt_{model_type}_model.pth'},
     {'name': 'svm', 'path': f'../svm_{model_type}_model.pth'}, {'name': 'knn', 'path': f'../knn_{model_type}_model.pth'},
     {'name': 'gnb', 'path': f'../gnb_{model_type}_model.pth'}, {'name': 'lr', 'path': f'../lr_{model_type}_model.pth'},
     {'name': 'mlp', 'path': f'../{model_type}_mlp_model.pth'}]

    bb_models = []

    for bb in blackboxes:
        if bb['name'] == 'mlp':
            load_model = torch.load(bb['path'])
            blackbox = Classifier2(d_input_dim=350, l1=len(load_model['input.weight']),
                                   l2=len(load_model['fc1.weight']),
                                   l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))
            blackbox.load_state_dict(load_model)
            blackbox = blackbox.to(DEVICE)
            blackbox.eval()
        else:

            blackbox = torch.load(bb['path'])
            if bb['name'] == 'rf' or bb['name'] == 'knn':
                blackbox = blackbox.to(DEVICE_CPU)
            else:
                blackbox = blackbox.to(DEVICE)
        bb_models.append(blackbox)

    combined = []
    for bb in range(len(bb_models)):
        if blackboxes[bb]['name'] == 'mlp':
            test_data = test_data.to(DEVICE)
            results = bb_models[bb](test_data)
            results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
        else:
            test_data = test_data.to(DEVICE_CPU)
            results = bb_models[bb].predict_proba(test_data)
            if blackboxes[bb]['name'] == 'knn':
                results = results[:len(test_data)]
        # if svm
        if blackboxes[bb]['name'] == 'svm':
            results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        combined.append(results)
    combined_results = []
    for results in combined:
        if not isinstance(results, list):
            results = results.tolist()
        results = list(results)
        combined_results.append(results)
    combined_results_ordered = []
    for y in range(len(combined_results[0])):
        row = []
        for x in combined_results:
            row.append(x[y])
        combined_results_ordered.append(row)
    # combined_results = [list(a) for a in zip(combined_results, results)]
    final = []
    for sample in combined_results_ordered:
        mal = 0
        ben = 0
        for result in sample:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        if ben > mal:
            final.append([0.0, 1.0])
        else:
            final.append([1.0, 0.0])

    return final


def validate_ensemble(generator, bb_name, model_name, data_malware, data_benign):

    if bb_name == 'ensemble':
        return
    generator.eval()
    generator.to(DEVICE)
    test_data_malware = data_malware.to(DEVICE)
    test_data_benign = data_benign.to(DEVICE_CPU)
    gen_malware = generator(test_data_malware)

    # gen_malware = generator(malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE_CPU)
    test_data_malware = data_malware.to(DEVICE_CPU)

    results = ensemble_detector(model_type=model_name, test_data=test_data_malware)
    results_benign = ensemble_detector(model_type=model_name, test_data=test_data_benign)

    mal = 0
    ben = 0
    for result in results_benign:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1

    print(f'test set benign predicted: {str(ben)} benign files and {str(mal)} malicious files on ensemble model against malgan trained on {bb_name}')
    acc_ben = ben / (ben + mal)
    tn = ben
    fp = mal

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1

    print(f'test set malware predicted: {str(ben)} benign files and {str(mal)} malicious files on ensemble model against malgan trained on {bb_name}')
    acc_mal = mal / (ben + mal)
    tp_mal = mal
    fn_mal = ben
    acc_mal_ben = (tp_mal + tn) / (fn_mal + fp + tp_mal + tn)
    mal_ben_cm = {'true_pos': tp_mal, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn_mal}
    precision_mal_ben = tp_mal / (tp_mal + fp)
    recall_mal_ben = tp_mal / (tp_mal + fn_mal)
    # f1_mal_ben = 2 * (1 / ((1 / precision_mal_ben) + (1 / recall_mal_ben)))
    if precision_mal_ben + recall_mal_ben == 0:
        f1_mal_ben = None
    else:
        f1_mal_ben = (2 * precision_mal_ben * recall_mal_ben) / (precision_mal_ben + recall_mal_ben)
    results = ensemble_detector(model_type=model_name, test_data=gen_malware)

    # results = torch.where(results > 0.5, True, False)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1

    print(f'test set modified predicted: {str(ben)} benign files and {str(mal)} malicious files on ensemble model against malgan trained on {bb_name}')
    acc_gen = mal / (ben + mal)
    tp_gen = mal
    fn_gen = ben
    acc_gen_ben = (tp_gen + tn) / (fn_gen + fp + tp_gen + tn)
    gen_ben_cm = {'true_pos': tp_gen, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn_gen}
    precision_gen_ben = tp_gen / (tp_gen + fp)
    recall_gen_ben = tp_gen / (tp_gen + fn_gen)
    # f1_gen_ben = 2*(1/((1/precision_gen_ben) + (1/recall_gen_ben)))
    if precision_gen_ben + recall_gen_ben == 0:
        f1_gen_ben = None
    else:
        f1_gen_ben = (2 * precision_gen_ben * recall_gen_ben) / (precision_gen_ben + recall_gen_ben)
    perturbations = 0
    for i in range(len(gen_malware)):
        diff = gen_malware[i] - test_data_malware[i]
        perturbations += diff.sum()
    perturbations = perturbations/len(gen_malware)
    results = {'model': model_name,
               'black box': bb_name + '_vs_ensemble',
               'black box score benign': acc_ben,
               'black box score malware': acc_mal,
               'black box score gen malware': acc_gen,
               'black box accuracy malware': acc_mal_ben,
               'black box accuracy gen malware': acc_gen_ben,
               'malware set confusion matrix': mal_ben_cm,
               'gen malware set confusion matrix': gen_ben_cm,
               'malware set precision': precision_mal_ben,
               'malware set recall': recall_mal_ben,
               'malware set f1': f1_mal_ben,
               'gen malware set precision': precision_gen_ben,
               'gen malware set recall': recall_gen_ben,
               'gen malware set f1': f1_gen_ben,
               'gen malware perturbations avg': perturbations
               }
    if os.path.isfile(f'results.csv'):
        df = pd.DataFrame([results])
        df.to_csv(f'results.csv', mode='a', header=False)
    else:
        df = pd.DataFrame([results])
        df.to_csv(f'results.csv')
