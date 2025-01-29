import configparser
import copy
import pandas as pd
import json
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from train_blackbox import train_blackbox, Classifier2
from ensemble_blackbox import ensemble_detector
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from ensemble_blackbox import validate_ensemble

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import ray
from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
import tempfile

import sys
import os
import logging
from datetime import datetime

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

configs = configparser.ConfigParser()
configs.read("settings.ini")

BB_MODELS = [{'name': 'rf', 'path': '../rf_ngram_model.pth'}, {'name': 'dt', 'path': '../dt_ngram_model.pth'},
             {'name': 'svm', 'path': '../svm_ngram_model.pth'}, {'name': 'knn', 'path': '../knn_ngram_model.pth'},
             {'name': 'gnb', 'path': '../gnb_ngram_model.pth'}, {'name': 'lr', 'path': '../lr_ngram_model.pth'},
             {'name': 'mlp', 'path': '../mlp_ngram_model.pth'}]

# FEATURE_COUNT = int(config.get('Features', 'TotalFeatureCount'))
# LEARNING_RATE = 0.0002
LEARNING_RATE = 0.001
EARLY_STOPPAGE_THRESHOLD = 100
BB_LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
L2_LAMBDA = 0.01
BB_L2_LAMBDA = 0.01
BATCH_SIZE = 150
NOISE = 0
TRAIN_BLACKBOX = False
RAY_TUNE = True
SPLIT_DATA = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/opcode_ngram_'
SAVED_BEST_MODEL_PATH = 'opcode_ngram_malgan_best.pth'
TEST_HYBRID = False


# BB_SAVED_MODEL_PATH = 'opcode_ngram_blackbox.pth'

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

LOGGER = logging.getLogger(__name__)
logging.basicConfig(filename=datetime.now().strftime('ngram_%H_%M_%d_%m_%Y.log'), encoding='utf-8', level=logging.INFO)


def train_ngram_model(config, blackbox=None, bb_name='', n_count=3):
    # num_epochs = 10000, batch_size = 150, learning_rate = 0.001, l2_lambda = 0.01, g_noise = 0, g_input = 0, g_1 = 0, g_2 = 0, g_3 = 0, c_input = 0, c_1 = 0, c_2 = 0, c_3 = 0
    os.chdir('/home/dsu/Documents/AndroidMalGAN/AndroidMalGAN')

    classifier_params = {'l1': config['c_1'], 'l2': config['c_2'], 'l3': config['c_3']}
    generator_params = {'l1': config['g_1'], 'l2': config['g_2'], 'l3': config['g_3'], 'noise': config['g_noise']}
    discriminator, generator, lossfun, disc_optimizer, gen_optimizer = create_opcode_ngram_model(config['lr_gen'],
                                                                                                 config[
                                                                                                     'l2_lambda_gen'],
                                                                                                 config['lr_disc'],
                                                                                                 config[
                                                                                                     'l2_lambda_disc'],
                                                                                                 classifier_params,
                                                                                                 generator_params)
    batch_size = config['batch_size']
    discriminator = discriminator.to(DEVICE)
    generator = generator.to(DEVICE)


    # with open('malware_ngram.csv') as f:
    #     ncols = len(f.readline().split(','))
    malware_csv = f'/home/dsu/Documents/AndroidMalGAN/malware_ngram_{str(n_count)}.csv'
    benign_csv = f'/home/dsu/Documents/AndroidMalGAN/benign_ngram_{str(n_count)}.csv'
    data_malware = np.loadtxt(malware_csv, delimiter=',', skiprows=1)
    # data_malware = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1, usecols=range(0, 301))
    data_malware = (data_malware.astype(np.bool_)).astype(float)

    # data_malware_gen = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1, usecols=range(0, 151))
    # with open('benign_ngram.csv') as f:
    #     ncols = len(f.readline().split(','))
    data_benign = np.loadtxt(benign_csv, delimiter=',', skiprows=1)
    data_benign = (data_benign.astype(np.bool_)).astype(float)
    labels_benign = data_benign[:, 0]
    data_benign = data_benign[:, 1:]

    labels_malware = data_malware[:, 0]

    data_malware = data_malware[:, 1:]

    data_malware = np.array(data_malware)
    data_benign = np.array(data_benign)
    # convert to tensor
    data_tensor_benign = torch.tensor(data_benign).float()
    data_tensor_malware = torch.tensor(data_malware).float()
    partition = [.8, .1, .1]
    # partition = [.8, .2]
    # use scikitlearn to split the data
    if SPLIT_DATA:
        data_tensor_benign, test_data_benign, labels_benign, test_labels_benign = train_test_split(
            data_tensor_benign, labels_benign, test_size=0.4, random_state=42)
        data_tensor_malware, test_data_malware, labels_malware, test_labels_malware = train_test_split(
            data_tensor_malware, labels_malware, test_size=0.4, random_state=42)
    train_data_benign, test_data_benign, train_labels_benign, test_labels_benign = train_test_split(
        data_tensor_benign, labels_benign, test_size=partition[1], random_state=42)
    dev_data_benign, test_data_benign, dev_labels_benign, test_labels_benign = train_test_split(test_data_benign,
        test_labels_benign, test_size=partition[1]/(partition[1] + partition[2]), random_state=42)

    train_data_malware, test_data_malware, train_labels_malware, test_labels_malware = train_test_split(
        data_tensor_malware, labels_malware, test_size=partition[1], random_state=42)
    dev_data_malware, test_data_malware, dev_labels_malware, test_labels_malware = train_test_split(
        test_data_malware, test_labels_malware, test_size=partition[1] / (partition[1] + partition[2]), random_state=42)

    last_loss = float('inf')
    early_stoppage_counter = 0

    losses_disc = torch.zeros((NUM_EPOCHS, 1))
    losses_gen = torch.zeros((NUM_EPOCHS, 1))
    disDecs_ben = np.zeros((NUM_EPOCHS, 1))  # disDecs = discriminator decisions
    disDecs_mal = np.zeros((NUM_EPOCHS, 1))
    # disDecs_dev = np.zeros((NUM_EPOCHS, 1))
    disDecs_dev_ben = np.zeros((NUM_EPOCHS, 1))
    disDecs_dev_mal = np.zeros((NUM_EPOCHS, 1))
    acc_test_train = np.zeros((NUM_EPOCHS, 1))
    acc_test_dev = np.zeros((NUM_EPOCHS, 1))
    LOGGER.info('Training MalGAN Model: ' + bb_name)
    print('Training MalGAN Model: ' + bb_name)

    for e in range(NUM_EPOCHS):
        # start = 0
        # for step in range(data_tensor_malware.shape[0] // BATCH_SIZE):
        # for X, y in train_loader_benign:
        mal_idx = np.random.randint(0, train_data_malware.shape[0], batch_size)
        ben_idx = np.random.randint(0, train_data_benign.shape[0], batch_size)
        malware = train_data_malware[mal_idx]
        benign = train_data_benign[ben_idx]

        malware = malware.to(DEVICE)
        # generator.eval()
        # with torch.no_grad():
        generator.eval()
        gen_malware = generator(malware)
        binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
        binarized_gen_malware_logical_or = torch.logical_or(malware, binarized_gen_malware).float()
        gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

        ### ---------------- Train the discriminator ---------------- ###
        discriminator.train()
        # forward pass and loss for benign
        if bb_name == 'ensemble':
            results = ensemble_detector(model_type=f'ngram_{n_count}', test_data=gen_malware)
            results = np.array([[row[1]] for row in results])
            bb_mal_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)
            results = ensemble_detector(model_type=f'ngram_{n_count}', test_data=benign)
            results = np.array([[row[1]] for row in results])
            bb_benign_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)
        else:
            if bb_name == 'rf' or bb_name == 'knn':
                benign = benign.to(DEVICE_CPU)
                gen_malware = gen_malware.to(DEVICE_CPU)
                blackbox = blackbox.to(DEVICE_CPU)
            else:
                benign = benign.to(DEVICE)
                gen_malware = gen_malware.to(DEVICE)
                blackbox = blackbox.to(DEVICE)
            # with torch.no_grad():
            # bb_benign_labels = blackbox(benign).to(DEVICE)

            if bb_name == 'mlp':
                results = blackbox(benign)
                results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
            else:
                results = blackbox.predict_proba(benign)
                if bb_name == 'knn':
                    results = results[:config['batch_size']]
            # if svm
            if bb_name == 'svm':
                results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

            results = np.array([[row[1]] for row in results])
            bb_benign_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)

            # bb_benign_labels = torch.where(bb_benign_labels > .5, 0.0, 1.0)
            # bb_benign_labels = torch.where(bb_benign_labels > 0.5, 1.0, 0.0)
            # bb_mal_labels = blackbox(gen_malware).to(DEVICE)
            if bb_name == 'mlp':
                results = blackbox(gen_malware)
                results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
            else:
                results = blackbox.predict_proba(gen_malware)
                if bb_name == 'knn':
                    results = results[:config['batch_size']]
            # if svm
            if bb_name == 'svm':
                results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

            results = np.array([[row[1]] for row in results])
            bb_mal_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)

        benign = benign.to(DEVICE)
        gen_malware = gen_malware.to(DEVICE)


        # bb_mal_labels = torch.where(bb_mal_labels > .5, 0.0, 1.0)
        # bb_mal_labels = torch.where(bb_mal_labels > 0.5, 1.0, 0.0)

        pred_benign = discriminator(benign)  # REAL images into discriminator
        disc_loss_benign = lossfun(pred_benign, bb_benign_labels)  # all labels are 1

        # forward pass and loss for generated malware
        pred_malware = discriminator(gen_malware)  # FAKE images into discriminator
        disc_loss_malware = lossfun(pred_malware, bb_mal_labels)  # all labels are 0
        # batch_train = torch.cat((benign, gen_malware), 0)
        # pred_disc = discriminator(batch_train)
        # bb_labels = torch.cat((bb_benign_labels, bb_mal_labels), 0)
        # disc_loss = lossfun(pred_disc, bb_labels)
        # disc_loss = (disc_loss_benign + disc_loss_malware)*0.5
        disc_loss = (disc_loss_benign + disc_loss_malware)
        # disc_loss = (disc_loss_benign + disc_loss_malware)
        # disc_loss = np.add(disc_loss_benign, disc_loss_malware) * 0.5
        losses_disc[e, 0] = disc_loss.item()
        disDecs_ben[e, 0] = torch.mean((pred_benign < .5).float()).detach()

        # backprop
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        ### ---------------- Train the generator ---------------- ###

        # create fake images and compute loss
        mal_idx = np.random.randint(0, train_data_malware.shape[0], config['batch_size'])
        malware = train_data_malware[mal_idx]

        # noise = torch.as_tensor(np.random.randint(0, 2, (BATCH_SIZE, generator.noise_dims)))
        # malware = torch.cat((malware, noise), 1)

        generator.train()
        discriminator.eval()
        malware = malware.to(DEVICE)
        gen_malware = generator(malware)
        ################################################
        binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
        binarized_gen_malware_logical_or = torch.logical_or(malware, binarized_gen_malware).float()
        binarized_gen_malware_logical_or = binarized_gen_malware_logical_or.to(DEVICE)
        ################################################
        # binarized_gen_malware_logical_or = gen_malware.to(DEVICE)

        # with torch.no_grad():
        pred_malware = discriminator(binarized_gen_malware_logical_or)
        benign_labels = torch.ones(config['batch_size'], 1).to(DEVICE)

        # gen_malware = torch.where(malware >= 1, 1.0, gen_malware)
        # compute and collect loss and accuracy
        gen_loss = lossfun(pred_malware, benign_labels)
        losses_gen[e, 0] = gen_loss.item()

        acc = torch.mean((pred_malware > .5).float()).detach()
        disDecs_mal[e, 0] = acc
        # gen_malware = torch.where(malware >= 1, 1.0, gen_malware)
        # backprop
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        if not RAY_TUNE:
            if bb_name == 'rf' or bb_name == 'knn':
                gen_malware = gen_malware.to(DEVICE_CPU)

            if bb_name == 'mlp':
                results = blackbox(gen_malware)
                results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
            else:
                results = blackbox.predict_proba(gen_malware)
                if bb_name == 'knn':
                    results = results[:config['batch_size']]
            if bb_name == 'svm':
                results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

            mal = 0
            ben = 0
            for result in results:
                if result[0] < 0.5:
                    ben += 1
                else:
                    mal += 1
            score = ben/(ben+mal)

            acc_test_train[e, 0] = score

            benign = dev_data_benign.to(DEVICE)
            pred_benign = discriminator(benign)  # REAL images into discriminator
            disDecs_dev_ben[e, 0] = torch.mean((pred_benign < .5).float()).detach()

            malware = dev_data_malware.to(DEVICE)
            gen_malware = generator(malware)
            gen_malware = gen_malware.to(DEVICE)
            pred_malware = discriminator(gen_malware)
            disDecs_dev_mal[e, 0] = torch.mean((pred_malware > .5).float()).detach()

            if bb_name == 'rf' or bb_name == 'knn':
                gen_malware = gen_malware.to(DEVICE_CPU)
            if bb_name == 'mlp':
                results = blackbox(gen_malware)
                results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
            else:
                results = blackbox.predict_proba(gen_malware)
                if bb_name == 'knn':
                    results = results[:config['batch_size']]
            if bb_name == 'svm':
                results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

        gen_malware = generator(malware)
        gen_malware = gen_malware.to(DEVICE)
        binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
        binarized_gen_malware_logical_or = torch.logical_or(malware, binarized_gen_malware).float()
        gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

        if bb_name == 'ensemble':
            results = ensemble_detector(model_type=f'ngram_{n_count}', test_data=gen_malware)
        else:
            if bb_name == 'rf' or bb_name == 'knn':
                gen_malware = gen_malware.to(DEVICE_CPU)
            if bb_name == 'mlp':
                results = blackbox(gen_malware)
                results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
            else:
                results = blackbox.predict_proba(gen_malware)
                if bb_name == 'knn':
                    results = results[:config['batch_size']]
            if bb_name == 'svm':
                results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        mal = 0
        ben = 0
        for result in results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        score = ben / (ben + mal)

        acc_test_dev[e, 0] = score


        if not RAY_TUNE:
            # if bb_name == 'rf':
            #     gen_malware = gen_malware.to(DEVICE_CPU)
            #
            # if bb_name == 'mlp':
            #     results = blackbox(gen_malware)[:, -1]
            # else:
            #     results = blackbox.predict_proba(gen_malware)[:, -1]
            # if bb_name == 'svm':
            #     results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
            #
            # mal = 0
            # ben = 0
            # for result in results:
            #     if result[0] < 0.5:
            #         ben += 1
            #     else:
            #         mal += 1
            # score = ben/(ben+mal)
            #
            # acc_test_dev[e, 0] = score

            # benign = dev_data_benign.to(DEVICE)
            # pred_benign = discriminator(benign)  # REAL images into discriminator
            # disDecs_dev[e, 0] = torch.mean((pred_benign < .5).float()).detach()

            # malware = dev_data_malware.to(DEVICE)
            # gen_malware = generator(malware)
            # gen_malware = gen_malware.to(DEVICE)
            # pred_malware = discriminator(gen_malware)
            # disDecs_dev[e, 1] = torch.mean((pred_malware > .5).float()).detach()
            #
            # if bb_name == 'rf':
            #     gen_malware = gen_malware.to(DEVICE_CPU)
            # if bb_name == 'mlp':
            #     results = blackbox(gen_malware)[:, -1]
            # else:
            #     results = blackbox.predict_proba(gen_malware)[:, -1]
            # if bb_name == 'svm':
            #     results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
            #
            # mal = 0
            # ben = 0
            # for result in results:
            #     if result[0] < 0.5:
            #         ben += 1
            #     else:
            #         mal += 1
            # score = ben/(ben+mal)

            acc_test_dev[e, 1] = score

        else:
            metrics = dict(d_loss=disc_loss.item(), g_loss=gen_loss.item(), mean_accuracy=float(score),
                           training_iteration=e)
            ray.train.report(metrics)
            # ray.train.report(dict(d_loss=disc_loss.item(), g_loss=gen_loss.item(), accuracy=float(acc)))

        if e % 1000 == 0:
            msg = f'Finished epoch {e}/{NUM_EPOCHS}'
            sys.stdout.write('\r' + msg)

    sys.stdout.write('\nMalGAN training finished!\n')
    # use test data?


    if not RAY_TUNE:
        torch.save(generator.state_dict(), SAVED_MODEL_PATH + bb_name + f'_{str(n_count)}_final.pth')
        # fig, ax = plt.subplots(1, 5, figsize=(20, 10))
        plt.figure(figsize=(10, 10))
        plt.plot(losses_gen)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Ngram {str(n_count)} Opcode Model gen loss ({str(bb_name)})')
        # plt.legend(['Discrimator', 'Generator'])
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results', f'ngram_{str(n_count)}_' + bb_name + '_gen_loss.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(losses_disc)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Ngram {str(n_count)} Opcode Model disc loss ({str(bb_name)})')
        # plt.legend(['Discrimator', 'Generator'])
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results', f'ngram_{str(n_count)}_' + bb_name + '_disc_loss.png'),
            bbox_inches='tight')
        plt.close('all')
        # ax[0].set_xlim([4000,5000])

        plt.figure(figsize=(10, 10))
        plt.plot(losses_disc[::5, 0], losses_gen[::5, 0], 'k.', alpha=.1)
        plt.xlabel('Discriminator loss')
        plt.ylabel('Generator loss')
        plt.title(f'Ngram {str(n_count)} Opcode Model Loss Mapping ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_loss_map.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(disDecs_ben)
        plt.xlabel('Epochs')
        plt.ylabel('Probablity Malicious')
        plt.title(f'Ngram {str(n_count)} Opcode Discriminator Output Train Set Benign ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_disc_train_ben.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(disDecs_mal)
        plt.xlabel('Epochs')
        plt.ylabel('Probablity Malicious')
        plt.title(f'Ngram {str(n_count)} Opcode Discriminator Output Train Set Malware ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_disc_train_mal.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(disDecs_dev_ben)
        plt.xlabel('Epochs')
        plt.ylabel('Probablity Malicious')
        plt.title(f'Ngram {str(n_count)} Opcode Discriminator Output Dev Set Benign ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_disc_dev_ben.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(disDecs_dev_mal)
        plt.xlabel('Epochs')
        plt.ylabel('Probablity Malicious')
        plt.title(f'Ngram {str(n_count)} Opcode Discriminator Output Dev Set Malware ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_disc_dev_mal.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(acc_test_train)
        plt.xlabel('Epochs')
        plt.ylabel('% Blackbox Bypass')
        plt.title(f'Ngram {str(n_count)} Opcode Model Accuracy Train ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_model_acc_train.png'),
            bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(10, 10))
        plt.plot(acc_test_dev)
        plt.xlabel('Epochs')
        plt.ylabel('% Blackbox Bypass')
        plt.title(f'Ngram {str(n_count)} Opcode Model Accuracy Dev ({str(bb_name)})')
        plt.savefig(
            os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                         f'ngram_{str(n_count)}_' + bb_name + '_model_acc_dev.png'),
            bbox_inches='tight')
        plt.close('all')
        # plt.savefig(os.path.join('/home/dsu/Documents/AndroidMalGAN/results', f'ngram_{str(n_count)}_' + bb_name + '.png'), bbox_inches='tight')
        # plt.close(fig)
        # plt.show()

    # diff = False
    # for p1, p2 in zip(best_model.parameters(), ngram_generator.parameters()):
    #     if p1.data.ne(p2.data).sum() > 0:
    #         diff = True
    #         break
    # if diff:
    #     print('models are different!')
    # else:
    #     print('models are same!')
    # print('/////////////////////////////////////////////////////////////')
    # print(f'Generator model saved to: {SAVED_MODEL_PATH}')
    # print(f'Testing with {str(NOISE)} noise inputs')
    # print(f'Testing best performing model (epoch: {str(best_epoch)})')
    # best_model = NgramGenerator(noise_dims=NOISE)
    # best_model.load_state_dict(torch.load(SAVED_BEST_MODEL_PATH))
    # validate(best_model, blackbox, test_data_malware)
    # print('############################################################')

    if not RAY_TUNE:
        LOGGER.info('*******************************************************************************************************')
        print('*******************************************************************************************************')
        LOGGER.info('Testing final model')
        print('Testing final model')
        validate(generator, blackbox, bb_name, test_data_malware, test_data_benign, n_count)
        validate_ensemble(generator, bb_name, f'ngram_{str(n_count)}', test_data_malware, test_data_benign)
    # test_data_malware = test_data_malware.to(DEVICE)
    # results = discriminator(test_data_malware)
    # # print(results)
    # mal = 0
    # ben = 0
    # for result in results:
    #     if result[0] > 0.5:
    #         ben += 1
    #     else:
    #         mal += 1
    # print(f'discriminator set modified predicted: {str(ben)} benign files and {str(mal)} malicious files')

    return


def create_opcode_ngram_model(learning_rate_gen, l2lambda_gen, learning_rate_disc, l2lambda_disc, classifier,
                              generator):
    # build the model
    # blackbox = BlackBoxDetector()
    discriminator = NgramClassifier(l2=classifier['l1'], l3=classifier['l2'],
                                    l4=classifier['l3'])
    generator = NgramGenerator(l2=generator['l1'], l3=generator['l2'],
                               l4=generator['l3'], noise_dims=generator['noise'])

    # loss function
    # lossfun = nn.BCEWithLogitsLoss()
    lossfun = nn.BCELoss()
    # optimizer
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_disc, weight_decay=l2lambda_disc,
                                      betas=(.9, .999))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate_gen, weight_decay=l2lambda_gen,
                                     betas=(.9, .999))

    return discriminator, generator, lossfun, disc_optimizer, gen_optimizer


class NgramClassifier(nn.Module):
    def __init__(self, d_input_dim=350, l2=700, l3=450, l4=325):
        super(NgramClassifier, self).__init__()

        # input layer
        self.input = nn.Linear(d_input_dim, l2)
        # input layer
        self.fc1 = nn.Linear(l2, l3)
        # input layer
        # self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.fc3 = nn.Linear(l3, l4)
        # output layer
        self.output = nn.Linear(l4, 1)
        # batch norm
        self.batch_norm1 = torch.nn.BatchNorm1d(l2)
        # self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(l3)

    def forward(self, x):
        ######################################
        x = self.input(x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #######################################
        x = self.batch_norm1(x)
        x = self.fc1(x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #######################################
        # x = self.fc2(x)
        # # batch norm
        # x = self.batch_norm2(x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)
        # # x = F.relu(x)
        # # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #######################################
        x = self.batch_norm3(x)
        x = self.fc3(x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #########################################
        x = self.output(x)
        x = F.sigmoid(x)
        # x = F.softmax(x)
        #########################################
        return x


class NgramGenerator(nn.Module):
    def __init__(self, noise_dims=70, input_layers=350, l2=840, l3=1480, l4=740):
        super(NgramGenerator, self).__init__()

        # amount of noise to add
        self.noise_dims = noise_dims
        self.input_layers = input_layers + self.noise_dims
        # self.input_layers = input_layers
        # input layer
        # self.input = nn.Linear(self.noise_dims + self.input_layers, 75)

        self.input = nn.Linear(self.input_layers, l2)
        # input layer
        self.fc1 = nn.Linear(l2, l3)
        # # input layer
        # self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)

        self.fc3 = nn.Linear(l3, l4)

        # output layer
        self.output = nn.Linear(l4, input_layers)

        # self.double()
        self.batch_norm1 = torch.nn.BatchNorm1d(l2)
        # self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(l3)

        # self.input = nn.Linear(self.input_layers, self.input_layers)
        # self.fc1 = nn.Linear(self.input.out_features, self.input.out_features)
        # self.fc3 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        # self.output = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        # noise = torch.rand(len(x), self.noise_dims)
        # noise = torch.where(noise > 0.5, 1.0, 0.0)
        # noise = noise.to(DEVICE)
        # x = torch.cat((x, noise), -1)
        # orig = x.detach().clone()

        # if self.training:
        #     self.input = nn.Linear(self.input_layers + self.noise_dims, self.input_layers*2)
        # noise = torch.as_tensor(np.random.randint(0, 2, (x.shape[0], self.noise_dims)))
        noise = torch.rand(x.shape[0], self.noise_dims)

        noise = noise.to(DEVICE)
        x = torch.cat((x, noise), 1)
        # else:
        #     self.input = nn.Linear(self.input_layers, self.input_layers*2)

        ###########################################
        x = self.input(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.leaky_relu(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = torch.maximum(x, orig)
        # x = F.dropout(x, 0.3)
        ############################################
        x = self.batch_norm1(x)
        x = self.fc1(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.leaky_relu(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = torch.maximum(x, orig)
        # x = F.dropout(x, 0.3)
        ###########################################
        # x = self.fc2(x)
        # x = self.batch_norm2(x)
        # x = F.tanh(x)
        # x = F.dropout(x, 0.3)
        ###########################################
        x = self.batch_norm3(x)
        x = self.fc3(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.leaky_relu(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = torch.where(x > 0.5, 1.0, 0.0)
        # x = torch.logical_or(orig, x).float()
        # x = torch.maximum(x, orig)
        # x = F.dropout(x, 0.3)
        ############################################
        x = self.output(x)
        # x = torch.tanh(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        x = F.sigmoid(x)
        ############################################

        return x


def validate(generator, blackbox, bb_name, data_malware, data_benign, n_count):
    generator.eval()
    # if bb_name == 'rf' or bb_name == 'knn':
    generator = generator.to(DEVICE)
    blackbox = blackbox.to(DEVICE_CPU)
    test_data_malware = data_malware.to(DEVICE)
    test_data_benign = data_benign.to(DEVICE_CPU)
    gen_malware = generator(test_data_malware)
    gen_malware = gen_malware.to(DEVICE_CPU)
    test_data_malware = test_data_malware.to(DEVICE_CPU)

    # gen_malware = generator(malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()

    # if bb_name == 'rf' or bb_name == 'knn':
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE_CPU)
    # else:
    #     gen_malware = binarized_gen_malware_logical_or.to(DEVICE)

    if bb_name == 'mlp':
        results = blackbox(test_data_malware)
        results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
        results_benign = blackbox(test_data_benign)
        results_benign = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results_benign]
    else:
        results = blackbox.predict_proba(test_data_malware)
        if bb_name == 'knn':
            results = results[:len(test_data_malware)]
        results_benign = blackbox.predict_proba(test_data_benign)
        if bb_name == 'knn':
            results_benign = results_benign[:len(test_data_benign)]
    # if svm
    if bb_name == 'svm':
        results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]
    # results = torch.where(results > 0.5, True, False)
    mal = 0
    ben = 0
    for result in results_benign:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    LOGGER.info(f'test set benign predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
    print(f'test set benign predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
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
    LOGGER.info(f'test set malware predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
    print(f'test set malware predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
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

    if bb_name == 'mlp':
        results = blackbox(gen_malware)
        results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
    else:
        results = blackbox.predict_proba(gen_malware)
        if bb_name == 'knn':
            results = results[:len(gen_malware)]
    # if svm
    if bb_name == 'svm':
        results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
    # results = torch.where(results > 0.5, True, False)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    LOGGER.info(f'test set modified predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
    print(f'test set modified predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
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
    results = {'model': f'{str(n_count)} gram opcode boolean',
               'black box': bb_name,
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

    bb_models = [{'name': 'rf', 'path': f'../rf_ngram_{str(n_count)}_model.pth'}, {'name': 'dt', 'path': f'../dt_ngram_{str(n_count)}_model.pth'},
                 {'name': 'svm', 'path': f'../svm_ngram_{str(n_count)}_model.pth'}, {'name': 'knn', 'path': f'../knn_ngram_{str(n_count)}_model.pth'},
                 {'name': 'gnb', 'path': f'../gnb_ngram_{str(n_count)}_model.pth'}, {'name': 'lr', 'path': f'../lr_ngram_{str(n_count)}_model.pth'},
                 {'name': 'mlp', 'path': f'../opcode_ngram_{str(n_count)}_mlp.pth'}]
    for bb_model in bb_models:
        if bb_model['name'] == bb_name:
           continue
        if bb_model['name'] != 'mlp':
            bb = torch.load(bb_model['path'])
            bb = bb.to(DEVICE_CPU)
        else:
            load_model = torch.load(bb_model['path'])
            bb = Classifier2(d_input_dim=350, l1=len(load_model['input.weight']),
                                    l2=len(load_model['fc1.weight']),
                                    l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))
            bb.load_state_dict(load_model)
            bb = bb.to(DEVICE_CPU)
            bb.eval()

        if bb_model['name'] == 'mlp':
            results = bb(gen_malware)
            results = [[0.0, 1.0] if result[0] > 0.5 else [1.0, 0.0] for result in results]
        else:
            results = bb.predict_proba(gen_malware)
            if bb_model['name'] == 'knn':
                results = results[:len(gen_malware)]

        if bb_model['name'] == 'svm':
            results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        # results = torch.where(results > 0.5, True, False)
        mal = 0
        ben = 0
        for result in results:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        result_str = f'{bb_name} ngram {str(n_count)} malgan tested against {bb_model["name"]}: {str(ben)} benign files and {str(mal)} malicious files'
        print(result_str)
        LOGGER.info(result_str)
        with open(f'blackbox_crosscheck_ngram_{str(n_count)}.txt', 'a') as f:
            f.write(result_str + '\n')


def train():
    if RAY_TUNE:
        ray.init()
    for n in range(3, 11):
        print('#######################################################################################################')
        print(f'Starting training for {str(n)}-gram MalGAN')
        print('#######################################################################################################')
        LOGGER.info('#######################################################################################################')
        LOGGER.info(f'Starting training for {str(n)}-gram MalGAN')
        LOGGER.info('#######################################################################################################')
        if RAY_TUNE:
            if ray.is_initialized():
                ray.shutdown()
            ray.init()
        if TRAIN_BLACKBOX:
            train_blackbox(f'../malware_ngram_{str(n)}.csv', f'../benign_ngram_{str(n)}.csv', f'ngram_{str(n)}', split_data=SPLIT_DATA)
        # blackbox = BlackBoxDetector()
        # blackbox.load_state_dict(torch.load(BB_SAVED_MODEL_PATH))
        # blackbox.eval()
        # blackbox = torch.load('dt_model.pth')

        if os.path.exists(f'blackbox_crosscheck_ngram_{str(n)}.txt'):
            os.remove(f'blackbox_crosscheck_ngram_{str(n)}.txt')
        bb_models = [{'name': 'rf', 'path': f'rf_ngram_{str(n)}_model.pth'},
                     {'name': 'dt', 'path': f'dt_ngram_{str(n)}_model.pth'},
                     {'name': 'svm', 'path': f'svm_ngram_{str(n)}_model.pth'},
                     {'name': 'knn', 'path': f'knn_ngram_{str(n)}_model.pth'},
                     {'name': 'gnb', 'path': f'gnb_ngram_{str(n)}_model.pth'},
                     {'name': 'lr', 'path': f'lr_ngram_{str(n)}_model.pth'},
                     {'name': 'mlp', 'path': f'opcode_ngram_{str(n)}_mlp.pth'},
                     {'name': 'ensemble', 'path': ''}]
        for bb_model in bb_models:
            if bb_model['name'] == 'ensemble':
                blackbox = None
            else:
                if bb_model['name'] != 'mlp':
                    blackbox = torch.load('../' + bb_model['path'])
                    blackbox = blackbox.to(DEVICE)
                else:
                    # {'name': 'mlp', 'path': 'mlp_ngram_model.pth'}
                    load_model = torch.load('../' + bb_model['path'])
                    blackbox = Classifier2(d_input_dim=350, l1=len(load_model['input.weight']), l2=len(load_model['fc1.weight']),
                                           l3=len(load_model['fc2.weight']), l4=len(load_model['fc3.weight']))
                    blackbox.load_state_dict(load_model)
                    blackbox = blackbox.to(DEVICE)
                    blackbox.eval()
            if RAY_TUNE:
                search_space = {
                    "g_noise": tune.choice([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
                    "g_1": tune.choice([500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]),
                    "g_2": tune.choice([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]),
                    "g_3": tune.choice([500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]),
                    "c_1": tune.choice([500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]),
                    "c_2": tune.choice([200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]),
                    "c_3": tune.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]),
                    "lr_gen": tune.uniform(0.001, 0.1),
                    "lr_disc": tune.uniform(0.001, 0.1),
                    "l2_lambda_gen": tune.uniform(0.001, 0.1),
                    "l2_lambda_disc": tune.uniform(0.001, 0.1),
                    "batch_size": tune.choice([50, 100, 150, 200, 250, 300, 350]),
                }

                try:
                    with open(f'../config_ngram_{str(n)}_{bb_model["name"]}_malgan.json') as f:
                        best_param = json.load(f)
                except FileNotFoundError as e:
                    with open(f'../config_ngram_{str(n)}_mlp_malgan.json') as f:
                        best_param = json.load(f)
                current_best_params = [best_param]

                scheduler = ASHAScheduler(
                    time_attr='training_iteration',
                    metric='mean_accuracy',
                    mode='max',
                    max_t=100,
                    grace_period=10,
                    reduction_factor=3,
                    brackets=1,
                )
                hyperopt = HyperOptSearch(metric="mean_accuracy", mode="max", points_to_evaluate=current_best_params)
                trainable_with_resource = tune.with_resources(partial(train_ngram_model, blackbox=blackbox, bb_name=bb_model['name']), {"cpu": 4, "gpu": 1})
                tuner = tune.Tuner(
                    trainable_with_resource,
                    run_config=ray.train.RunConfig(
                        verbose=False,
                        name=f"ngram_{str(n)}_test",
                        # Stop when we've reached a threshold accuracy, or a maximum
                        # training_iteration, whichever comes first
                        stop={"mean_accuracy": 0.96, "training_iteration": 1000},
                        storage_path="/tmp/ray_results",
                    ),
                    tune_config=tune.TuneConfig(
                        scheduler=scheduler,
                        search_alg=hyperopt,
                        reuse_actors=True,
                        num_samples=500,

                    ),
                    param_space=search_space
                )

                results = tuner.fit()

                best_trial = results.get_best_result(metric="mean_accuracy", mode="max")
                best_config = best_trial.config
                print(f"Best trial config:\n {best_trial.config}")
                LOGGER.info(f"Best trial config:\n {best_trial.config}")

                df = best_trial.metrics_dataframe
                df = df.drop_duplicates(subset="training_iteration", keep="last")
                plt.figure(figsize=(10, 10))
                df.plot("training_iteration", "mean_accuracy")
                plt.xlabel("Training Iterations")
                plt.ylabel("Mean Accuracy")
                plt.title(f'Ngram {str(n)} Opcode Ray Tune Mean Accuracy ({str(bb_model["name"])})')
                plt.savefig(
                    os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                                 f'ngram_{str(n)}_' + bb_model['name'] + '_ray_mean_acc.png'),
                    bbox_inches='tight')
                plt.close('all')

                plt.figure(figsize=(10, 10))
                df.plot("training_iteration", "g_loss")
                plt.xlabel("Training Iterations")
                plt.ylabel("Generator Loss")
                plt.title(f'Ngram {str(n)} Opcode Ray Tune Generator Loss ({str(bb_model["name"])})')
                plt.savefig(
                    os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                                 f'ngram_{str(n)}_' + bb_model['name'] + '_ray_gen_loss.png'),
                    bbox_inches='tight')
                plt.close('all')

                plt.figure(figsize=(10, 10))
                df.plot("training_iteration", "d_loss")
                plt.xlabel("Training Iterations")
                plt.ylabel("Discriminator Loss")
                plt.title(f'Ngram {str(n)} Opcode Ray Tune Discriminator Loss ({str(bb_model["name"])})')
                plt.savefig(
                    os.path.join('/home/dsu/Documents/AndroidMalGAN/results',
                                 f'ngram_{str(n)}_' + bb_model['name'] + '_ray_disc_loss.png'),
                    bbox_inches='tight')
                plt.close('all')

                with open(f'../config_ngram_{str(n)}_{bb_model["name"]}_malgan.json', 'w') as f:
                    json.dump(best_config, f)

            else:
                with open(f'../config_ngram_{str(n)}_{bb_model["name"]}_malgan.json', 'r') as f:
                    config = json.load(f)
                    train_ngram_model(config, blackbox=blackbox, bb_name=bb_model['name'], n_count=n)
    LOGGER.info('Finished!')
    print('Finished!')


train()