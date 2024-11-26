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
from train_blackbox import train_blackbox
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import ray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

import sys
import os

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

configs = configparser.ConfigParser()
configs.read("settings.ini")

BB_MODELS = [{'name': 'rf', 'path': 'rf_apis_model.pth'}, {'name': 'dt', 'path': 'dt_apis_model.pth'},
             {'name': 'svm', 'path': 'svm_apis_model.pth'}, {'name': 'knn', 'path': 'knn_apis_model.pth'},
             {'name': 'gnb', 'path': 'gnb_apis_model.pth'}, {'name': 'lr', 'path': 'lr_apis_model.pth'},
             {'name': 'mlp', 'path': 'mlp_apis_model.pth'}]

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
RAY_TUNE = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/apis_'
SAVED_BEST_MODEL_PATH = 'apis_malgan_best.pth'

MALWARE_CSV = '/home/dsu/Documents/AndroidMalGAN/malware_apis.csv'
BENIGN_CSV = '/home/dsu/Documents/AndroidMalGAN/benign_apis.csv'
# BB_SAVED_MODEL_PATH = 'opcode_ngram_blackbox.pth'

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'


def train_apis_model(config, blackbox=None, bb_name=''):
    # num_epochs = 10000, batch_size = 150, learning_rate = 0.001, l2_lambda = 0.01, g_noise = 0, g_input = 0, g_1 = 0, g_2 = 0, g_3 = 0, c_input = 0, c_1 = 0, c_2 = 0, c_3 = 0
    os.chdir('/home/dsu/Documents/AndroidMalGAN/AndroidMalGAN')
    classifier_params = {'l1': config['c_1'], 'l2': config['c_2'], 'l3': config['c_3']}
    generator_params = {'l1': config['g_1'], 'l2': config['g_2'], 'l3': config['g_3'], 'noise': config['g_noise']}
    discriminator, generator, lossfun, disc_optimizer, gen_optimizer = create_apis_model(config['lr_gen'],
                                                                                                 config[
                                                                                                     'l2_lambda_gen'],
                                                                                                 config['lr_disc'],
                                                                                                 config[
                                                                                                     'l2_lambda_disc'],
                                                                                                 classifier_params,
                                                                                                 generator_params)
    discriminator = discriminator.to(DEVICE)
    generator = generator.to(DEVICE)
    # with open('malware_ngram.csv') as f:
    #     ncols = len(f.readline().split(','))
    data_malware = np.loadtxt(MALWARE_CSV, delimiter=',', skiprows=1)
    # data_malware = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1, usecols=range(0, 301))
    data_malware = (data_malware.astype(np.bool_)).astype(float)

    data_benign = np.loadtxt(BENIGN_CSV, delimiter=',', skiprows=1)
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
    train_data_benign, test_data_benign, train_labels_benign, test_labels_benign = train_test_split(
        data_tensor_benign, labels_benign, test_size=partition[1])
    dev_data_benign, test_data_benign, dev_labels_benign, test_labels_benign = train_test_split(test_data_benign,
        test_labels_benign, test_size=partition[1] / (partition[1] + partition[2]))
    # dev_data_benign, test_data_benign, dev_labels_benign, test_labels_benign = train_test_split(test_data_tensor_benign,
    #     test_labels_benign, test_size=partition[1]/(partition[1] + partition[2]))

    train_data_malware, test_data_malware, train_labels_malware, test_labels_malware = train_test_split(
        data_tensor_malware, labels_malware, test_size=partition[1])
    dev_data_malware, test_data_malware, dev_labels_malware, test_labels_malware = train_test_split(
        test_data_malware, test_labels_malware, test_size=partition[1] / (partition[1] + partition[2]))
    last_loss = float('inf')
    early_stoppage_counter = 0

    losses = torch.zeros((NUM_EPOCHS, 2))
    disDecs = np.zeros((NUM_EPOCHS, 2))  # disDecs = discriminator decisions
    disDecs_dev = np.zeros((NUM_EPOCHS, 2))
    acc_test_dev = np.zeros((NUM_EPOCHS, 2))
    print('Training MalGAN Model: ' + bb_name)
    for e in range(NUM_EPOCHS):

        mal_idx = np.random.randint(0, train_data_malware.shape[0], config['batch_size'])
        ben_idx = np.random.randint(0, train_data_benign.shape[0], config['batch_size'])
        malware = train_data_malware[mal_idx]
        benign = train_data_malware[ben_idx]

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
        if bb_name == 'rf':
            benign = benign.to(DEVICE_CPU)
            gen_malware = gen_malware.to(DEVICE_CPU)
            blackbox = blackbox.to(DEVICE_CPU)
        else:
            benign = benign.to(DEVICE)
            gen_malware = gen_malware.to(DEVICE)
            blackbox = blackbox.to(DEVICE)
        # with torch.no_grad():
        # bb_benign_labels = blackbox(benign).to(DEVICE)
        results = blackbox.predict_proba(benign)
        # if svm
        if bb_name == 'svm':
            results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

        results = np.array([[row[1]] for row in results])
        bb_benign_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)

        results = blackbox.predict_proba(gen_malware)
        # if svm
        if bb_name == 'svm':
            results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]

        benign = benign.to(DEVICE)
        gen_malware = gen_malware.to(DEVICE)

        results = np.array([[row[1]] for row in results])

        bb_mal_labels = torch.from_numpy(results).type(torch.float32).to(DEVICE)

        pred_benign = discriminator(benign)  # REAL images into discriminator
        disc_loss_benign = lossfun(pred_benign, bb_benign_labels)  # all labels are 1

        # forward pass and loss for generated malware
        pred_malware = discriminator(gen_malware)  # FAKE images into discriminator
        disc_loss_malware = lossfun(pred_malware, bb_mal_labels)  # all labels are 0

        disc_loss = (disc_loss_benign + disc_loss_malware)

        losses[e, 0] = disc_loss.item()
        disDecs[e, 0] = torch.mean((pred_benign < .5).float()).detach()

        # backprop
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        ### ---------------- Train the generator ---------------- ###

        # create fake images and compute loss
        mal_idx = np.random.randint(0, train_data_malware.shape[0], config['batch_size'])
        malware = train_data_malware[mal_idx]

        generator.train()
        discriminator.eval()
        malware = malware.to(DEVICE)
        gen_malware = generator(malware)

        binarized_gen_malware_logical_or = gen_malware.to(DEVICE)

        # with torch.no_grad():
        pred_malware = discriminator(binarized_gen_malware_logical_or)
        benign_labels = torch.ones(config['batch_size'], 1).to(DEVICE)

        # compute and collect loss and accuracy
        gen_loss = lossfun(pred_malware, benign_labels)
        losses[e, 1] = gen_loss.item()

        acc = torch.mean((pred_malware > .5).float()).detach()
        disDecs[e, 1] = acc

        # backprop
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        if not RAY_TUNE:
            if bb_name == 'rf':
                gen_malware = gen_malware.to(DEVICE_CPU)

            results = blackbox.predict_proba(gen_malware)[:, -1]
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

            benign = dev_data_benign.to(DEVICE)
            pred_benign = discriminator(benign)  # REAL images into discriminator
            disDecs_dev[e, 0] = torch.mean((pred_benign < .5).float()).detach()

            malware = dev_data_malware.to(DEVICE)
            gen_malware = generator(malware)
            gen_malware = gen_malware.to(DEVICE)
            pred_malware = discriminator(gen_malware)
            disDecs_dev[e, 1] = torch.mean((pred_malware > .5).float()).detach()

            if bb_name == 'rf':
                gen_malware = gen_malware.to(DEVICE_CPU)

            results = blackbox.predict_proba(gen_malware)[:, -1]
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

            acc_test_dev[e, 1] = score

        else:
            ray.train.report(dict(d_loss=disc_loss.item(), g_loss=gen_loss.item(), accuracy=float(acc)))

        if (e + 1) % 1000 == 0:
            # gen_loss, disc_loss = validation(generator, discriminator, test_data_malware, lossfun)
            # msg = f'Gen loss: {str(gen_loss)} / Disc loss: {str(disc_loss)}'
            # sys.stdout.write('\r' + msg + '\n')
            msg = f'Finished epoch {e + 1}/{NUM_EPOCHS}'
            sys.stdout.write('\r' + msg)

            # start = start + BATCH_SIZE
        if not RAY_TUNE:
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))

            ax[0].plot(losses)
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Loss')
            ax[0].set_title(f'APIs Model loss ({str(bb_name)})')
            ax[0].legend(['Discrimator', 'Generator'])
            # ax[0].set_xlim([4000,5000])

            ax[1].plot(losses[::5, 0], losses[::5, 1], 'k.', alpha=.1)
            ax[1].set_xlabel('Discriminator loss')
            ax[1].set_ylabel('Generator loss')
            ax[1].set_title(f'APIs Model Loss Mapping ({str(bb_name)})')

            ax[2].plot(disDecs)
            ax[2].set_xlabel('Epochs')
            ax[2].set_ylabel('Probablity Malicious')
            ax[2].set_title(f'APIs Discriminator Output Train Set ({str(bb_name)})')
            ax[2].legend(['Benign', 'Malware'])

            ax[3].plot(disDecs)
            ax[3].set_xlabel('Epochs')
            ax[3].set_ylabel('Probablity Malicious')
            ax[3].set_title(f'APIs Discriminator Output Dev Set ({str(bb_name)})')
            ax[3].legend(['Benign', 'Malware'])

            ax[4].plot(disDecs)
            ax[4].set_xlabel('Epochs')
            ax[4].set_ylabel('% Blackbox Bypass')
            ax[4].set_title(f'APIs Model Accuracy ({str(bb_name)})')
            ax[4].legend(['Test', 'Dev'])

            plt.savefig(os.path.join('/home/dsu/Documents/AndroidMalGAN/results', 'apis_' + bb_name + '.png'),
                        bbox_inches='tight')
            plt.close(fig)

            torch.save(generator.state_dict(), SAVED_MODEL_PATH + bb_name + '.pth')
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
    print('Testing final model')
    if not RAY_TUNE:
        print('Testing final model')
        validate(generator, blackbox, bb_name, test_data_malware, test_data_benign)
    # validate(generator, blackbox, bb_name, test_data_malware)
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
    print('##############################################################################')

    return
#
#
# def validation(generator, discriminator, malware, lossfun):
#     generator.eval()
#     discriminator.eval()
#     malware = malware.to(DEVICE)
#     gen_malware = generator(malware)
#     binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
#     binarized_gen_malware_logical_or = torch.logical_or(malware, binarized_gen_malware).float()
#     binarized_gen_malware_logical_or = binarized_gen_malware_logical_or.to(DEVICE)
#     # binarized_gen_malware_logical_or = gen_malware.to(DEVICE)
#     pred_malware = discriminator(binarized_gen_malware_logical_or)
#     benign_labels = torch.ones(pred_malware.size(dim=0), 1).to(DEVICE)
#     mal_labels = torch.zeros(pred_malware.size(dim=0), 1).to(DEVICE)
#     # gen_malware = torch.where(malware >= 1, 1.0, gen_malware)
#     # compute and collect loss and accuracy
#     disc_loss = lossfun(pred_malware, mal_labels).item()
#     gen_loss = lossfun(pred_malware, benign_labels).item()
#     return gen_loss, disc_loss


def create_apis_model(learning_rate_gen, l2lambda_gen, learning_rate_disc, l2lambda_disc, classifier,
                              generator):
    # build the model
    # blackbox = BlackBoxDetector()
    discriminator = ApisClassifier(l2=classifier['l1'], l3=classifier['l2'],
                                    l4=classifier['l3'])
    generator = ApisGenerator(l2=generator['l1'], l3=generator['l2'],
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


class ApisClassifier(nn.Module):
    def __init__(self, d_input_dim=350, l2=700, l3=450, l4=325):
        super(ApisClassifier, self).__init__()

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
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        # self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):
        ######################################
        x = self.input(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #######################################
        x = self.fc1(x)
        #  batch norm
        # x = self.batch_norm1(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
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
        x = self.fc3(x)
        # batch norm
        x = self.batch_norm3(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        #########################################
        x = self.output(x)
        x = F.sigmoid(x)
        # x = F.softmax(x)
        #########################################
        return x


class ApisGenerator(nn.Module):
    def __init__(self, noise_dims=70, input_layers=350, l2=840, l3=1480, l4=740):
        super(ApisGenerator, self).__init__()

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
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        # self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

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
        x = self.fc1(x)
        # x = self.batch_norm1(x)
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
        x = self.fc3(x)
        # x = self.batch_norm3(x)
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


def validate(generator, blackbox, bb_name, data_malware, data_benign):
    generator.eval()
    generator.to(DEVICE)
    blackbox.to(DEVICE)
    test_data_malware = data_malware.to(DEVICE)
    test_data_benign = data_benign.to(DEVICE)
    gen_malware = generator(test_data_malware)
    # gen_malware = generator(malware)
    binarized_gen_malware = torch.where(gen_malware > 0.5, 1.0, 0.0)
    binarized_gen_malware_logical_or = torch.logical_or(test_data_malware, binarized_gen_malware).float()
    gen_malware = binarized_gen_malware_logical_or.to(DEVICE)
    results = blackbox.predict_proba(test_data_malware)
    results_benign = blackbox.predict_proba(test_data_benign)
    # if svm
    if bb_name == 'svm':
        results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        results_benign = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results_benign]

    mal = 0
    ben = 0
    for result in results_benign:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1

    print(f'test set benign predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
    acc_ben = ben / (ben + mal)
    tn = ben
    fp = mal

    # results = torch.where(results > 0.5, True, False)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1

    print(f'test set malware predicted: {str(ben)} benign files and {str(mal)} malicious files on {bb_name}')
    acc_mal = mal / (ben + mal)
    tp_mal = mal
    fn_mal = ben
    acc_mal_ben = (tp_mal + tn) / (fn_mal + fp + tp_mal + tn)
    mal_ben_cm = {'true_pos': tp_mal, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn_mal}
    precision_mal_ben = tp_mal / (tp_mal + fp)
    recall_mal_ben = tp_mal / (tp_mal + fn_mal)
    f1_mal_ben = 2 * (1 / ((1 / precision_mal_ben) + (1 / recall_mal_ben)))
    # print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files')

    results = blackbox.predict_proba(gen_malware)
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
    print(f'test set modified predicted: {str(ben)} benign files and {str(mal)} malicious files')
    acc_gen = mal / (ben + mal)
    tp_gen = mal
    fn_gen = ben
    acc_gen_ben = (tp_gen + tn) / (fn_gen + fp + tp_gen + tn)
    gen_ben_cm = {'true_pos': tp_gen, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn_gen}
    precision_gen_ben = tp_gen / (tp_gen + fp)
    recall_gen_ben = tp_gen / (tp_gen + fn_gen)
    f1_gen_ben = 2 * (1 / ((1 / precision_gen_ben) + (1 / recall_gen_ben)))
    perturbations = 0
    for i in range(len(gen_malware)):
        diff = gen_malware[i] - test_data_malware[i]
        perturbations += diff.sum()
    perturbations = perturbations / len(gen_malware)
    results = {'model': 'APIs',
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


def train():
    if RAY_TUNE:
        ray.init()
    if TRAIN_BLACKBOX:
        train_blackbox('malware_apis.csv', 'benign_apis.csv', 'apis')

    for bb_model in BB_MODELS:
        blackbox = torch.load(bb_model['path'])
        blackbox = blackbox.to(DEVICE)
        if RAY_TUNE:
            scheduler = ASHAScheduler(
                metric="g_loss",
                mode="min",
                max_t=NUM_EPOCHS,
                grace_period=60,
                reduction_factor=2,
            )
            tune_config = {
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

            result = tune.run(
                partial(train_apis_model, blackbox=blackbox, bb_name=bb_model['name']),
                config=tune_config,
                num_samples=10000,
                scheduler=scheduler,
                resources_per_trial={"cpu": 4, "gpu": 1},
            )

            best_trial = result.get_best_trial("g_loss", "min", "last")
            best_config_gen = result.get_best_config(metric="g_loss", mode="min")
            best_config_disc = result.get_best_config(metric="d_loss", mode="min")

            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final validation loss: {best_trial.last_result['g_loss']}")
            print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

            print("Best config gen:", best_config_gen)
            print("Best config disc:", best_config_disc)
        else:
            with open('config_apis.json', 'r') as f:
                config = json.load(f)
                train_apis_model(config, blackbox=blackbox, bb_name=bb_model['name'])


    print('Finished!')

