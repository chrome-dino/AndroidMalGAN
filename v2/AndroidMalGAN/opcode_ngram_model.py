import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

import sys

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

config = configparser.ConfigParser()
config.read("settings.ini")

# FEATURE_COUNT = int(config.get('Features', 'TotalFeatureCount'))
# LEARNING_RATE = 0.0002
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000
L2_LAMBDA = 0.01
BATCH_SIZE = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = 'opcode_ngram_malgan.pth'


def train_ngram_model():
    discriminator, generator, lossfun, disc_optimizer, gen_optimizer = create_opcode_ngram_model(LEARNING_RATE,
                                                                                                 L2_LAMBDA)
    discriminator = discriminator.to(DEVICE)
    generator = generator.to(DEVICE)
    # with open('malware_ngram.csv') as f:
    #     ncols = len(f.readline().split(','))
    data_malware = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1)
    # data_malware = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1, usecols=range(0, 301))
    data_malware = (data_malware.astype(np.bool_)).astype(float)

    # data_malware_gen = np.loadtxt('malware_ngram.csv', delimiter=',', skiprows=1, usecols=range(0, 151))
    # with open('benign_ngram.csv') as f:
    #     ncols = len(f.readline().split(','))
    data_benign = np.loadtxt('benign_ngram.csv', delimiter=',', skiprows=1)
    data_benign = (data_benign.astype(np.bool_)).astype(float)
    labels_benign = data_benign[:, 0]
    data_benign = data_benign[:, 1:]

    labels_malware = data_malware[:, 0]

    data_malware = data_malware[:, 1:]
    # labels_malware_gen = data_malware_gen[:, 0]
    # data_malware_gen = data_malware_gen[:, 1:]
    
    # normalize the data to a range of [-1 1] (b/c tanh output)
    # data_benign = data_benign / np.max(data_benign)
    # data_benign = 2 * data_benign - 1

    # data_malware = data_malware / np.max(data_malware)
    # data_malware = 2 * data_malware - 1

    # trans = RobustScaler()
    # trans = MinMaxScaler()
    # data_malware = trans.fit_transform(data_malware)
    # data_benign = trans.fit_transform(data_benign)

    # mean, std = np.mean(data_benign), np.std(data_benign)
    # data_benign = [(element - mean)/std for element in data_benign]
    # mean, std = np.mean(data_malware_gen), np.std(data_malware_gen)
    # data_malware = [(element - mean)/std for element in data_malware_gen]

    # mean, std = np.mean(data_malware), np.std(data_malware)
    # data_malware = [(element - mean) / std for element in data_malware]

    data_malware = np.array(data_malware)
    data_benign = np.array(data_benign)
    # convert to tensor
    data_tensor_benign = torch.tensor(data_benign).float()
    data_tensor_malware = torch.tensor(data_malware).float()
    # partition = [.8, .1, .1]
    partition = [.95, .05]
    # use scikitlearn to split the data
    train_data_benign, test_data_benign, train_labels_benign, test_labels_benign = train_test_split(
        data_tensor_benign, labels_benign, test_size=partition[1])
    # dev_data_benign, test_data_benign, dev_labels_benign, test_labels_benign = train_test_split(test_data_tensor_benign,
    #     test_labels_benign, test_size=partition[1]/(partition[1] + partition[2]))

    train_data_malware, test_data_malware, train_labels_malware, test_labels_malware = train_test_split(
        data_tensor_malware, labels_malware, test_size=partition[1])
    # dev_data_malware, test_data_malware, dev_labels_malware, test_labels_malware = train_test_split(
    #     test_data_tensor_malware, test_labels_malware, test_size=partition[1] / (partition[1] + partition[2]))

    # then convert them into PyTorch Datasets (note: already converted to tensors)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # train_data_benign = TensorDataset(train_d`a`ta_benign, train_labels_benign)
    # dev_data_benign = TensorDataset(dev_data_benign, dev_labels_benign)
    # test_data_benign = TensorDataset(test_data_benign, test_labels_benign)

    # train_data_malware = TensorDataset(train_data_malware, train_labels_malware)
    # dev_data_malware = TensorDataset(dev_data_malware, dev_labels_malware)
    # test_data_malware = TensorDataset(test_data_malware, test_labels_malware)

    # # finally, translate into dataloader objects
    # train_loader_benign = DataLoader(train_data_benign, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    # test_loader_benign = DataLoader(test_data_benign, shuffle=True, batch_size=test_data_benign.tensors[0].shape[0])
    #
    # train_loader_malware = DataLoader(train_data_malware, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    # test_loader_malware = DataLoader(test_data_malware, shuffle=True, batch_size=test_data_malware.tensors[0].shape[0])

    # initialize accuracies as empties (not storing losses here)
    # trainAcc = []
    # testAcc = []



    losses = torch.zeros((NUM_EPOCHS, 2))
    disDecs = np.zeros((NUM_EPOCHS, 2))  # disDecs = discriminator decisions
    for e in range(NUM_EPOCHS):
        start = 0
        for step in range(data_tensor_malware.shape[0] // BATCH_SIZE):
        # for X, y in train_loader_benign:
            malware = train_data_malware[start: start + BATCH_SIZE]

            benign = train_data_benign[start: start + BATCH_SIZE]
            if len(malware) != BATCH_SIZE or len(benign) != BATCH_SIZE:
                break
            # noise = torch.as_tensor(np.random.uniform(0, 1, (BATCH_SIZE, generator.noise_dims)))
            # malware_noise = torch.cat((malware, noise), 1)
            # malware_noise = malware_noise.to(DEVICE)
            malware = malware.to(DEVICE)
            gen_malware = generator(malware)




            # create minibatches of REAL and FAKE images
            # randidx = torch.randint(data_tensor_benign.shape[0], (BATCH_SIZE,))

            # get batch of benign
            # benign = data_tensor_benign[randidx, :].to(DEVICE)

            # get batch of generated malware
            # gen_malware = generator(torch.randn(BATCH_SIZE, 85).to(DEVICE))  # output of generator

            # labels used for real and fake images
            benign_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
            mal_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)


            ### ---------------- Train the discriminator ---------------- ###

            # forward pass and loss for benign
            benign = benign.to(DEVICE)
            pred_benign = discriminator(benign)  # REAL images into discriminator
            disc_loss_benign = lossfun(pred_benign, benign_labels)  # all labels are 1

            # forward pass and loss for generated malware
            pred_malware = discriminator(gen_malware)  # FAKE images into discriminator
            disc_loss_malware = lossfun(pred_malware, mal_labels)  # all labels are 0

            disc_loss = disc_loss_benign + disc_loss_malware

            losses[e, 0] = disc_loss.item()
            disDecs[e, 0] = torch.mean((pred_benign > .5).float()).detach()

            # backprop
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()


            ### ---------------- Train the generator ---------------- ###

            # create fake images and compute loss
            # gen_malware = generator(torch.randn(BATCH_SIZE, 85).to(DEVICE))

            malware = train_data_malware[start: start + BATCH_SIZE]

            # noise = torch.as_tensor(np.random.uniform(0, 1, (BATCH_SIZE, generator.noise_dims)))
            # malware_noise = torch.cat((malware, noise), 1)
            malware = malware.to(DEVICE)
            gen_malware = generator(malware)
            gen_malware = gen_malware.to(DEVICE)
            pred_malware = discriminator(gen_malware)

            # compute and collect loss and accuracy
            gen_loss = lossfun(pred_malware, benign_labels)
            losses[e, 1] = gen_loss.item()
            disDecs[e, 1] = torch.mean((pred_malware > .5).float()).detach()
            # backprop
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # print out a status message
            if (e + 1) % 500 == 0:
                msg = f'Finished epoch {e + 1}/{NUM_EPOCHS}'
                sys.stdout.write('\r' + msg)

            start = start + BATCH_SIZE


        # use test data?


    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].plot(losses)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Model loss')
    ax[0].legend(['Discrimator', 'Generator'])
    # ax[0].set_xlim([4000,5000])

    ax[1].plot(losses[::5, 0], losses[::5, 1], 'k.', alpha=.1)
    ax[1].set_xlabel('Discriminator loss')
    ax[1].set_ylabel('Generator loss')

    ax[2].plot(disDecs)
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Probablity ("real")')
    ax[2].set_title('Discriminator output')
    ax[2].legend(['Real', 'Fake'])
    plt.savefig('opcode_ngram.png', bbox_inches='tight')
    plt.show()

    return losses, generator, discriminator, disDecs, test_data_malware


def create_opcode_ngram_model(learning_rate, l2lambda):
    # build the model
    discriminator = NgramClassifier()
    generator = NgramGenerator()

    # loss function
    lossfun = nn.BCEWithLogitsLoss()

    # optimizer
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=l2lambda, betas=(.9, .999))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=l2lambda, betas=(.9, .999))

    return discriminator, generator, lossfun, disc_optimizer, gen_optimizer


class NgramClassifier(nn.Module):
    def __init__(self, d_input_dim=350):
        super(NgramClassifier, self).__init__()

        # input layer
        self.input = nn.Linear(d_input_dim, d_input_dim//2)
        # input layer
        self.fc1 = nn.Linear(self.input.out_features, self.input.out_features)
        # input layer
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        # output layer
        self.output = nn.Linear(self.fc3.out_features, 1)
        # batch norm
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):

        x = self.input(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc1(x)
        #  batch norm
        x = self.batch_norm1(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc2(x)
        # batch norm
        x = self.batch_norm2(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc3(x)
        # batch norm
        x = self.batch_norm3(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.output(x)
        x = F.sigmoid(x)
        # x = F.softmax(x)
        return x


class NgramGenerator(nn.Module):
    def __init__(self, noise_dims=50, input_layers=35, g_output_dim=350):
        super(NgramGenerator, self).__init__()

        # amount of noise to add
        self.noise_dims = noise_dims
        self.input_layers = input_layers
        # input layer
        # self.input = nn.Linear(self.noise_dims + self.input_layers, 75)
        self.input = nn.Linear(g_output_dim, g_output_dim*2)
        # input layer
        self.fc1 = nn.Linear(self.input.out_features, self.input.out_features)
        # input layer
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)

        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        # output layer
        self.output = nn.Linear(self.fc3.out_features, 350)
        # self.double()
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):
        # noise = torch.rand(len(x), self.noise_dims)
        # noise = torch.where(noise > 0.5, 1.0, 0.0)
        # noise = noise.to(DEVICE)
        # x = torch.cat((x, noise), -1)
        orig = x.detach().clone()
        x = self.input(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = F.dropout(x, 0.3)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        # x = F.leaky_relu(x, 0.2)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = self.output(x)
        # x = torch.tanh(x)
        # x = F.tanh(x)
        # x = F.relu(x)
        x = F.sigmoid(x)
        x = torch.where(x > 0.5, 1.0, 0.0)
        x = torch.logical_or(orig, x).float()
        return x


def train():
    losses, ngram_generator, discriminator, disDecs, test_data_malware = train_ngram_model()
    print(f'\nLosses: {str(losses)}')
    # print(f'Training Accuracy: {str(trainAcc)}')
    # print(f'Test Accuracy: {str(testAcc)}')
    print(disDecs)
    torch.save(ngram_generator.state_dict(), SAVED_MODEL_PATH)
    print(f'Generator model saved to: {SAVED_MODEL_PATH}')
    ngram_generator.eval()
    test_data_malware = test_data_malware.to(DEVICE)
    gen_malware = ngram_generator(test_data_malware)
    results = discriminator(gen_malware)
    results = torch.where(results > 0.5, True, False)
    mal = 0
    ben = 0
    for result in results:
        if result[0]:
            ben += 1
        else:
            mal += 1
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files')
    print('Finished!')
