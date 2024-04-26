import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import sys

display.set_matplotlib_formats('svg')

config = configparser.ConfigParser()
config.read("settings.ini")

FEATURE_COUNT = int(config.get('Features', 'TotalFeatureCount'))
LEARNING_RATE = 0.0003
NUM_EPOCHS = 50000
L2_LAMBDA = 0.01
BATCH_SIZE = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVED_MODEL_PATH = ''


def train_ngram_model():
    discriminator, generator, lossfun, disc_optimizer, gen_optimizer = create_opcode_ngram_model(LEARNING_RATE,
                                                                                                 L2_LAMBDA)
    discriminator.to(DEVICE)
    generator.to(DEVICE)

    data_malware = np.loadtxt(open('malware.csv', 'rb'), delimiter=',')
    data_benign = np.loadtxt(open('benign.csv', 'rb'), delimiter=',')

    labels_benign = data_benign[:, 0]
    data_benign = data_benign[:, 1:]

    labels_malware = data_malware[:, 0]
    data_malware = data_malware[:, 1:]
    
    # normalize the data to a range of [-1 1] (b/c tanh output)
    # dataNorm_benign = data_benign / np.max(data_benign)
    # dataNorm_benign = 2 * dataNorm_benign - 1
    #
    # dataNorm_malware = data_malware / np.max(data_malware)
    # dataNorm_malware = 2 * dataNorm_malware - 1

    # convert to tensor
    data_tensor_benign = torch.tensor(data_benign).float()
    data_tensor_malware = torch.tensor(data_malware).float()

    partition = [.8, .1, .1]
    # use scikitlearn to split the data
    train_data_benign, test_data_tensor_benign, train_labels_benign, test_labels_benign = train_test_split(
        data_tensor_benign, labels_benign, test_size=partition[0])
    dev_data_benign, test_data_benign, dev_labels_benign, test_labels_benign = train_test_split(test_data_tensor_benign,
        test_labels_benign, test_size=partition[1]/(partition[1] + partition[2]))

    train_data_malware, test_data_tensor_malware, train_labels_malware, test_labels_malware = train_test_split(
        data_tensor_malware, labels_malware, test_size=partition[0])
    dev_data_malware, test_data_malware, dev_labels_malware, test_labels_malware = train_test_split(
        test_data_tensor_malware, test_labels_malware, test_size=partition[1] / (partition[1] + partition[2]))

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    # train_data_benign = TensorDataset(train_data_benign, train_labels_benign)
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
    trainAcc = []
    testAcc = []

    start = 0

    losses = torch.zeros((NUM_EPOCHS, 2))
    disDecs = np.zeros((NUM_EPOCHS, 2))  # disDecs = discriminator decisions
    for e in range(NUM_EPOCHS):
        for step in range(data_tensor_benign.shape[0] // BATCH_SIZE):
        # for X, y in train_loader_benign:
            malware = train_data_malware[start: start + BATCH_SIZE]

            noise = np.random.uniform(0, 1, (BATCH_SIZE, generator.noise_dims))
            malware_noise = torch.cat(malware, noise, 1)

            gen_malware = generator(malware_noise)
            benign = train_data_benign[start: start + BATCH_SIZE]



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

            noise = np.random.uniform(0, 1, (BATCH_SIZE, generator.noise_dims))
            malware_noise = torch.cat(malware, noise, 1)

            gen_malware = generator(malware_noise)
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

        plt.show()

    return losses, trainAcc, testAcc, generator


def create_opcode_ngram_model(learning_rate, l2lambda):
    # build the model
    discriminator = NgramClassifier()
    generator = NgramGenerator()

    # loss function
    lossfun = nn.BCEWithLogitsLoss()

    # optimizer
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=l2lambda)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=l2lambda)

    return discriminator, generator, lossfun, disc_optimizer, gen_optimizer


class NgramClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # input layer
        self.input = nn.Linear(NUM_EPOCHS, 350)
        # input layer
        self.fc1 = nn.Linear(350, 175)
        # input layer
        self.fc2 = nn.Linear(175, 85)
        # output layer
        self.output = nn.Linear(85, 1)

    def forward(self, x):

        x = self.input(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.output(x)

        return x


class NgramGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # amount of noise to add
        self.noise_dims = 50
        self.input_layers = 85
        # input layer
        self.input = nn.Linear(NUM_EPOCHS, self.noise_dims + self.input_layers)
        # input layer
        self.fc1 = nn.Linear(self.noise_dims + self.input_layers, 175)
        # input layer
        self.fc2 = nn.Linear(175, 175)
        # output layer
        self.output = nn.Linear(175, 350)

    def forward(self, x):
        x = self.input(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        # x = torch.tanh(x)
        return x


losses, trainAcc, testAcc, ngram_generator = train_ngram_model()
torch.save(ngram_generator.state_dict(), SAVED_MODEL_PATH)
