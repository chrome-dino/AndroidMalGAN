import configparser
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import ray
import os
import json

BATCH_SIZE = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
RAY_TUNE = True
NUM_EPOCHS = 1000
SAVED_MODEL_PATH = '/home/dsu/Documents/AndroidMalGAN/'


class Classifier(nn.Module):
    def __init__(self, d_input_dim=350, l2=700, l3=450, l4=325):
        super(Classifier, self).__init__()

        self.input = nn.Linear(d_input_dim, l2)
        self.fc1 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.output = nn.Linear(l4, 1)
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):
        x = self.input(x)
        x = F.tanh(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = F.tanh(x)
        x = self.output(x)
        x = F.sigmoid(x)
        return x

class Classifier2(nn.Module):
    def __init__(self, d_input_dim=350, l1=275, l2=200, l3=125, l4=50):
        super(Classifier2, self).__init__()

        self.input = nn.Linear(d_input_dim, l1)
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.output = nn.Linear(l4, 1)
        # self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        # self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):
        x = self.input(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.output(x)
        x = F.sigmoid(x)
        return x


def train_mlps(config, data_benign=None, data_malware=None, model=''):
    os.chdir('/home/dsu/Documents/AndroidMalGAN/AndroidMalGAN')
    classifier = {'l1': config['c_1'], 'l2': config['c_2'], 'l3': config['c_3'], 'l4': config['c_4']}
    discriminator = Classifier2(l2=classifier['l2'], l3=classifier['l3'], l4=classifier['l4'])
    learning_rate_disc = config['lr_disc']
    l2lambda_disc = config['l2_lambda_disc']
    lossfun = nn.BCELoss()

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_disc, weight_decay=l2lambda_disc,
                                      betas=(.9, .999))
    discriminator = discriminator.to(DEVICE)

    for e in range(NUM_EPOCHS):
        mal_idx = np.random.randint(0, data_malware.shape[0], config['batch_size'])
        ben_idx = np.random.randint(0, data_benign.shape[0], config['batch_size'])
        malware = data_malware[mal_idx]
        benign = data_benign[ben_idx]
        malware = malware.to(DEVICE)
        benign = benign.to(DEVICE)
        bb_benign_labels = torch.ones(list(benign.size())[0], 1)
        bb_benign_labels = bb_benign_labels.to(DEVICE)
        bb_mal_labels = torch.zeros(list(malware.size())[0], 1)
        bb_mal_labels = bb_mal_labels.to(DEVICE)
        pred_benign = discriminator(benign)  # REAL images into discriminator
        disc_loss_benign = lossfun(pred_benign, bb_benign_labels)
        pred_malware = discriminator(malware)  # FAKE images into discriminator
        disc_loss_malware = lossfun(pred_malware, bb_mal_labels)  # all labels are 0
        disc_loss = (disc_loss_benign + disc_loss_malware)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        if RAY_TUNE:
            ray.train.report(dict(d_loss=disc_loss.item()))
    torch.save(discriminator.state_dict(), SAVED_MODEL_PATH + model + '_mlp.pth')
    return


split_data = True
# for n in range(3, 11):
model_type = f'apis'
malware_data = f'malware_apis.csv'
benign_data = f'benign_apis.csv'
data_malware = np.loadtxt(malware_data, delimiter=',', skiprows=1)
data_malware = (data_malware.astype(np.bool_)).astype(float)

data_benign = np.loadtxt(benign_data, delimiter=',', skiprows=1)
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
if split_data:
    partition = [0.6, 0.4]
    # partition = [.8, .2]
    # use scikitlearn to split the data
    train_data_benign, data_tensor_benign, train_labels_benign, test_labels_benign = train_test_split(
        data_tensor_benign, labels_benign, test_size=partition[1], random_state=42)
    train_data_malware, data_tensor_malware, train_labels_malware, test_labels_malware = train_test_split(
        data_tensor_malware, labels_malware, test_size=partition[1], random_state=42)
data_tensor_malware = torch.split(data_tensor_malware, list(data_tensor_benign.size())[0])[0]

benign_labels = torch.ones(list(data_tensor_benign.size())[0], 1)
mal_labels = torch.zeros(list(data_tensor_benign.size())[0], 1)
labels = torch.cat((mal_labels, benign_labels))
dataset = torch.cat((data_tensor_malware, data_tensor_benign))

partition = [0.9, 0.1]
xTrain, xTest, yTrain, yTest = train_test_split(
    dataset, labels, train_size=partition[0], random_state=42)

tune_config = {
    "c_1": tune.choice([350, 325, 300, 275, 250, 225, 200, 175, 150]),
    "c_2": tune.choice([325, 300, 275, 250, 225, 200, 175, 150, 125]),
    "c_3": tune.choice([225, 200, 175, 150, 125, 100, 75, 50]),
    "c_4": tune.choice([150, 125, 100, 75, 50, 40, 30, 20, 10]),
    "lr_disc": tune.uniform(0.001, 0.1),
    "l2_lambda_disc": tune.uniform(0.001, 0.1),
    "batch_size": tune.choice([50, 100, 150, 200, 250, 300, 350]),
}

scheduler = ASHAScheduler(
    metric="d_loss",
    mode="min",
    max_t=NUM_EPOCHS,
    grace_period=1,
    reduction_factor=2,
)
result = tune.run(
    tune.with_parameters(train_mlps, data_benign=data_tensor_benign, data_malware=data_tensor_malware, model=model_type),
    config=tune_config,
    num_samples=500,
    scheduler=scheduler,
    resources_per_trial={"cpu": 4, "gpu": 1},
    verbose=0,
    log_to_file=False
)
# best_trial = result.get_best_trial("g_loss", "min", "last")
# best_config_gen = result.get_best_config(metric="g_loss", mode="min")
best_config_disc = result.get_best_config(metric="d_loss", mode="min")
# print(f"Best trial config: {best_trial.config}")
# print(f"Best trial final loss: {best_trial.last_result['g_loss']}")
# print(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")

# print("Best config gen:", best_config_gen)
print("Best config disc:", best_config_disc)

mlp_config = {
    "c_1": best_config_disc['c_1'],
    "c_2": best_config_disc['c_2'],
    "c_3": best_config_disc['c_3'],
    "c_4": best_config_disc['c_4'],
    "lr_disc": best_config_disc['lr_disc'],
    "l2_lambda_disc": best_config_disc['l2_lambda_disc'],
    "batch_size": best_config_disc['batch_size'],
}

with open(f'config_{model_type}_mlp.json', 'w') as f:
    json.dump(mlp_config, f)