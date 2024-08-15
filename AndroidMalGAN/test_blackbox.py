from opcode_ngram_model import BlackBoxDetector
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from hummingbird.ml import convert
import hummingbird.ml.operator_converters.constants


import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
import sys


matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

config = configparser.ConfigParser()
config.read("settings.ini")

# FEATURE_COUNT = int(config.get('Features', 'TotalFeatureCount'))
# LEARNING_RATE = 0.0002
# LEARNING_RATE = 0.001
# BB_LEARNING_RATE = 0.001
NUM_EPOCHS = 50000
BB_LEARNING_RATE = 0.001
L2_LAMBDA = 0.01
BB_L2_LAMBDA = 0.01
BATCH_SIZE = 150


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')
SAVED_MODEL_PATH = '../opcode_ngram_malgan.pth'
BB_SAVED_MODEL_PATH = '../opcode_ngram_blackbox.pth'


class BlackBoxDetector(nn.Module):
    def __init__(self, d_input_dim=350):
        super(BlackBoxDetector, self).__init__()

        # input layer
        self.input = nn.Linear(d_input_dim, d_input_dim)
        # input layer
        self.fc1 = nn.Linear(self.input.out_features, self.input.out_features * 2)
        # input layer
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        # self.fc3 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        # output layer
        self.output = nn.Linear(self.fc3.out_features, 1)
        # batch norm
        self.batch_norm1 = torch.nn.BatchNorm1d(self.fc1.out_features)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.fc2.out_features)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.fc3.out_features)

    def forward(self, x):

        ######################################
        x = self.input(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.dropout(x, 0.3)
        #######################################
        x = self.fc1(x)
        #  batch norm
        # x = self.batch_norm1(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.dropout(x, 0.3)
        #######################################
        x = self.fc2(x)
        # # batch norm
        # x = self.batch_norm2(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.dropout(x, 0.3)
        #######################################
        x = self.fc3(x)
        # batch norm
        # x = self.batch_norm3(x)
        # x = F.leaky_relu(x)
        x = F.tanh(x)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.dropout(x, 0.3)
        #########################################
        x = self.output(x)
        x = F.sigmoid(x)
        # x = F.softmax(x)
        #########################################
        return x


def create_blackbox_model(learning_rate, l2lambda):
    # build the model
    blackbox = BlackBoxDetector()

    # loss function
    # lossfun = nn.BCEWithLogitsLoss()
    lossfun = nn.BCELoss()

    # optimizer
    # blackbox_optimizer = torch.optim.Adam(blackbox.parameters(), lr=learning_rate, weight_decay=l2lambda,
    #                                       betas=(.9, .999))
    blackbox_optimizer = torch.optim.Adam(blackbox.parameters(), lr=learning_rate)
    return blackbox, lossfun, blackbox_optimizer


blackbox, lossfun, blackbox_optimizer = create_blackbox_model(BB_LEARNING_RATE, BB_L2_LAMBDA)
blackbox = blackbox.to(DEVICE)

data_malware = np.loadtxt('../malware.csv', delimiter=',', skiprows=1)
data_malware = (data_malware.astype(np.bool_)).astype(float)

data_benign = np.loadtxt('../benign.csv', delimiter=',', skiprows=1)
data_benign = (data_benign.astype(np.bool_)).astype(float)
labels_benign = data_benign[:, 0]
data_benign = data_benign[:, 1:]

# labels_malware = data_malware[:, 0]
labels_malware = data_benign[:, 0]
data_malware = data_malware[:, 1:]

# trans = RobustScaler()
# trans = MinMaxScaler()
# data_malware = trans.fit_transform(data_malware)
# data_benign = trans.fit_transform(data_benign)

data_malware = np.array(data_malware)
data_benign = np.array(data_benign)

# convert to tensor
data_tensor_benign = torch.tensor(data_benign).float()
data_tensor_malware = torch.tensor(data_malware).float()

# data_tensor_benign = (data_tensor_benign > 0.5).float()
# data_tensor_malware = (data_tensor_malware > 0.5).float()
data_tensor_malware = torch.split(data_tensor_malware, list(data_tensor_benign.size())[0])[0]

# partition = [.95, .05]
partition = [0.7, 0.3]
# use scikitlearn to split the data
# train_data_benign, test_data_benign, train_labels_benign, test_labels_benign = train_test_split(
#     data_tensor_benign, labels_benign, test_size=partition[1])
#
# train_data_malware, test_data_malware, train_labels_malware, test_labels_malware = train_test_split(
#     data_tensor_malware, labels_malware, test_size=partition[1])
benign_labels = torch.ones(list(data_tensor_benign.size())[0], 1)
mal_labels = torch.zeros(list(data_tensor_benign.size())[0], 1)
labels = torch.cat((mal_labels, benign_labels))
dataset = torch.cat((data_tensor_malware, data_tensor_benign))
xTrain, xTest, yTrain, yTest = train_test_split(
    dataset, labels, train_size=partition[0], random_state=42)

le = LabelEncoder()
yTrain = le.fit_transform(yTrain)
yTest = le.fit_transform(yTest)

DT = DecisionTreeClassifier()
yPredict = DT.fit(xTrain, yTrain).predict(xTest)
torch_dt = convert(DT, 'pytorch')
torch.save(torch_dt, 'dt_model.pth')
print('DT Classification Report')
print(classification_report(yTest, yPredict))
print('DT Accuracy')
print(str(accuracy_score(yTest, yPredict)*100))

KNN = KNeighborsClassifier()
yPredict = KNN.fit(xTrain, yTrain).predict(xTest)
extra_config = {hummingbird.ml.operator_converters.constants.BATCH_SIZE: list(data_tensor_benign.size())[0]}
torch_knn = convert(KNN, 'pytorch', extra_config=extra_config)
torch.save(torch_knn, 'knn_model.pth')
# pickle.dump(KNN, open('knn_model.sav', 'wb'))
print('KNN Classification Report')
print(classification_report(yTest, yPredict))
print('KNN Accuracy')
print(str(accuracy_score(yTest, yPredict)*100))

RF = RandomForestClassifier()
yPredict = RF.fit(xTrain, yTrain).predict(xTest)
torch_rf = convert(RF, 'pytorch')
torch.save(torch_rf, 'rf_model.pth')
print('RF Classification Report')
print(classification_report(yTest, yPredict))
print('RF Accuracy')
print(str(accuracy_score(yTest, yPredict)*100))


torch_dt = torch.load('dt_model.pth')
# torch_rf.eval()
torch_dt = torch_dt.to(DEVICE)
data_tensor_malware = data_tensor_malware.to(DEVICE)
data_tensor_benign = data_tensor_benign.to(DEVICE)
results = torch_dt.predict_proba(data_tensor_malware)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox unmodified malware DT')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((mal/(mal+ben))*100) + '%')


results = torch_dt.predict_proba(data_tensor_benign)

# results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox benign DT')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((ben/(mal+ben))*100) + '%')


torch_knn = torch.load('knn_model.pth')
# torch_rf.eval()
# torch_knn = torch_knn.to(DEVICE)
data_tensor_malware = data_tensor_malware.to(DEVICE_CPU)
data_tensor_benign = data_tensor_benign.to(DEVICE_CPU)
results = torch_knn.predict_proba(data_tensor_malware)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox unmodified malware KNN')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((mal/(mal+ben))*100) + '%')


results = torch_knn.predict_proba(data_tensor_benign)

# results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox benign KNN')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((ben/(mal+ben))*100) + '%')

torch_rf = torch.load('rf_model.pth')
# torch_rf.eval()
torch_rf = torch_rf.to(DEVICE)
# data_tensor_malware = data_tensor_malware.to(DEVICE)
# data_tensor_benign = data_tensor_benign.to(DEVICE)
results = torch_rf.predict_proba(data_tensor_malware)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox unmodified malware RF')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((mal/(mal+ben))*100) + '%')


results = torch_rf.predict_proba(data_tensor_benign)

# results = torch.where(results > 0.5, True, False)
mal = 0
ben = 0
for result in results:
    if result[0] < 0.5:
        ben += 1
    else:
        mal += 1
print('Blackbox benign RF')
print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
print('Accuracy:' + str((ben/(mal+ben))*100) + '%')
exit(-1)




# losses = torch.zeros((NUM_EPOCHS, 2))
# blackbox_decs = np.zeros((NUM_EPOCHS, 2))  # black box decisions
# print('Training Black Box Detector')
#
# for e in range(NUM_EPOCHS):
#     start = 0
#     # for step in range(data_tensor_malware.shape[0] // BATCH_SIZE):
#     mal_idx = np.random.randint(0, train_data_malware.shape[0], BATCH_SIZE)
#     ben_idx = np.random.randint(0, train_data_benign.shape[0], BATCH_SIZE)
#     malware = train_data_malware[mal_idx]
#     benign = train_data_malware[ben_idx]
#
#     # labels used for benign and malicious samples
#     benign_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
#     mal_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
#
#     ### ---------------- Train the black box detector ---------------- ###
#
#     # forward pass and loss for benign
#     benign = benign.to(DEVICE)
#
#     pred_benign = blackbox(benign)  # REAL images into discriminator
#     bb_loss_benign = lossfun(pred_benign, benign_labels)  # all labels are 1
#
#     # forward pass and loss for malware
#     malware = malware.to(DEVICE)
#     pred_malware = blackbox(malware)  # FAKE images into discriminator
#     bb_loss_malware = lossfun(pred_malware, mal_labels)  # all labels are 0
#
#     # bb_loss = (bb_loss_benign + bb_loss_malware) * 0.5
#     bb_loss = (bb_loss_benign + bb_loss_malware)
#
#     losses[e, 0] = bb_loss.item()
#     blackbox_decs[e, 0] = torch.mean((pred_benign > .5).float()).detach()
#
#     # backprop
#     blackbox_optimizer.zero_grad()
#     bb_loss.backward()
#     blackbox_optimizer.step()
#     if (e + 1) % 100 == 0:
#         msg = f'Finished epoch {e + 1}/{NUM_EPOCHS}'
#         sys.stdout.write('\r' + msg)

# plt.plot(losses)
# plt.xlabel('Epoch'), plt.ylabel('loss')
# plt.show()
#
# sys.stdout.write('\nBlackbox training finished!\n')
# torch.save(blackbox.state_dict(), BB_SAVED_MODEL_PATH)
#
# blackbox.eval()
#
# data_malware = np.loadtxt('../malware.csv', delimiter=',', skiprows=1)
# data_malware = (data_malware.astype(np.bool_)).astype(float)
#
# data_benign = np.loadtxt('../benign.csv', delimiter=',', skiprows=1)
# data_benign = (data_benign.astype(np.bool_)).astype(float)
#
# data_benign = data_benign[:, 1:]
# data_malware = data_malware[:, 1:]
# data_malware = np.array(data_malware)
# data_benign = np.array(data_benign)
#
# # convert to tensor
# benign = torch.tensor(data_benign).float()
# malware = torch.tensor(data_malware).float()
#
# malware = malware.to(DEVICE)
# results = blackbox(malware)
#
# results = torch.where(results > 0.5, True, False)
# mal = 0
# ben = 0
# for result in results:
#     if result[0]:
#         ben += 1
#     else:
#         mal += 1
# print('test 1: Blackbox unmodified malware')
# print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
# print('Accuracy:' + str((mal/(mal+ben))*100) + '%')
# benign = benign.to(DEVICE)
# results = blackbox(benign)
# results = torch.where(results > 0.5, True, False)
# mal = 0
# ben = 0
# for result in results:
#     if result[0]:
#         ben += 1
#     else:
#         mal += 1
#
# print('test 2: Blackbox benign')
# print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
# print('Accuracy:' + str((ben/(mal+ben))*100) + '%')
