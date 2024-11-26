import configparser
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from hummingbird.ml import convert
import hummingbird.ml.operator_converters.constants
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

config = configparser.ConfigParser()
config.read("settings.ini")

BATCH_SIZE = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')


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


def train_mlp(data_benign, data_malware):
    discriminator = Classifier(l2=classifier['l1'], l3=classifier['l2'],
                                    l4=classifier['l3'])
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
        bb_mal_labels = torch.zeros(list(malware.size())[0], 1)
        pred_benign = discriminator(benign)  # REAL images into discriminator
        disc_loss_benign = lossfun(pred_benign, bb_benign_labels)
        pred_malware = discriminator(malware)  # FAKE images into discriminator
        disc_loss_malware = lossfun(pred_malware, bb_mal_labels)  # all labels are 0
        disc_loss = (disc_loss_benign + disc_loss_malware)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
    torch.save(discriminator.state_dict(), SAVED_MODEL_PATH + bb_name + f'_{str(N_COUNT)}.pth')
    return


def train_blackbox(malware_data, benign_data, model_type, split_data=False):
    data_malware = np.loadtxt(malware_data, delimiter=',', skiprows=1)
    data_malware = (data_malware.astype(np.bool_)).astype(float)

    data_benign = np.loadtxt(benign_data, delimiter=',', skiprows=1)
    data_benign = (data_benign.astype(np.bool_)).astype(float)
    labels_benign = data_benign[:, 0]
    data_benign = data_benign[:, 1:]

    labels_malware = data_benign[:, 0]
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
            data_tensor_benign, labels_benign, test_size=partition[1])
        train_data_malware, data_tensor_malware, train_labels_malware, test_labels_malware = train_test_split(
            data_tensor_malware, labels_malware, test_size=partition[1])
    data_tensor_malware = torch.split(data_tensor_malware, list(data_tensor_benign.size())[0])[0]

    benign_labels = torch.ones(list(data_tensor_benign.size())[0], 1)
    mal_labels = torch.zeros(list(data_tensor_benign.size())[0], 1)
    labels = torch.cat((mal_labels, benign_labels))
    dataset = torch.cat((data_tensor_malware, data_tensor_benign))

    partition = [0.9, 0.1]
    xTrain, xTest, yTrain, yTest = train_test_split(
        dataset, labels, train_size=partition[0], random_state=42)

    le = LabelEncoder()
    yTrain = le.fit_transform(yTrain)
    yTest = le.fit_transform(yTest)

    DT = DecisionTreeClassifier()
    yPredict = DT.fit(xTrain, yTrain).predict(xTest)
    torch_dt = convert(DT, 'pytorch')
    torch.save(torch_dt, f'dt_{model_type}_model.pth')
    print('DT Classification Report')
    print(classification_report(yTest, yPredict))
    print('DT Accuracy')
    print(str(accuracy_score(yTest, yPredict)*100))

    KNN = KNeighborsClassifier()
    yPredict = KNN.fit(xTrain, yTrain).predict(xTest)
    extra_config = {hummingbird.ml.operator_converters.constants.BATCH_SIZE: list(data_tensor_benign.size())[0]}
    torch_knn = convert(KNN, 'pytorch', extra_config=extra_config)
    torch.save(torch_knn, f'knn_{model_type}_model.pth')
    # pickle.dump(KNN, open('knn_model.sav', 'wb'))
    print('KNN Classification Report')
    print(classification_report(yTest, yPredict))
    print('KNN Accuracy')
    print(str(accuracy_score(yTest, yPredict)*100))

    RF = RandomForestClassifier()
    yPredict = RF.fit(xTrain, yTrain).predict(xTest)
    torch_rf = convert(RF, 'pytorch')
    torch.save(torch_rf, f'rf_{model_type}_model.pth')
    print('RF Classification Report')
    print(classification_report(yTest, yPredict))
    print('RF Accuracy')
    print(str(accuracy_score(yTest, yPredict)*100))

    SVM = svm.SVC(kernel='linear')
    # SVM = svm.SVC(kernel='nonlinear')
    yPredict = SVM.fit(xTrain, yTrain).predict(xTest)
    torch_svm = convert(SVM, 'pytorch')
    torch.save(torch_svm, f'svm_{model_type}_model.pth')
    print('SVM Classification Report')
    print(classification_report(yTest, yPredict))
    print('SVM Accuracy')
    print(str(accuracy_score(yTest, yPredict) * 100))

    GNB = GaussianNB()
    yPredict = GNB.fit(xTrain, yTrain).predict(xTest)
    torch_gnb = convert(GNB, 'pytorch')
    torch.save(torch_gnb, f'gnb_{model_type}_model.pth')
    print('GNB Classification Report')
    print(classification_report(yTest, yPredict))
    print('GNB Accuracy')
    print(str(accuracy_score(yTest, yPredict) * 100))

    LR = LogisticRegression()
    yPredict = LR.fit(xTrain, yTrain).predict(xTest)
    torch_lr = convert(LR, 'pytorch')
    torch.save(torch_lr, f'lr_{model_type}_model.pth')
    print('LR Classification Report')
    print(classification_report(yTest, yPredict))
    print('LR Accuracy')
    print(str(accuracy_score(yTest, yPredict) * 100))

    train_mlp(data_tensor_benign, data_tensor_malware)

    torch_dt = torch.load(f'dt_{model_type}_model.pth')
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

    torch_knn = torch.load(f'knn_{model_type}_model.pth')
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

    torch_rf = torch.load(f'rf_{model_type}_model.pth')

    torch_rf = torch_rf.to(DEVICE)
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    data_tensor_benign = data_tensor_benign.to(DEVICE)
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

    torch_svm = torch.load(f'svm_{model_type}_model.pth')
    torch_svm = torch_svm.to(DEVICE)
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    data_tensor_benign = data_tensor_benign.to(DEVICE)
    results = torch_svm.predict_proba(data_tensor_malware)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox unmodified malware SVM')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((mal/(mal+ben))*100) + '%')

    results = torch_svm.predict_proba(data_tensor_benign)

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox benign SVM')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((ben/(mal+ben))*100) + '%')

    torch_gnb = torch.load(f'gnb_{model_type}_model.pth')
    torch_gnb = torch_gnb.to(DEVICE)
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    data_tensor_benign = data_tensor_benign.to(DEVICE)
    results = torch_gnb.predict_proba(data_tensor_malware)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox unmodified malware GNB')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((mal/(mal+ben))*100) + '%')

    results = torch_gnb.predict_proba(data_tensor_benign)

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox benign GNB')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((ben/(mal+ben))*100) + '%')

    torch_lr = torch.load(f'lr _{model_type}_model.pth')
    torch_lr = torch_lr.to(DEVICE)
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    data_tensor_benign = data_tensor_benign.to(DEVICE)
    results = torch_lr.predict_proba(data_tensor_malware)
    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox unmodified malware LR')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((mal/(mal+ben))*100) + '%')

    results = torch_lr.predict_proba(data_tensor_benign)

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox benign LR')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((ben/(mal+ben))*100) + '%')

    torch_mlp = Classifier()
    torch_mlp.load_state_dict(torch.load(SAVED_MODEL_PATH))
    torch_mlp = torch_mlp.to(DEVICE)
    torch_mlp.eval()
    data_tensor_malware = data_tensor_malware.to(DEVICE)
    data_tensor_benign = data_tensor_benign.to(DEVICE)
    results = torch_mlp(data_tensor_malware)

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox unmodified malware MLP')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((mal/(mal+ben))*100) + '%')

    results = torch_mlp(data_tensor_benign)

    mal = 0
    ben = 0
    for result in results:
        if result[0] < 0.5:
            ben += 1
        else:
            mal += 1
    print('Blackbox benign MLP')
    print(f'test set predicted: {str(ben)} benign files and {str(mal)} malicious files\n')
    print('Accuracy:' + str((ben/(mal+ben))*100) + '%')