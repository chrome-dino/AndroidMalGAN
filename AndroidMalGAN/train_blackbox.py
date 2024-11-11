import configparser
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from hummingbird.ml import convert
import hummingbird.ml.operator_converters.constants
import numpy as np


config = configparser.ConfigParser()
config.read("settings.ini")

BATCH_SIZE = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')


def train_blackbox(malware_data, benign_data, model_type):
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

    data_tensor_malware = torch.split(data_tensor_malware, list(data_tensor_benign.size())[0])[0]

    partition = [0.7, 0.3]

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

