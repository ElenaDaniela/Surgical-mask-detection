import librosa
import librosa.display
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import csv
from features import features

t_inainte = int(round(time.time() * 1000))

path_base_train = "train/train/"  # bazele fiecarui path
path_base_test = "test/test/"
path_base_validation = "validation/validation/"
file_train = open('train.txt')
file_test = open('test.txt')
file_validation = open('validation.txt')
filename_train = file_train.readline()
filename_test = file_test.readline()
filename_validation = file_validation.readline()

features_train = []

while filename_train:  # citire si stocarea datelor
    if filename_train != "":
        filename_train, clasificator = filename_train.split(',')  # separam numele si clasificarea
        clasificator = clasificator.rstrip()
        filename_train = filename_train.rstrip()  # stergem '\n' de la finalul numelui
        path_train = path_base_train + filename_train
        dataa = features.extract_features(path_train)  # extragerea caracteristicilor
        features_train.append([dataa, clasificator])
        features_traindf = pd.DataFrame(features_train, columns=['caracteristici', 'clasificator'])
        # adaugarea caracteristicilor si clasificarea in Panda dataframe
        X_train = np.array(features_traindf.caracteristici.tolist())  # datele de train
        y_train = np.array(features_traindf.clasificator.tolist())  # clasificatorul de train
    filename_train = file_train.readline()
file_train.close()

features_validation = []

while filename_validation:
    if filename_validation != "":
        filename_validation, clasificator = filename_validation.split(',')  # separam numele si clasificarea
        clasificator = clasificator.rstrip()
        filename_validation = filename_validation.rstrip()  # stergem '\n' de la finalul numelui
        path_validation = path_base_validation + filename_validation
        data = features.extract_features(path_validation)
        features_validation.append([data, clasificator])
        features_validationdf = pd.DataFrame(features_validation, columns=['caracteristici', 'clasificator'])
        X_validation = np.array(features_validationdf.caracteristici.tolist())
        y_validation = np.array(features_validationdf.clasificator.tolist())
        # count += 1
    filename_validation = file_validation.readline()
file_validation.close()

features_test = []

filesname_test = []
while filename_test:  # la fel ca la train doar ca nu mai avem clasificator
    if filename_test != "":
        filename_test = filename_test.rstrip()
        filesname_test.append(filename_test)
        path_test = path_base_test + filename_test
        data = features.extract_features(path_test)
        features_test.append([data])
        features_testdf = pd.DataFrame(features_test, columns=['caracteristici'])
        X_test = np.array(features_testdf.caracteristici.tolist())
    filename_test = file_test.readline()
file_test.close()

# Pentru calcularea scorului pe datele de validare

knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_validation)
print("Accuracy:", metrics.accuracy_score(y_validation, y_pred))
t_dupa = int(round(time.time() * 1000))
print("Calculatorul a \"gandit\" timp de " + str((t_dupa - t_inainte) / 1000 / 60) + " minute. \n")

# matricea de confuzie

cm = confusion_matrix(y_validation, y_pred)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
plt.show()

# pentru predictia pentru datele de test

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# scrierea in fisierul csv

with open('test.csv', mode='w') as csv_file:
    fieldnames = ['name', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',', lineterminator='\n')

    writer.writeheader()
    for i in range(len(filesname_test)):
        writer.writerow({'name': filesname_test[i], 'label': y_pred[i]})

t_dupa = int(round(time.time() * 1000))
print("Calculatorul a \"gandit\" timp de " + str((t_dupa - t_inainte) / 1000 / 60) + " minute. \n")
