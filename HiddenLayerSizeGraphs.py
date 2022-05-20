import os
import random

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

dirPath = os.path.dirname(os.path.realpath(__file__))
dataPath  = dirPath + "/data"

rawX = []
X = []
Y = []
YNumber = []
paddingArray = [0]*22


for tctodd in os.listdir(dataPath):
    i = 0
    currentPath = dataPath + '/' + tctodd
    for xPath in os.listdir(currentPath):
        i = i + 1
        f = os.path.join(dataPath, tctodd)
        fileData = open(currentPath + '/' + xPath, 'r')
        Y.append(xPath.partition("-")[0])
        rawX.append(fileData.readlines())
        fileData.close()  

#formatting rawData
for samples in rawX:
    rows = []
    for items in samples:
        items = items.replace('\n','')
        a_list = items.split('\t')
        map_object = map(float, a_list)
        list_of_integers = list(map_object)
        rows.append(list_of_integers)
        #rows.append(items.split('\t'))

    X.append(rows)


#zero padding to make all samples the same length
for i in range(len(X)):
    while len(X[i]) < 136:
        X[i].append(paddingArray)
    

X = np.array(X, dtype=object)
Y = np.array(Y, dtype=object)

nsamples, nx, ny = X.shape
X = X.reshape((nsamples,nx*ny))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state = 1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=5/9, random_state = 1)

#mlp = MLPClassifier(random_state=1, max_iter=500).fit(x_train, y_train)


mlp = []
mlp.append(MLPClassifier(hidden_layer_sizes=([250, 200, 150]), max_iter=10, random_state=1, activation='relu'))
mlp.append(MLPClassifier(hidden_layer_sizes=([200,100, 50]), max_iter=10, random_state=1, activation='relu'))
mlp.append(MLPClassifier(hidden_layer_sizes=([200,100]), max_iter=10, random_state=1, activation='relu'))
mlp.append(MLPClassifier(hidden_layer_sizes=([250,150]), max_iter=10, random_state=1, activation='relu'))



graphingLabels = ['[250, 200, 150]', '[200,100, 50]' ,'[200,100]' , '[250,150]']

N_TRAIN_SAMPLES = x_train.shape[0]
N_EPOCHS = 200
N_BATCH = 128
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []
scores_val = []


# EPOCH
for i in range(len(mlp)):
    epoch = 0
    temp_scores_train = []
    temp_scores_test = []
    temp_scores_val = []
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(x_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp[i].partial_fit(x_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        temp_scores_train.append(mlp[i].score(x_train, y_train))
        

        # SCORE VALIDATION
        temp_scores_val.append(mlp[i].score(x_val, y_val))
        

        # SCORE TEST
        temp_scores_test.append(mlp[i].score(x_test, y_test))  
        

        epoch += 1
    scores_train.append(temp_scores_train)
    scores_val.append(temp_scores_val)
    scores_test.append(temp_scores_test)    


""" Plot """
fig, ax = plt.subplots(3, sharex=True, sharey=True)

ax[0].set_title('Train')

ax[1].set_title('Validation')

ax[2].set_title('Test')
for i in range(len(mlp)):
    ax[0].plot(scores_train[i], label = graphingLabels[i])
    ax[1].plot(scores_val[i], label = graphingLabels[i])
    ax[2].plot(scores_test[i], label = graphingLabels[i])
ax[0].legend(loc=8)
ax[1].legend(loc=8)
ax[2].legend(loc=8)
fig.suptitle("Accuracy over epochs", fontsize=14)
for i in range(len(scores_val)):
    print(graphingLabels[i], '\t', scores_val[i][-1])
plt.show()