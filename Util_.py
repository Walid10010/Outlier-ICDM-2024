from densityBasedOutlierDetection import expansionR, k_nachbaren_berechnen
import densityBasedOutlierDetection
from sklearn.metrics import roc_auc_score
import scipy.io as sio
import  numpy as np
import matplotlib.pyplot as plt
from readFile import read_file_lec
from sklearn.preprocessing import MinMaxScaler
from math import log2

def DDIO(X):
    para = int((log2(X.shape[0]))**2) +5, 1, 0.
    liste_von_D2Punkte = []
    k_nachbaren_berechnen(X, para[0], liste_von_D2Punkte)
    try:
        return expansionR(para[1])
    except Exception as E:
        print(E)
        return np.array([0]*X.shape[0])


#
# X, Y = np.loadtxt('Dataset/SyntheticNoise1Data'), np.loadtxt('Dataset/SyntheticNoise1Label') ## MODIFY
#
#
# y_pred = DDIO(X)
#
# y_pred = densityBasedOutlierDetection.final_label
# y_score = densityBasedOutlierDetection.y_score
# print('Evaluation Outlier Detection:')
# print('roc:' , roc_auc_score(Y, y_score))



def getData(data):
    if 'SpamBase' in data  or 'Waveform' in data:
        X, Y = read_file_lec(data)
        Y[Y > 0] = -2
        Y[Y == 0] = 1
        y_org = Y.copy()
        y_org[y_org == -2] = 0
        outliers_fraction = np.count_nonzero(y_org) / len(y_org)
        Y[Y == -2] = -1
        print(Y[Y == 1].shape)

    else:
        if 'smtp' in data:
            import mat73
            load_mat = mat73.loadmat('Dataset/{}.mat'.format(data))
            X, Y = load_mat['X'], load_mat['y']
            X = MinMaxScaler().fit_transform(X)

        else:
            load_mat = sio.loadmat('Dataset/{}.mat'.format(data))
            X, Y = load_mat['X'], load_mat['y']
        if 'cover' in data:
            X = MinMaxScaler().fit_transform(X)

        Y = Y.astype(np.float32)
        y_org = Y.copy()
        outliers_fraction = np.count_nonzero(y_org) / len(y_org)
        Y[Y == 0] = -1
    return X, Y



# X, Y = getData('musk')
#
# y_pred = DDIO(X)
#
# y_pred = densityBasedOutlierDetection.final_label
# y_score = densityBasedOutlierDetection.y_score
#
#
# print('Evaluation Outlier Detection:')
# print('roc:' , roc_auc_score(Y, y_score))



