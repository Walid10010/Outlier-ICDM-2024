from pyod.models.abod import ABOD

from pyod.models.deep_svdd import DeepSVDD
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.iforest import  IForest
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.metrics import roc_auc_score as ROC, average_precision_score

from sklearn.metrics import roc_auc_score
import numpy as np
from Util_ import DDIO
from estimators import mle
from math import log2
# np.random.seed(45)
import sys
from sklearn.neighbors import NearestNeighbors
from algorithms import  slof, dao
from Util_ import getData
algo_dic = {'DDIO': DDIO, 'LOF':LOF, 'COF':COF, 'IForest':IForest, 'LODA':LODA,
            'CBLOF':CBLOF, 'DBSCAN':DBSCAN, 'HDBSCAN': hdbscan.HDBSCAN,  'DAO': dao, 'SLOF':slof}
            #'HDBSCAN':HDBSCAN()}


import glob

for algo_name, algo in algo_dic.items():
    for dataname in glob.glob('Classical/*'):
        data = np.load('{}'.format(dataname),
                       allow_pickle=True)
        X, Y = data['X'], data['y']
        from sklearn.preprocessing import MinMaxScaler
        X = MinMaxScaler().fit_transform(X)
        if algo_name == 'DDIO':
            y_pred = DDIO(X)
            import densityBasedOutlierDetection
            from sklearn.metrics import roc_auc_score
            y_score = densityBasedOutlierDetection.y_score
            print('outlier algo', algo_name, 'data name', dataname, 'roc-score:', roc_auc_score(Y, y_score))
        elif  algo_name in ['LOF', 'COF']:
            for k in [20, int((log2(X.shape[0])) ** 2) + 5]:
                algo_ = algo(n_neighbors=k)
                algo_.fit(X)
                y_score = y_prob = algo_.decision_function(X)
                print('outlier algo',algo_name,'k', k  ,'data name', dataname, 'roc-score:', roc_auc_score(Y, y_score))
                print('outlier algo', algo_name, 'k', k,'data name', dataname, 'ap-score:', average_precision_score(Y, y_score))

            pass
        elif algo_name in ['CBLOF', 'IForest']:
            roc__ = 0
            ap__  = 0
            for random_seed in [0, 1, 2, 3, 4, 5, 10, 100, 1000, 10000]:
                algo_ = algo(random_state=random_seed)
                algo_.fit(X)
                y_score = y_prob = algo_.decision_function(X)
                roc__ +=roc_auc_score(Y, y_score)/len(random_seed)
                ap__ +=average_precision_score(Y, y_score)/len(random_seed)
            print('outlier algo',algo_name,  'data name', dataname, 'roc-score:', roc__)
            print('outlier algo',algo_name,  'data name', dataname, 'ap-score:', ap__)



        elif algo_name in ['DBSCAN', 'HDBSCAN']:
            for k in [20, int((log2(X.shape[0])) ** 2) + 5]:
                algo_ = algo(min_samples=k)
                y_pred = y_prob = algo_.fit_predict(X)
                y_pred[y_pred >= 0] = -2
                y_pred[y_pred == -1] = 1
                y_pred[y_pred == -2] = 0
                roc__ = roc_auc_score(Y, y_pred)
                ap__ = average_precision_score(Y, y_pred)

                print('outlier algo', algo_name, 'data name', dataname, 'roc-score:', roc__)
                print('outlier algo', algo_name, 'data name', dataname, 'ap-score:', ap__)


        elif algo_name in ['SLOF']:
            n, d = X.shape
            n = min(int((log2(X.shape[0])) ** 2) + 5, n)
            nn = NearestNeighbors(n_neighbors=n + 1).fit(X)
            dists, idx = nn.kneighbors(X)
            par_estimator = np.asarray([5, 10, 15, 30, 50, 90, 150, 260, 320, 450, 560, 780])
            par_estimator = np.asarray([20])
            par_estimator = par_estimator[par_estimator <= n + 1]
            for k in [20, n]:
                sc_slof = slof(dists[:, :k + 1], idx[:, :k + 1])
                roc__ = roc_auc_score(Y, sc_slof)
                ap__ = average_precision_score(Y, sc_slof)
                print('outlier algo', algo_name, 'data name', dataname, 'roc-score:', roc__)
                print('outlier algo', algo_name, 'data name', dataname, 'ap-score:', ap__)
        elif algo_name in ['DAO']:
            print(dataname)
            n, d = X.shape
            n = min(int((log2(X.shape[0])) ** 2) + 5, n)
            nn = NearestNeighbors(n_neighbors=n + 1).fit(X)
            dists, idx = nn.kneighbors(X)
            par_estimator = np.asarray([5, 10, 15, 30, 50, 90, 150, 260, 320, 450, 560, 780])
            par_estimator = np.asarray([20])
            par_estimator = par_estimator[par_estimator <= n + 1]
            for mpts in par_estimator:

                ids_mle = mle(dists[:, 1:mpts + 1])
                for k in [20, int((log2(X.shape[0])) ** 2) + 5]:
                    sc_dao = dao(dists[:, :k + 1], idx[:, :k + 1], ids_mle)

                    if (np.any(np.isinf(sc_dao))):
                        sc_dao[np.isinf(sc_dao)] = sys.maxsize
                    if (np.any(np.isnan(sc_dao))):
                        sc_dao[np.isnan(sc_dao)] = sys.maxsize

                    roc__ = roc_auc_score(Y, sc_dao)
                    ap__ = average_precision_score(Y, sc_dao)
                    print('outlier algo', algo_name, 'data name', dataname, 'roc-score:', roc__)
                    print('outlier algo', algo_name, 'data name', dataname, 'ap-score:', ap__)












