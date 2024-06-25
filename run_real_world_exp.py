from pyod.models.abod import ABOD

from pyod.models.deep_svdd import DeepSVDD
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.iforest import  IForest
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import roc_auc_score as ROC
import numpy as np
from Util_ import DIDO
y_pred = DIDO(X)
import densityBasedOutlierDetection
from sklearn.metrics import roc_auc_score
y_score = densityBasedOutlierDetection.y_score
# np.random.seed(45)
from Util_ import getData
algo_dic = {'DIDO': DIDO,   'LOF':LOF(), 'COF':COF(), 'IForest':IForest(), 'LODA':LODA(),
            'CBLOF':CBLOF(), 'DBSCAN':DBSCAN(), 'HDBSCAN':HDBSCAN()}


import glob
for dataanme in glob.glob('Classical/*'):
	data = np.load('Classical/{}'.format(dataanme),
	               allow_pickle=True)
	X, Y = data['X'], data['y']
	from sklearn.preprocessing import MinMaxScaler
	X = MinMaxScaler().fit_transform(X)

print('Roc-score:', 'DIDO', data,   roc_auc_score(Y, y_score))


