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

# np.random.seed(45)
from Util_ import getData
algo_dic = {'LOF':LOF(), 'COF':COF(), 'IForest':IForest(), 'LODA':LODA(),
            'CBLOF':CBLOF(), 'DBSCAN':DBSCAN(), 'HDBSCAN':HDBSCAN()}

name = '26_optdigits.npz'
data = np.load('/Users/walid/Downloads/CSS Animation/DODO_PKDD/Classical/{}'.format(name),
               allow_pickle=True)
X, Y = data['X'], data['y']
from  sklearn.preprocessing import MinMaxScaler
X= MinMaxScaler().fit_transform(X)
print('Evaluation Outlier Detection:')
data = 'Pathto'
#X,  Y = getData(data)
algo_name = 'INNE' ### Write name of comparsion algo
algo_dic[algo_name].fit(X)
y_pred = y_prob = algo_dic[algo_name].decision_function(X)
print('Roc-score:', algo_name, data, ROC(Y, y_pred))


from Util_ import fit_prdict

####DBDO

y_pred = fit_prdict(X)
import densityBasedOutlierDetection
from sklearn.metrics import roc_auc_score
y_score = densityBasedOutlierDetection.y_score
print('Roc-score:', 'DBDO', data,   roc_auc_score(Y, y_score))


