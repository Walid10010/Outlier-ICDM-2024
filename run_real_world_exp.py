from pyod.models.abod import ABOD

from pyod.models.deep_svdd import DeepSVDD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.inne import  INNE
from pyod.models.iforest import  IForest
from pyod.models.alad import ALAD
from pyod.models.mo_gaal import MO_GAAL
from sklearn.metrics import roc_auc_score as ROC
import numpy as np

# np.random.seed(45)
from Util_ import getData
algo_dic = {'LOF':LOF(), 'ABOD':ABOD(), 'IForest':IForest(random_state=45), 'INNE': INNE(random_state=45), 'LODA':LODA(),
            'HBOS':HBOS(), 'DeepSVD':DeepSVDD(), 'ALAD':ALAD(), 'SOGAAL':SO_GAAL(), 'MOGAAL': MO_GAAL()}

name = '28_pendigits.npz'
data = np.load('/Users/walid/Downloads/CSS Animation/DODO_PKDD/Classical/{}'.format(name),
               allow_pickle=True)
X, Y = data['X'], data['y']

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


