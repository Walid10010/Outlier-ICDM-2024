import sys

sys.setrecursionlimit(1000000)
from sklearn.neighbors import NearestNeighbors
from D2Point import D2Point
from detectNormalRegion import expandNormalRegion
from sortedcontainers import SortedDict
from collections import OrderedDict
import numpy as np

index_i = {}
indexPrim = {}
indexaverage = {}
indices = None
index_kdistance = {}
sorted_index_kdistance = SortedDict()
index_label = {}
index_d2punkt = {}
delete_set = set([])
noise_set = set([])
label_counter = 1
daten_matrix = None
final_label = []
outlier_key_set = set([])
all_n_distance = None
y_score = None
import functools

n_Size = None


def expansionR(min1_):
    global label_counter
    start_punkt = epsilon_abschaetzen()  # sigma = name !
    try:
        epsilon = index_kdistance[start_punkt]
    except:
        return zeichne()
        return
    if not (start_punkt is None or np.all(index_d2punkt[start_punkt].neigh == None)):
        expandNormalRegion(start_punkt, epsilon, index_d2punkt, index_i)
    start_punkt = index_d2punkt[start_punkt]
    density_list = []
    for nachbar in start_punkt.epsilon_neigh:
        try:
            delete_set.add(indexPrim[index_i[nachbar]])
            d2Punkt = index_d2punkt[index_i[nachbar]]
            density_list.append(d2Punkt.k_distance)
            index_label[index_i[nachbar]] = label_counter
            k_dist = index_kdistance[d2Punkt.name]
            del index_kdistance[d2Punkt.name]
            try:
                del sorted_index_kdistance[k_dist][d2Punkt.name]
                if len(sorted_index_kdistance[k_dist]) == 0: del sorted_index_kdistance[k_dist]
            except:
                pass
        except KeyError:
            pass
    label_counter += 1

    if (len(index_kdistance) > 0):
        expansionR(min1_)
    else:
        return zeichne()


def str_to_name(datenpunkt):
    name = ""
    for item in datenpunkt:
        name += str(item) + "x"
    return name


def init_all():
    global index_i, indexPrim, indexaverage, indices, index_kdistance, index_label, index_d2punkt, delete_set, \
        noise_set, label_counter, daten_matrix, final_label, outlier_key_set, sorted_index_kdistance, all_n_distance
    index_i = {}
    indexPrim = {}
    indexaverage = {}
    indices = None
    index_kdistance = {}
    index_label = {}
    index_d2punkt = {}
    delete_set = set([])
    noise_set = set([])
    label_counter = 1
    daten_matrix = None
    final_label = []
    outlier_key_set = set([])
    sorted_index_kdistance = SortedDict()
    all_n_distance = None


def k_nachbaren_berechnen(X, minClusterSize, liste_von_D2Punkte):
    init_all()
    global n_Size, all_n_distance
    n_Size = minClusterSize
    try:
        global daten_matrix
        daten_matrix = X
        global distances, indices
        minClusterSize = minClusterSize
        metricNN = NearestNeighbors(n_neighbors=minClusterSize + 1, leaf_size=minClusterSize + 1).fit(X)
        distances, indices = metricNN.kneighbors(X)
        all_n_distance = distances
        for i in range(len(distances)):
            str = str_to_name(X[i])
            if str in indexPrim:
                pass
            else:
                abstand = 0
                for d in distances[i]:
                    abstand += d
                indexaverage[str] = abstand / minClusterSize
                d2Punkt = D2Point(X[i], indices[i], distances[i], indexaverage[str_to_name(X[i])], str, None)
                liste_von_D2Punkte.append(d2Punkt)
                index_d2punkt[str] = d2Punkt
                indexPrim[str] = i
                index_i[i] = str

                index_kdistance[str] = distances[i][minClusterSize]
                if distances[i][minClusterSize] in sorted_index_kdistance:
                    sorted_index_kdistance[distances[i][minClusterSize]].append(str)
                else:
                    sorted_index_kdistance[distances[i][minClusterSize]] = [str]
        for key in sorted_index_kdistance:
            sorted_index_kdistance[key] = sorted(sorted_index_kdistance[key], key=functools.cmp_to_key(compare_))
            sorted_index_kdistance[key] = OrderedDict.fromkeys(sorted_index_kdistance[key])
        return True
    except ValueError:
        return False


def create_D2Punkt(X):
    liste_von_D2Punkte = []
    for i, item in enumerate(X):
        d2Punkt = D2Point(X[i], indices[i], indexaverage[str_to_name(X[i])])
        liste_von_D2Punkte.append(d2Punkt)

    return liste_von_D2Punkte


def verfeinere(i, str):
    counter = 0
    for y in indices[i]:
        if (y in delete_set):
            counter += 1
        if counter > 1:
            delete_set.add(i)
            noise_set.add(str)
            del index_kdistance[str]
            return False
    return True


counter_fig = 0


def epsilon_abschaetzen_():
    global counter_fig
    min = None
    gefunden = None
    for key, value in index_kdistance.items():
        if min == None:
            min = value
            gefunden = key
        if (value < min):
            gefunden = key
            min = value
        if value == min:
            gefunden = compare(gefunden, key)
    try:
        if (not verfeinere(indexPrim[gefunden], gefunden)):
            koordinate = index_d2punkt[gefunden].corr
            counter_fig += 1
            return epsilon_abschaetzen_()
        if (min == None):
            return None
        global label_counter
        return gefunden
    except KeyError:
        return None


def epsilon_abschaetzen():
    c = 0
    while True:
        global counter_fig
        # min = None
        # sigma_tmp = None
        # for key, value in index_kdistance.items():
        #     if min == None:
        #         min = value
        #         sigma_tmp = key
        #     if (value < min):
        #         sigma_tmp = key
        #         min = value
        #     if value == min:
        #        sigma_tmp =  compare(sigma_tmp, key)
        try:
            (value1, gefunden_list) = sorted_index_kdistance.popitem(0)
            sigma_tmp = gefunden_list.popitem(last=False)[0]
            if len(gefunden_list) > 0:
                sorted_index_kdistance[value1] = gefunden_list
        except Exception as e:
            return None
        try:
            c += 1
            if (not verfeinere(indexPrim[sigma_tmp], sigma_tmp)):
                koordinate = index_d2punkt[sigma_tmp].corr
                counter_fig += 1
                continue
            if (min == None):
                return None
            return sigma_tmp
        except KeyError:
            return None


def compare(d1, d2):
    x = d1
    y = d2
    d1 = index_d2punkt[d1]
    d2 = index_d2punkt[d2]
    d2Coor = d2.corr
    for i, value in enumerate(d1.corr):
        if value == d2Coor[i]:
            continue
        elif value < d2Coor[i]:
            return y
        else:
            return x
    return x


def compare_(d1, d2):
    d1 = index_d2punkt[d1]
    d2 = index_d2punkt[d2]
    d2Coor = d2.corr
    for i, value in enumerate(d1.corr):
        if value == d2Coor[i]:
            continue
        elif value < d2Coor[i]:
            return 1
        else:
            return -1
    return 1


def zeichne():
    y_label = []
    x_label = []
    global daten_matrix, indexPrim
    for daten_punkt in daten_matrix:
        name = str_to_name(daten_punkt)
        if name in index_label:
            x_label.append(daten_punkt)
            y_label.append(index_label[name])
        else:
            y_label.append(0)
            if name in indexPrim:
                outlier_key_set.add(indexPrim[name])
            else:
                pass

    global n_Size, y_score
    global distances
    distances  = np.average(distances, axis=-1)
    #distances = distances[:, -1]
    max_dis = max(distances)
    y_score = []
    for ii, daten_punkt in enumerate(daten_matrix):

        name = str_to_name(daten_punkt)
        if not (name in index_d2punkt):
            y_score.append(0)
            continue
        d2Point = index_d2punkt[name]
        count = 0
        for n in d2Point.neigh:
            if n in outlier_key_set: count += 1

        y_score.append(0.5*count / n_Size  + 0.5*distances[ii]/max_dis)
       # y_score.append((count / n_Size ))


    y_pred_array = np.array(y_label).reshape(-1)
    y_pred_array[y_pred_array == 0] = -1
    y_pred_array[y_pred_array > 0] = 0
    y_pred_array[y_pred_array == -1] = 1
    print('hier')

    threshold_ = np.percentile(distances,100 * (1 - 0.1))
    labels_ = (distances > threshold_).astype(
        'int').ravel()
    global final_label
    final_label = y_pred_array
    y_score = np.array(y_score).reshape(-1)
    #y_score  = distances
