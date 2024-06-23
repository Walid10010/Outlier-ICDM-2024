import numpy as np


def read_file_lec(str):
    index_counter = - 1
    index_id  = -1
    li_index = []
    outlier_index = -1
    arr_list = []
    with open('Dataset/{}.arff'.format(str), 'r') as file:
        for line in file:
            if '@ATTRIBUTE' in line:
                index_counter = index_counter + 1
                if 'id' in line:
                    index_id = index_counter
                elif'outlier' in line:
                    outlier_index = index_counter
                else:
                    li_index.append(index_counter)
            elif '@' not in line and len(line) > 1:
                arr = line.split(',')
                arr[outlier_index] = 0 if 'yes' in arr[outlier_index]  else 1
                arr = list(map(lambda x: float(x), arr))
                #del arr[index_id]
                arr_list.append(arr)

    complete_arr = np.array(arr_list)

    return complete_arr[:, li_index], np.array(arr_list)[:, outlier_index]







def read_file_mulcross(str, i):
    dic = {}
    classmember = 0
    if (i == 0):
        file = open("artificial" + "/" + str + ".arff", "r")
    else:
        file = open("real-world" + "/" + str + ".arff", "r")
    y = []
    label = []
    for line in file:
        # print(line)
        if (line.startswith("@") or line.startswith("%") or len(line.strip()) == 0):
            pass
        else:
            j = line.split(",")
            alpha = 1
            if ("?" in j):
                continue
            x = 0
            k = []

            for i in range(len(j) - 1):
                k.append(float(j[i]) * alpha + x)
            # if (not j[len(j) - 1].startswith("noise")):
            if ('Anomaly' not in j[-1][1:-2]):

                str = j[len(j) - 1]
                if (str in dic.keys()):
                    label.append(dic[str])
                else:
                    dic[str] = classmember
                    label.append(dic[str])
                    classmember += 1
            else:
                label.append(-1)
            y.append(k)
    return np.array(y), np.array(label).reshape(1, len(label))[0]