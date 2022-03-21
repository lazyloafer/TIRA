import numpy as np
import csv
from sklearn.metrics import f1_score, accuracy_score
import time

def find_centroid(ins, i):
    index = np.where(ins[:, i] == max(ins[:, i]))[0][0]
    return ins[index]

def kmeans(dataset, feature):
    with open("../data/" + dataset + "/" + feature+ ".txt","r") as f:
        instances = f.readlines()
    thetas = []
    f_len = len(instances[0].split("\t"))
    for instance in instances:
        theta = []
        for i in range(f_len):
            theta.append(float(instance.split("\t")[i].strip()))
        thetas.append(theta)

    # set Centroid
    ins = np.array(thetas)
    print(ins)
    centroid = []
    k = f_len - 2
    for i in range(k):
        centroid.append(find_centroid(ins, i))

    # repeat process
    max_iti = 100
    iti = 0
    final_cluster = []
    while iti < max_iti:
        iti = iti + 1
        print(str(iti) + "th iterative...")
        # start cluster
        cluster = []
        for i in range(k):
            cluster.append([])

        for i in range(len(ins)):
            min_distance = np.linalg.norm(centroid[0][0:-1]-ins[i][0:-1])
            min_index = 0
            for j in range(1, k):
                eucl = np.linalg.norm(centroid[j][0:-1]-ins[i][0:-1])
                if eucl < min_distance:
                    min_distance = eucl
                    min_index = j
            cluster[min_index].append(ins[i])

        # update centroid
        temp_centroid = []
        for i in range(k):
            sum_j = cluster[i][0]
            for j in range(1, len(cluster[i])):
                sum_j = sum_j + cluster[i][j]
            mean = sum_j / len(cluster[i])

            min_distance = np.linalg.norm(mean[0:-1] - cluster[i][0][0:-1])
            min_index = 0
            for j in range(1, len(cluster[i])):
                eucl = np.linalg.norm(mean[0:-1] - cluster[i][j][0:-1])
                if eucl < min_distance:
                    min_distance = eucl
                    min_index = j

            temp_centroid.append(cluster[i][min_index])

        # the condition of the convergence
        is_convergence = 0
        for i in range(k):
            if not (centroid[i] == temp_centroid[i]).all():
                is_convergence = 1
        if is_convergence == 1:
            centroid = temp_centroid.copy()
        else:
            final_cluster = cluster
            break

    with open("../data/" + dataset + "/truth.txt", "r") as f:
        truthlines = f.readlines()

    # for i in range(k):
    #     for j in range(len(final_cluster[i])):
    #         print(final_cluster[i][j][k+1])

    pred = []
    true = []

    truth_dict = dict()
    for line in truthlines:
        questionId = int(line.split("\t")[0])
        value = int(line.split("\t")[1].strip())
        truth_dict[questionId] = value

    # for key in truth_dict.keys():
    #     #print(key, truth_dict[key])
    #     pred.append(truth_dict[key])
    # pred = np.array(pred)
    # print(pred)

    common = 0
    trueNum = 0
    for i in range(k):
        for j in range(len(final_cluster[i])):
            questionId = int(final_cluster[i][j][k+1])
            if questionId in truth_dict:
                common = common + 1
                pred.append(truth_dict[questionId])
                true.append(i+1)
                if truth_dict[questionId] == i+1:
                    trueNum = trueNum + 1
    # print(true)
    # print(pred)
    print("Macro-F1-Score:", f1_score(true, pred, average='macro'))
    print("Weighted-F1-Score:",f1_score(true, pred, average='weighted'))
    print("testaccu:",accuracy_score(true, pred))
    print("Accuracy:",trueNum / common)

    # csv_gold = csv.reader(open("../data/" + dataset + "/truth.csv", "r"))
    # true = []  # 用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行
    # for line in csv_gold:
    #     if line[1] != "truth":
    #         # print(line[1])  # 打印文件每一行的信息
    #         true.append(int(line[1])+1)
    # true = np.array(true)
    # print(true)
    # print(f1_score(true, pred, average='macro'))
# print(final_cluster)

