import csv
import numpy as np

def generate_CRH_feature(dataset):
    csv_file = csv.reader(open("../data/" + dataset + "/answer.csv", "r"))

    qid = dict()
    qid_value = 1
    aid = dict()
    aid_value = 1
    instances = []
    for line in csv_file:
        if line[0] != "question":
            question = line[0]
            worker = line[1]
            answer = line[2]
            if question not in qid:
                qid[question] = qid_value
                qid_value += 1
            if answer not in aid:
                aid[answer] = aid_value
                aid_value += 1
            instances.append([worker, qid[question], aid[answer]])

    q2wl = dict()
    for instance in instances:
        questionId = instance[1]
        if questionId not in q2wl:
            q2wl[questionId] = []
        q2wl[questionId].append(instance)

    with open("../data/" + dataset + "/workers_weight.txt", "r") as f:
        weights = f.readlines()

    w2w = dict()
    for line in weights:
        worker = line.split("\t")[0]
        weight = float(line.split("\t")[1].strip())
        w2w[worker] = weight

    thetas = np.zeros(shape=(len(q2wl), len(aid) + 1), dtype=float)
    for i in range(1, len(q2wl)+1):
        ins = q2wl[i]
        answers = np.zeros(len(aid))
        for j in range(len(ins)):
            answers[ins[j][2]-1] = answers[ins[j][2]-1] + w2w[ins[j][0]]
        for k in range(len(aid)):
            thetas[i-1][k] = answers[k] / answers.sum()
        theta_z = 0
        for k in range(1, len(aid)):
            theta_z = theta_z + thetas[i-1][k] - thetas[i-1][k-1]
        thetas[i-1][len(aid)] = theta_z / len(aid)
        # print(thetas[i-1])
        # print(answers)

    q_feature = open("../data/" + dataset + "/feature.txt", "w")
    for i in range(1, len(q2wl)+1):
        for k in range(len(aid) + 1):
            q_feature.write(str(thetas[i-1][k]))
            q_feature.write("\t")
        q_feature.write(str(i))
        q_feature.write("\n")
    q_feature.close()

    truth = open("../data/" + dataset + "/truth.txt", "w")
    csv_gold = csv.reader(open("../data/" + dataset + "/truth.csv", "r"))

    for line in csv_gold:
        if line[0] != "question":
            question = line[0]
            answer = line[1]
            if question in qid:
                truth.write(str(qid[question]) + "\t" + str(aid[answer]) + "\n")
    truth.close()
