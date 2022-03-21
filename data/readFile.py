import csv
import numpy as np

def generate_CRH_feature(dataset):
    csv_file = csv.reader(open("./zzfx/" + dataset + "/original/answer.csv", "r"))

    qid = dict()
    qid_value = 1
    wid = dict()
    wid_value = 1
    aid = dict()
    aid_value = 1
    instances = []
    answer_line = []
    for line in csv_file:
        if line[0] != "question":
            question = line[0]
            worker = line[1]
            answer = line[2]
            if question not in qid:
                qid[question] = qid_value
                qid_value += 1
            if worker not in wid:
                wid[worker] = wid_value
                wid_value += 1
            if answer not in aid:
                aid[answer] = aid_value
                aid_value += 1
            instances.append([worker, qid[question], aid[answer]])

            answer_line.append([str(qid[question]), str(wid[worker]), answer])
    print(len(qid))

    with open("./zzfx/" + dataset + "/answer.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in answer_line:
            writer.writerow(line)

    # workers_weight = np.loadtxt("./zzfx/" + dataset + "/original/workers_weight.txt")
    # for i in range(workers_weight.shape[0]):
    #     worker_name = str(int(workers_weight[i][0]))
    #     worker_ID = wid[worker_name]
    #     workers_weight[i][0] = int(worker_ID)
    # # print(workers_weight)
    # np.savetxt("./zzfx/" + dataset + "/workers_weight.txt", workers_weight)

    workers_weight = []
    workers_weight_array = []
    with open("./zzfx/" + dataset + "/original/workers_weight.txt", "r") as f:
        data = f.readlines()
        for line in data:
            workers_weight.append(line.strip('\n').split('\t'))
    for workers_weight_line in workers_weight:
        worker_name = workers_weight_line[0]
        worker_ID = int(wid[worker_name])
        worker_weight = float(workers_weight_line[1])
        workers_weight_array.append([worker_ID, worker_weight])
    # print(np.array(workers_weight_array))
    np.savetxt("./zzfx/" + dataset + "/workers_weight.txt", workers_weight_array)

    truth = csv.reader(open("./zzfx/" + dataset + "/original/truth.csv", "r"))
    truth_line = []
    for line in truth:
        if line[0] in qid:
            question_ID = qid[line[0]]
            unite = [str(question_ID),line[1]]
            if unite not in truth_line:
                truth_line.append(unite)
    with open("./zzfx/" + dataset + "/truth.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in truth_line:
            writer.writerow(line)


    q2wl = dict()
    for instance in instances:
        questionId = instance[1]
        if questionId not in q2wl:
            q2wl[questionId] = []
        q2wl[questionId].append(instance)

    with open("./zzfx/" + dataset + "/original/workers_weight.txt", "r") as f:
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

    q_feature = open("./zzfx/" + dataset + "/feature.txt", "w")
    for i in range(1, len(q2wl)+1):
        for k in range(len(aid) + 1):
            q_feature.write(str(thetas[i-1][k]))
            q_feature.write("\t")
        q_feature.write(str(i))
        q_feature.write("\n")
    q_feature.close()

# generate_CRH_feature('AdultContent')#
# generate_CRH_feature('FSI')#
# generate_CRH_feature('HITspam-UC')#
# generate_CRH_feature('leaves16')#
# generate_CRH_feature('product')#
# generate_CRH_feature('relevance')
# generate_CRH_feature('SP')#
# generate_CRH_feature('trec2010')#
# generate_CRH_feature('WeatherSentiment')#

# generate_CRH_feature('aircrowd6')#
# generate_CRH_feature('fej2013')#
# generate_CRH_feature('leaves9')#
# generate_CRH_feature('s5_AdultContent2')
# generate_CRH_feature('saj2013')#
# generate_CRH_feature('valence5')#
generate_CRH_feature('adult2')
