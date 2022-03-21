import csv
import numpy as np
from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge
from tf_geometric.data.graph import Graph

def get_feature(dataset=None, object_feature_path=None, answer_path=None, source_confidence_path=None):
    """"
        object_feature_path = feature.txt
        answer_path = answer.csv
        source_confidence_path = workers_weight.txt
        """
    object_feature = np.loadtxt(object_feature_path)[:,:-2]
    answer = csv.reader(open(answer_path, 'r'))
    source_weight = np.loadtxt(source_confidence_path)[:,-1]
    source_index = []
    edge_label = []
    for line in answer:
        if dataset == 'valence7' or dataset == 'trec2011':
            source_index.append(int(line[1]))
        else:
            source_index.append(int(line[1])-1)
        edge_label.append(int(line[2]))
    source_num = np.array(source_index, dtype=np.int32).max() + 1
    class_num = np.array(edge_label, dtype=np.int32).max() + 1
    source_feature = np.zeros(shape=(source_num, class_num))
    for i in range(len(source_index)):
        source_feature[source_index[i]][edge_label[i]] = source_weight[source_index[i]]

    node_features = np.vstack([object_feature, source_feature])
    return node_features


def get_edge(dataset=None, answer_path=None):
    """"
        answer_path = answer.csv
        """
    answer = csv.reader(open(answer_path, 'r'))
    object_index = []
    source_index = []
    edge_label = []
    for line in answer:
        if dataset == 'valence7' or dataset == 'trec2011':
            object_index.append(int(line[0]))
            source_index.append(int(line[1]))
        else:
            object_index.append(int(line[0])-1)
            source_index.append(int(line[1])-1)
        edge_label.append(int(line[2]))

    object_index = np.array(object_index, dtype=np.int32)

    object_num = object_index.max() + 1

    source_index = np.array(source_index, dtype=np.int32) + object_num

    edge_label = np.array(edge_label, dtype=np.int32)
    edge_label = np.vstack([object_index, source_index, edge_label])

    edge_index, _ = remove_self_loop_edge(edge_label[:2,:])
    edge_index, _ = convert_edge_to_directed(edge_index)

    return edge_index, edge_label

def get_label(dataset=None, object_label_path=None):
    object_classes = csv.reader(open(object_label_path, 'r'))
    object_index = []
    truth_index = []
    for line in object_classes:
        if dataset == 'valence7' or dataset == 'trec2011':
            object_index.append(int(line[0]))
        else:
            object_index.append(int(line[0])-1)

        truth_index.append(int(line[1]))
    object_index = np.array(object_index, dtype=np.int32)
    mask = np.argsort(object_index)
    label = np.array(truth_index, dtype=np.int32)[mask]
    num_class = label.max() + 1
    label = np.eye(num_class)[label]
    test_index = np.sort(object_index)
    print(test_index)

    return test_index, np.array(label, dtype=np.float32)

def get_graph(dataset=None,
              object_feature_path=None,
              answer_path=None,
              workers_weight_path=None,
              truth_path=None):
    test_index, y = get_label(dataset=dataset, object_label_path=truth_path)
    edge_index, edge_label = get_edge(dataset=dataset, answer_path=answer_path)
    x = get_feature(dataset=dataset, object_feature_path=object_feature_path, answer_path=answer_path, source_confidence_path=workers_weight_path)
    graph = Graph(x=x, edge_index=edge_index, y=y)
    return graph, edge_label, test_index

def load_data(dataset=None,
              object_feature_path=None,
              answer_path=None,
              workers_weight_path=None,
              truth_path=None):

    graph, edge_label, test_index = get_graph(dataset=dataset,
                                              object_feature_path=object_feature_path,
                                  answer_path=answer_path,
                                  workers_weight_path=workers_weight_path,
                                  truth_path=truth_path)

    return graph, edge_label, test_index
