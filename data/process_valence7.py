import csv
import numpy as np
from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge
from tf_geometric.data.graph import Graph


def get_feature(graph_dict=None, object_feature_path=None, source_classes_path=None, source_weight_path=None):

    object_features = np.loadtxt(object_feature_path)[:,:7]
    source_classes = 7
    source_weights = np.loadtxt(source_weight_path)[:,-1]

    source_num = source_weights.shape[0]##38
    source_classes_num = int(source_classes[:,-1].max()) + 1
    source_features = np.zeros(shape=(source_num, source_classes_num))##(38, 7)

    for i in range(source_classes.shape[0]):
        source_id = int(source_classes[i][0])
        source_class_id = int(source_classes[i][2])
        source_weight = source_weights[source_id]
        source_features[source_id][source_class_id] = source_weight

    # object_num = int(len(graph_dict))##100
    # object_features = np.zeros(shape=(object_num, source_classes_num))##(100, 7)
    # for i in range(object_num):
    #     mask = np.array(graph_dict[i]) - object_num
    #     mask_feature = source_features[mask]
    #     object_features[i] = np.sum(mask_feature, axis=0)
    # print(object_features)

    node_features = np.vstack([object_features, source_features])

    return node_features

def get_label(object_num, object_label_path, source_label_path):
    object_classes = csv.reader(open(object_label_path, 'r'))
    source_classes = np.loadtxt(source_label_path)

    source_classes_num = int(source_classes[:, -1].max()) + 1 #7
    object_label = np.zeros(shape=(object_num, source_classes_num)) #(100, 7)
    for i in object_classes:
        object_label[int(i[0])][int(i[1])] = 1

    return np.array(object_label, dtype=np.float32)

def get_edge(edge_label_path):
    f_edge_label = np.loadtxt(edge_label_path)[:,[1,0,2]] # v,u,class
    f_edge_label[:,1] += 100
    edge_index = f_edge_label[:,:2].T
    edge_label = np.array(np.vstack([edge_index, f_edge_label[:,-1]]),dtype=np.int32) #[u1,u2,u3,...],[v1,v2,v3,...],[l1,l2,l3,...]]

    graph_dict = {}

    for i in range(edge_index.shape[1]):
        u = edge_index[0][i]
        v = edge_index[1][i]
        if u in graph_dict:
            graph_dict[u].append(v)
        else:
            graph_dict.update({u: [v]})

    edge_index, _ = remove_self_loop_edge(edge_index)
    edge_index, _ = convert_edge_to_directed(edge_index)
    return graph_dict, edge_index, edge_label

def get_graph(object_num,
              object_label_path,
              source_label_path,
              object_feature_path,
              source_classes_path,
              source_weight_path):
    y = get_label(object_num=object_num, object_label_path=object_label_path, source_label_path=source_label_path)

    graph_dict, edge_index, edge_label = get_edge(edge_label_path=source_classes_path)

    x = get_feature(graph_dict=graph_dict,
                    object_feature_path=object_feature_path,
                    source_classes_path=source_classes_path,
                    source_weight_path=source_weight_path)

    # inv_sum_x = 1.0 / np.sum(x, axis=-1, keepdims=True)
    # inv_sum_x[np.isnan(inv_sum_x)] = 1.0
    # inv_sum_x[np.isinf(inv_sum_x)] = 1.0
    # x *= inv_sum_x#点乘，归一化

    graph = Graph(x=x, edge_index=edge_index, y=y)
    return graph, graph_dict, edge_label

def load_data(object_num,
              source_num,
              object_label_path,
              object_feature_path,
              source_classes_path,
              source_weight_path):

    total_index = list(range(object_num+source_num))
    # shuffle(total_index)
    train_index = total_index[object_num:]
    test_index = total_index[:object_num]
    graph, graph_dict, edge_label = get_graph(object_num=object_num,
                                              object_label_path=object_label_path,
                                              source_label_path=source_classes_path,
                                              object_feature_path=object_feature_path,
                                              source_classes_path=source_classes_path,
                                              source_weight_path=source_weight_path)
    return graph, graph_dict, edge_label, train_index, test_index

# graph, graph_dict, train_index, test_index = load_data(object_num=100,
#                                                        source_num=38,
#                                                        edge_path="./zzfx/valence7/answer.csv",
#                                                        object_label_path="./zzfx/valence7/question_truth.csv",
#                                                        object_feature_path = "./zzfx/valence7/question_feature.txt",
#                                                        source_classes_path="./zzfx/valence7/worker_classes.txt",
#                                                        source_weight_path="./zzfx/valence7/workers_weight.txt")
# print(graph_dict)
# print(train_index)
# print (test_index)
# print(graph.edge_index)
# print(graph.y[:100,:])
