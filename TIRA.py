# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from data.process_tira import load_data
from sklearn.metrics import f1_score, classification_report
import numpy as np
import time

# dataset={1:'duck', 2:'WS', 3:'SP', 4:'dog', 5:'product', 6:'UC', 7:'valence7', 8:'aircrowd6', 9:'valence5', 10:'fej2013', 11:'cf', 12'trec2011'}
dataset = 'aircrowd6'
graph, edge_label, test_index = load_data(dataset=dataset,
                                          object_feature_path='./data/zzfx/{}/feature.txt'.format(dataset),
                                          answer_path='./data/zzfx/{}/answer.csv'.format(dataset),
                                          workers_weight_path='./data/zzfx/{}/workers_weight.txt'.format(dataset),
                                          truth_path='./data/zzfx/{}/truth.csv'.format(dataset))
lr = 1e-3
num_classes = edge_label[2,:].max() + 1
node_num = int(graph.x.shape[0])
obj_num = int(max(graph.edge_index[0,:])+1)
class TIRA(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn10 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.gcn11 = tfg.layers.GCN(128, activation=None)
        self.gcn12 = tfg.layers.GCN(128, activation=tf.nn.relu)
        self.w1 = tf.Variable(tf.random.truncated_normal([128, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([node_num, 128], stddev=0.1))
        self.fc0 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.fc1 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h1 = self.gcn10([x, edge_index], training=training)

        h11 = tf.nn.relu(tf.matmul(h1, self.w1) + self.b1)
        h2 = self.gcn11([h11, edge_index], training=training)
        h2 = self.gcn12([h2, edge_index], training=training)

        h = h1 + h2

        h = self.fc0(h)
        h = self.fc1(h)
        return h

# x = tf.Variable(tf.random.truncated_normal([node_num, num_classes], stddev=0.1))
def forward(graph, training=False):
    return model([graph.x, graph.edge_index], training=training)
    # return model([x, graph.edge_index], training=training)

#@tf_utils.function
def predict_edge(embedded, edge_label):
    row, col = edge_label[0], edge_label[1]
    embedded_row = tf.gather(embedded, row)
    embedded_col = tf.gather(embedded, col)

    # dot product
    logits = embedded_row * embedded_col
    return logits

def compute_loss(logits, vars):
    masked_logits = logits
    masked_labels = tf.one_hot(edge_label[-1,:],depth=num_classes)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=masked_labels
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4

def evaluate(mask):
    logits = tf.nn.softmax(forward(graph))
    masked_logits = tf.gather(logits, mask)

    masked_labels = tf.argmax(graph.y, axis=-1, output_type=tf.int32)
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)

    return y_pred, accuracy_m.result().numpy(), f1_score(y_true=masked_labels, y_pred=y_pred, average='weighted')

model =TIRA()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
# ##load checkpoint
checkpoint.restore(tf.train.latest_checkpoint('./model/{}-1'.format(dataset)))

best_test_acc  = 0
best_weight = 0
best_macro = 0
y_pred_best = []
# start_time = time.time()
for step in range(1, 301):
    with tf.GradientTape() as tape:
        embedded = forward(graph, training=True)
        logits = predict_edge(embedded, edge_label[:2,:])
        loss = compute_loss(logits, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    y_pred, test_acc, weight = evaluate(test_index)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_weight = weight

    print("step = {}\tloss = {}\tacc = {}\tweight = {}".format(step, loss, best_test_acc, best_weight))
