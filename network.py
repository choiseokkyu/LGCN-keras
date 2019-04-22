import os
import time
import random
import numpy as np
import tensorflow as tf
import ops
from utils import load_data, preprocess_features, preprocess_adj
from batch_utils import get_sampled_index, get_indice_graph
from tensorflow.keras.layers import Input, Concatenate


def process_data(dataname):
    data = load_data(dataname)
    adj, feas = data[:2]
    adj = adj.todense()
    normed_adj = preprocess_adj(adj)
    feas = preprocess_features(feas, False)
    ys = data[2:5]
    masks = data[5:]

    return adj, feas, normed_adj, ys, masks


def inference(normed_matrix, outs, ch_num, adj_keep_r, keep_r, layer_num, k, embed_size, class_num):
    outs = getattr(ops, 'simple_conv')(
        normed_matrix, outs, 4*ch_num, adj_keep_r, keep_r, 'conv_s')
    for layer_index in range(layer_num):
        cur_outs = getattr(ops, 'graph_conv')(
            normed_matrix, outs, ch_num, adj_keep_r, keep_r, 'conv_%s' % (layer_index+1), k=k)
        outs = Concatenate(axis=1, name='concat_%s' % (layer_index+1))([outs, cur_outs])
    outs = ops.simple_conv(normed_matrix, outs, embed_size, adj_keep_r, keep_r, 'conv_f1')
    outs = ops.simple_conv(normed_matrix, outs, class_num, adj_keep_r, keep_r, 'conv_f2')
    return outs

def build_network(fea_shape):
    normed_matrix = Input(shape=[None,], name='sub_adj')
    inputs = Input(shape=[fea_shape,], name='sub_feas')
    return normed_matrix, inputs


def get_subgraph(action, adj, normed_adj, feas, masks, ys, batch_size):
    if action == 'train':
        indices = get_indice_graph(adj, masks[0], batch_size, 1.0)
        new_adj = adj[indices, :][:, indices]
        new_normed_adj = normed_adj[indices, :][:, indices]
        new_feas = feas[indices]
        new_mask = masks[0][indices]
        new_y = ys[0][indices]
    if action == 'valid':
        indices = get_indice_graph(adj, masks[1], 10000, 1.0)
        new_adj = adj[indices, :][:, indices]
        new_normed_adj = normed_adj[indices, :][:, indices]
        new_feas = feas[indices]
        new_mask = masks[1][indices]
        new_y = ys[1][indices]
    else:
        indices = get_indice_graph(adj, masks[2], 10000, 1.0)
        new_adj = adj[indices, :][:, indices]
        new_normed_adj = normed_adj[indices, :][:, indices]
        new_feas = feas[indices]
        new_mask = masks[2][indices]
        new_y = ys[2][indices]
    return new_adj, new_normed_adj, new_feas, new_mask, new_y
