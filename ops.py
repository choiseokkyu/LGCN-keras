import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Multiply, Permute, Lambda, Concatenate, Dropout, Dense, ELU
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv1D, Add
from tensorflow.keras.metrics import categorical_accuracy


def top_k_features(adj_m, fea_m, k, scope):
    adj_expanded = K.expand_dims(adj_m, axis=1)
    fea_expanded = K.expand_dims(fea_m, axis=-1)
    feas = Multiply(name=scope+'/mul')([adj_expanded, fea_expanded])
    feas = K.permute_dimensions(feas, (2, 1, 0))
    top_k_values = Lambda(top_k, arguments={'k': 8}, name=scope+'/top_k')(feas)
    top_k_values = Concatenate(name=scope+'/concat')([fea_expanded, top_k_values])
    top_k_values = K.permute_dimensions(top_k_values, (0, 2, 1))
    return top_k_values


def top_k(input, k):
    return tf.nn.top_k(input, k=k, sorted=True).values


def simple_conv(adj_m, outs, num_out, adj_keep_r, keep_r, scope, **kw):
    adj_m = Dropout(adj_keep_r, name=scope+'/drop1')(adj_m)
    outs = Dropout(keep_r, name=scope+'/drop2')(outs)
    outs = Dense(num_out, name=scope+'/dense')(outs)
    outs = K.dot(adj_m, outs)
    # outs = BatchNormalization(name=scope+'/norm')(outs)
    # outs = ELU(name=scope+'/act')(outs)
    return outs

def graph_conv(adj_m, outs, num_out, adj_keep_r, keep_r, scope, k=5, **kw):
    num_in = outs.shape[-1]
    adj_m = Dropout(adj_keep_r, name=scope+'/drop1')(adj_m)
    outs = top_k_features(adj_m, outs, k, scope+'/top_k')
    outs = Dropout(keep_r, name=scope+'/drop2')(outs)
    outs = Conv1D((num_in+num_out)//2, (k+1)//2+1, name=scope+'/conv1')(outs)
    # outs = ReLU(max_value=6, name=scope+'/act1')(outs)
    outs = Dropout(keep_r, name=scope+'/drop3')(outs)
    outs = Conv1D(num_out, k//2+1, name=scope+'/conv2')(outs)
    outs = K.squeeze(outs, axis=1)
    outs = BatchNormalization(name=scope+'/norm')(outs)
    # outs = ReLU(max_value=6, name=scope+'/act2')(outs)
    return outs

def l2_loss(decay, x):
    return decay * sum(x ** 2) / 2


def get_loss(mask, model, decay):
    mask = K.variable(mask, dtype='float32')
    weights = model.trainable_weights
    decay = decay
    def masked_categorical_crossentropy(y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred)
        mask_value = mask / K.mean(mask)
        loss *= mask_value
        loss = K.mean(loss)
        return loss
    return masked_categorical_crossentropy


def get_accuracy(mask):
    mask = K.variable(mask, dtype='float32')
    def masked_accuracy(y_true, y_pred):
        accuracy_all = categorical_accuracy(y_true, y_pred)
        mask_value = mask / K.mean(mask)
        accuracy_all *= mask_value
        return K.mean(accuracy_all)
    return masked_accuracy
