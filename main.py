import os
import argparse
from network import *
from ops import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_step', default=10000, help='# of step for training')
    parser.add_argument('--learning_rate', default=10)
    parser.add_argument('--class_num', default=7)
    parser.add_argument('--modeldir', default='./model')
    parser.add_argument('--model', default='LGCN', help='model name')
    parser.add_argument('--ch_num', default=8, help='channel number')
    parser.add_argument('--layer_num', default=2, help='number of graph convolution layer')
    parser.add_argument('--adj_keep_r', default=0.999, help='adjacency dropout keep rate')
    parser.add_argument('--keep_r', default=0.16, help='feature dropout keep rate')
    parser.add_argument('--weight_decay', default=5e-4, help='max sequence length')
    parser.add_argument('--k', default=8, help='top k')
    parser.add_argument('--batch_size', default=2500)
    parser.add_argument('--center_num', default=1500, help='start center number')
    parser.add_argument('--embed_size', default=128)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    config = configure()
    adj, feas, normed_adj, ys, masks = process_data('cora')
    train_y, val_y, test_y = ys
    train_mask, val_mask, test_mask = masks

    sub_adj, sub_normed_adj, sub_feas, sub_train_mask, sub_train_y = get_subgraph('train',
                                                adj, normed_adj, feas, masks, ys, config.batch_size)
    print(sub_normed_adj.shape, sub_feas.shape, sub_train_y.shape)
    sub_val_adj, sub_val_normed_adj, sub_val_feas, sub_val_mask, sub_val_y = get_subgraph('valid',
                                                adj, normed_adj, feas, masks, ys, None)
    print(sub_val_normed_adj.shape, sub_val_feas.shape, sub_val_y.shape)
    sub_test_adj, sub_test_normed_adj, sub_test_feas, sub_test_mask, sub_test_y = get_subgraph('test',
                                                adj, normed_adj, feas, masks, ys, None)
    normed_matrix, inputs = build_network(sub_feas.shape[-1])

    output = inference(normed_matrix, inputs, config.ch_num, config.adj_keep_r,
                        config.keep_r, config.layer_num, config.k, config.embed_size, config.class_num)

    model = Model(inputs=[normed_matrix, inputs], outputs=output)
    masked_categorical_crossentropy = get_loss(sub_train_mask, model, config.weight_decay)
    masked_accuracy = get_accuracy(sub_train_mask)
    early_stopping = EarlyStopping(patience=10)
    # plot_model(model, to_file='model.png')
    # print(model.trainable_weights)
    model.summary()
    tensorboard = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    model.compile(loss=masked_categorical_crossentropy, optimizer='nadam', metrics=[masked_accuracy])
    model.fit(x={'sub_adj' : sub_normed_adj, 'sub_feas' : sub_feas}, y=sub_train_y,
                batch_size=sub_adj.shape[0], epochs=config.max_step, callbacks=[early_stopping],
                validation_data=({'sub_adj' : sub_val_normed_adj, 'sub_feas' : sub_val_feas}, sub_val_y))

    model.evaluate(x={'sub_adj' : sub_test_normed_adj, 'sub_feas' : sub_test_feas}, y=sub_test_y)
