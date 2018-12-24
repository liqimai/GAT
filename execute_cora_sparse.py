import time
import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import argparse

from models import GAT
from models import SpGAT
from utils import process
from sys import stderr

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--dataset", type=str, default='cora')
parser.add_argument("--train_size", type=str, default='"public"')
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--validate", action="store_true")
args = parser.parse_args()
print(args)

seed = int(time.time())
np.random.seed(seed)

dataset = args.dataset
train_size = eval(args.train_size)
checkpt_file = 'pre_trained/{}/{}_{}.ckpt'.format(dataset, train_size, dataset)


# training params
batch_size = 1
nb_epochs = 400
patience = 100
lr = 0.05  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 8] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
# model = GAT
model = SpGAT
validation_size = 500

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

def train():
    sparse = True

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset, train_size, validation_size)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    if sparse:
        biases = process.preprocess_adj_bias(adj)
    else:
        adj = adj.todense()
        adj = adj[np.newaxis]
        biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            if sparse:
                #bias_idx = tf.placeholder(tf.int64)
                #bias_val = tf.placeholder(tf.float32)
                #bias_shape = tf.placeholder(tf.int64)
                bias_in = tf.sparse_placeholder(dtype=tf.float32)
            else:
                bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        train_op = model.training(loss, lr, l2_coef)

        saver = tf.train.Saver()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init_op)

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]

                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: bbias,
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                if args.validate:
                    vl_step = 0
                    vl_size = features.shape[0]

                    while vl_step * batch_size < vl_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[vl_step*batch_size:(vl_step+1)*batch_size]
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                            feed_dict={
                                ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                                bias_in: bbias,
                                lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                                msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                                is_train: False,
                                attn_drop: 0.0, ffd_drop: 0.0})
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1
                else:
                    tr_step = 0
                    vl_step = 0
                    vl_size = features.shape[0]
                    val_loss_avg = 0
                    val_acc_avg = 0
                    while tr_step * batch_size < tr_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[tr_step*batch_size:(tr_step+1)*batch_size]
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                            feed_dict={
                                ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                                bias_in: bbias,
                                lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                                msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                                is_train: False,
                                attn_drop: 0., ffd_drop: 0.})
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1
                        tr_step += 1

                print('%d Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                        val_loss_avg/vl_step, val_acc_avg/vl_step))


                if val_acc_avg/vl_step > vacc_mx or val_loss_avg/vl_step < vlss_mn:
                    if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        saver.save(sess, checkpt_file)

                        ts_size = features.shape[0]
                        ts_step = 0
                        ts_loss = 0.0
                        ts_acc = 0.0

                        while ts_step * batch_size < ts_size:
                            if sparse:
                                bbias = biases
                            else:
                                bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]
                            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                                             feed_dict={
                                                                 ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                 bias_in: bbias,
                                                                 lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                 msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                 is_train: False,
                                                                 attn_drop: 0.0, ffd_drop: 0.0})
                            ts_loss += loss_value_ts
                            ts_acc += acc_ts
                            ts_step += 1

                        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

            saver.restore(sess, checkpt_file)

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                if sparse:
                    bbias = biases
                else:
                    bbias = biases[ts_step*batch_size:(ts_step+1)*batch_size]
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: bbias,
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

            print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

            sess.close()
            return ts_loss/ts_step, ts_acc/ts_step

if __name__ == '__main__':
    result = []
    for _ in range(args.repeat):
        result.append(train())
    result = np.array(result)
    print(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)
    print(args)
    print(mean, std)

    print(args, file=stderr)
    print(mean, std, file=stderr)
