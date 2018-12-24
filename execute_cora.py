import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process

seed = int(time.time())
np.random.seed(seed)

dataset = 'cora'
checkpt_file = 'pre_trained/{}/mod_{}.ckpt'.format(dataset, dataset)


# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.02  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT
train_size = 20
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
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset, train_size, validation_size)
    features, spars = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    adj = adj.todense()

    features = features[np.newaxis]
    adj = adj[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size), name='ftr_in')
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes), name='bias_in')
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes), name='lbl_in')
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes), name='msk_in')
            attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
            is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')

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

        with tf.Session() as sess:
            sess.run(init_op)

            for epoch in range(nb_epochs):

                tr_step = 0
                tr_size = features.shape[0]
                train_loss_avg = 0
                train_acc_avg = 0
                while tr_step * batch_size < tr_size:
                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]
                val_loss_avg = 0
                val_acc_avg = 0
                while vl_step * batch_size < vl_size:
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                            lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                            msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                # tr_step = 0
                # vl_step = 0
                # vl_size = features.shape[0]
                # val_loss_avg = 0
                # val_acc_avg = 0
                # while tr_step * batch_size < tr_size:
                #     loss_value_vl, acc_vl = sess.run([loss, accuracy],
                #         feed_dict={
                #             ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                #             bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                #             lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                #             msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                #             is_train: False,
                #             attn_drop: 0., ffd_drop: 0.})
                #     val_loss_avg += loss_value_vl
                #     val_acc_avg += acc_vl
                #     vl_step += 1
                #     tr_step += 1

                print('%d Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                        val_loss_avg/vl_step, val_acc_avg/vl_step))

                # tr_step = 0
                # train_loss_avg = 0
                # train_acc_avg = 0
                # while tr_step * batch_size < tr_size:
                #     loss_value_tr, acc_tr = sess.run([loss, accuracy],
                #         feed_dict={
                #             ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                #             bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                #             lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                #             msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                #             is_train: False,
                #             attn_drop: 0., ffd_drop: 0.})
                #     val_loss_avg += loss_value_tr
                #     val_acc_avg += acc_tr
                #     vl_step += 1
                # print('%.5f %.5f %d' % (train_loss_avg, train_acc_avg, tr_step))


                if val_acc_avg/vl_step > vacc_mx or val_loss_avg/vl_step < vlss_mn:
                    if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        saver.save(sess, checkpt_file)
                        print('saveing model...')
                        ts_size = features.shape[0]
                        ts_step = 0
                        ts_loss = 0.0
                        ts_acc = 0.0
                        while ts_step * batch_size < ts_size:
                            loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                         feed_dict={
                                             ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             bias_in: biases[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                             is_train: False,
                                             attn_drop: 0.0, ffd_drop: 0.0})
                            ts_loss += loss_value_ts
                            ts_acc += acc_ts
                            ts_step += 1
                        print('[TEST]Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                # train_loss_avg = 0
                # train_acc_avg = 0
                # val_loss_avg = 0
                # val_acc_avg = 0

            saver.restore(sess, checkpt_file)

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                loss_value_ts, acc_ts = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
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
    for _ in range(1):
        result.append(train())
    result = np.array(result)
    print(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)
    print(mean, std)
