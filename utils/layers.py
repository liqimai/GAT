import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


def d_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        attn = []
        for _ in range(out_sz):
            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)

            f_1 = tf.reshape(f_1, (nb_nodes, 1))
            f_2 = tf.reshape(f_2, (nb_nodes, 1))

            f_1 = adj_mat * f_1
            f_2 = adj_mat * tf.transpose(f_2, [1, 0])

            logits = tf.sparse_add(f_1, f_2)
            lrelu = tf.SparseTensor(indices=logits.indices,
                                    values=tf.nn.leaky_relu(logits.values),
                                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)

            if coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                                        values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                        dense_shape=coefs.dense_shape)
            attn.append(tf.sparse_reshape(coefs, (nb_nodes, nb_nodes, 1)))

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # apply attention
        # coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        # seq_fts = tf.squeeze(seq_fts)
        # vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        attn = tf.sparse_concat(2, attn)
        seq_fts = tf.reshape(seq_fts, (nb_nodes, 1, out_sz))
        vals = attn*seq_fts
        vals = tf.sparse_reduce_sum(vals, axis=1)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation


# # dimensional attention head
# def d_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
#     with tf.name_scope('sp_attn'):
#         if in_drop != 0.0:
#             seq = tf.nn.dropout(seq, 1.0 - in_drop)
#
#         seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
#
#         # simplest self-attention possible
#         f_1 = tf.layers.conv1d(seq_fts, out_sz, 1)
#         f_2 = tf.layers.conv1d(seq_fts, out_sz, 1)
#
#         f_1 = tf.reshape(f_1, (1, nb_nodes, out_sz))
#         f_2 = tf.reshape(f_2, (nb_nodes, 1, out_sz))
#         adj_mat = tf.sparse_reshape(adj_mat, (nb_nodes, nb_nodes, 1))
#
#         logits = adj_mat * f_1 * f_2
#         # logits = adj_mat
#
#         lrelu = tf.SparseTensor(indices=logits.indices,
#                                 values=tf.exp(tf.nn.leaky_relu(logits.values)),
#                                 dense_shape=logits.dense_shape)
#         s = tf.sparse_reduce_sum(lrelu, axis=1, keep_dims=True)
#         coefs = lrelu/s
#
#         # drop out
#         if coef_drop != 0.0:
#             coefs = tf.SparseTensor(indices=coefs.indices,
#                                     values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
#                                     dense_shape=coefs.dense_shape)
#         if in_drop != 0.0:
#             seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
#
#         # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
#         # here we make an assumption that our input is of batch size 1, and reshape appropriately.
#         # The method will fail in all other cases!
#         # coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
#         seq_fts = tf.reshape(seq_fts, (nb_nodes, 1, out_sz))
#         vals = coefs*seq_fts
#         vals = tf.sparse_reduce_sum(vals, axis=1)
#         # vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
#         vals = tf.expand_dims(vals, axis=0)
#         vals.set_shape([1, nb_nodes, out_sz])
#         ret = tf.contrib.layers.bias_add(vals)
#
#         # residual connection
#         if residual:
#             if seq.shape[-1] != ret.shape[-1]:
#                 ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
#             else:
#                 seq_fts = ret + seq
#
#         return activation(ret)  # activation