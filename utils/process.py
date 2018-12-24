import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.io as sio
import sys

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, train_size, validation_size, test_size=None, shuffle=True):

    if dataset_str in ['cora_coauthor']:
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0],np.max(l)+1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']

        # if model_config['mu']:
        #     pca = decomposition.TruncatedSVD(n_components=1000)
        #     pca.fit(features)
        #     features = pca.transform(features)
        # adj = (1-mu)*data['G'] + mu*data['C']
        adj = data['G']
    else:
        if train_size == 'public':
            return load_public_split_data(dataset_str)
        """Load data."""
        global all_labels
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        # if dataset_str == 'citeseer':
        #     # Fix citeseer dataset (there are some isolated nodes in the graph)
        #     # Find isolated nodes, add them as zero-vecs into the right position
        #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
        #     tx = tx_extended
        #     ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
        #     ty_extended[test_idx_range - min(test_idx_range), :] = ty
        #     ty = ty_extended

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        # if dataset_str == 'citeseer':
        #     # Fix citeseer dataset (there are some isolated nodes in the graph)
        #     # Find isolated nodes, add them as zero-vecs into the right position
        #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
        #     tx = tx_extended
        #     ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1] - 1))
        #     ty_extended_ = np.ones((len(test_idx_range_full), 1))  # add dummy labels
        #     ty_extended = np.hstack([ty_extended, ty_extended_])
        #     ty_extended[test_idx_range - min(test_idx_range), :] = ty
        #     ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        # features = sp.eye(features.shape[0]).tolil()
        # features = sp.lil_matrix(allx)

        labels = np.vstack((ally, ty))
        # labels = np.vstack(ally)

        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # features = preprocess_features(features, feature_type=model_config['feature'])

    all_labels = labels.copy()

    # split the data set
    idx = np.arange(len(labels))
    if shuffle:
        np.random.shuffle(idx)
    if isinstance(train_size, int):
        assert train_size>0, "train size must bigger than 0."
        no_class = labels.shape[1]  # number of class
        train_size = [train_size for i in range(labels.shape[1])]
        idx_train = []
        count = [0 for i in range(no_class)]
        label_each_class = train_size
        next = 0
        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(no_class):
                if labels[i, j] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1
                    break

        if True: #model_config['validate']:
            if validation_size:
                assert next+validation_size<len(idx)
                next = next+validation_size
                assert next < len(idx), "Too many train data, no data left for validation."
                idx_val = idx[next:next+validation_size]
            else:
                idx_val =  idx[next:]


            if test_size:
                assert next+test_size<len(idx)
            assert next < len(idx), "Too many train and validation data, no data left for testing."
            idx_test = idx[-test_size:] if test_size else idx[next:]
        else:
            if test_size:
                assert next+test_size<len(idx)
            assert next < len(idx), "Too many train data, no data left for testing."
            idx_val = idx[-test_size:] if test_size else idx[next:]
            idx_test = idx[-test_size:] if test_size else idx[next:]
    else:
        # train
        assert isinstance(train_size, float)
        assert 0<train_size<1, "float train size must be between 0-1"
        labels_of_class = [0]
        train_size = int(len(idx) * train_size)
        next = 0
        try_time = 0
        while np.prod(labels_of_class) == 0 and try_time < 100:
            np.random.shuffle(idx)
            idx_train = idx[next:next+train_size]
            labels_of_class = np.sum(labels[idx_train], axis=0)
            try_time = try_time+1
        next = train_size

        # validate
        if True: # model_config['validate']:
            assert isinstance(validation_size, float)
            validation_size = int(len(idx) * validation_size)
            idx_val = idx[next: next+validation_size]
            next += validation_size
        else:
            idx_val = idx[next:]

        # test
        if test_size:
            assert isinstance(test_size, float)
            test_size = int(len(idx) * test_size)
            idx_test = idx[next: next+test_size]
        else:
            idx_test = idx[next:]

    print('labels of each class : ', np.sum(labels[idx_train], axis=0))
    # idx_val = idx[len(idx) * train_size // 100:len(idx) * (train_size // 2 + 50) // 100]
    # idx_test = idx[len(idx) * (train_size // 2 + 50) // 100:len(idx)]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)

    features = features.astype(np.float32)
    adj = sp.csr_matrix(adj)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_public_split_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)
    adj = sp.csr_matrix(adj)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape
