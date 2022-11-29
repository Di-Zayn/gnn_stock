from torch.nn.functional import pad
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, \
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score, \
    explained_variance_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def str2onehot_gen(data: np.array):
    label_encoder = LabelEncoder()
    label_encoder.fit(data)
    integer_encoded = label_encoder.transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    rt_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(rt_encoded)
    onehot_encoded = onehot_encoder.transform(rt_encoded)

    return label_encoder, onehot_encoder


def str2onehot_pro(data: str, l_enc: LabelEncoder, o_enc: OneHotEncoder):
    int_c = l_enc.transform([data])
    int_cc = int_c.reshape(len(int_c), 1)
    return int(int_c[0]), o_enc.transform(int_cc)


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """
    # 将输入y向量转换为数组
    y = np.array(y, dtype='int')
    # 获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # y变为1维数组
    y = y.ravel()
    # 计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    # 生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    # np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    # 进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def regression_metric(pred_list, true_list):
    """
    计算MSE、MAE、MAPE、MSLE、R2；RAE、RSE、、
    :param pred_list:
    :param true_list:
    :return:
    """
    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    # mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score
    mse = mean_squared_error(true_list, pred_list)
    mae = mean_absolute_error(true_list, pred_list)
    msle = mean_squared_log_error(true_list, pred_list)
    r2 = r2_score(true_list, pred_list)
    evs = explained_variance_score(true_list, pred_list)

    mape = mean_absolute_percentage_error(true_list + (pred_list == 0), pred_list + (pred_list == 0))
    mre = np.abs(true_list-pred_list).sum() / np.abs(true_list).sum()

    print(f'Log Info - Dev Count: {len(true_list)}')
    print(f'Log Info - MSE: {mse}, RMSE: {mse**0.5}, MAE: {mae}, MSLE: {msle}')
    print(f'Log Info - MAPE: {mape}, MRE: {mre}, R2-Score: {r2}, Explain Variance: {evs}')
    return r2, mae, mape,


def precision_recall_fscore(pred_list, true_list):
    """
    计算召回率、F值
    """
    # None(default), 'binary', 'micro', 'macro', 'samples', \
    # 'weighted'

    weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(true_list, pred_list, average='weighted')
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(true_list, pred_list, average='micro')
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(true_list, pred_list, average='macro')
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(true_list, pred_list, average='binary')
    acc = accuracy_score(true_list, pred_list)
    # samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(true_list, pred_list, average='samples')

    print(f'Log Info - Dev Count: {len(true_list)}')
    print(f'Log Info - Micro Metric: (Precession: {micro_p}, Recall: {micro_r}, F1-Score: {micro_f1})')
    print(f'Log Info - Macro Metric: (Precession: {macro_p}, Recall: {macro_r}, F1-Score: {macro_f1})')
    print(f'Log Info - Binary Metric: (Precession: {bin_p}, Recall: {bin_r}, F1-Score: {bin_f1})')
    print(f'Log Info - Weighted Metric: (Precession: {weight_p}, Recall: {weight_r}, F1-Score: {weight_f1})')
    print(f'Log Info - Accuracy: {acc}')
    # print(f'Log Info - Samples Metric: (Precession: {samples_p}, Recall: {samples_r}, F1-Score: {samples_f1})')

    eval_results = {
        "micro": [micro_p, micro_r, micro_f1],
        "macro": [macro_p, macro_r, macro_f1],
        "weighted": [weight_p, weight_r, weight_f1],
        # "samples": [samples_p, samples_r, samples_f1]
    }

    return eval_results
