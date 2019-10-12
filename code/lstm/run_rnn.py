import numpy as np
import math
import mxnet as mx
import mxnet.ndarray as nd
from sklearn import metrics
import os


def norm_clipping(params_grad, threshold):
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        norm_val += np.sqrt(
            sum([nd.norm(grads[i]).asnumpy()[0] ** 2
                 for grads in params_grad]))
    norm_val /= float(len(params_grad[0]))

    if norm_val > threshold:
        ratio = threshold / float(norm_val)
        for grads in params_grad:
            for grad in grads:
                grad[:] *= ratio

    return norm_val


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log( np.maximum(1e-10,pred)) + \
           (1.0 - target) * np.log( np.maximum(1e-10, 1.0-pred) )
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, q_data, qa_data, label):
    N = int(math.floor(len(qa_data) / params.batch_size))
    q_data = q_data.T
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(qa_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        if params.show: bar.next()

        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)
        target = mx.nd.array(target)

        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)

        # 需要把target和 input_qa 错一位
        target = target[1:]
        input_q = input_q[1:]
        input_qa = input_qa[:-1]

        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[target])
        net.forward(data_batch, is_train=True)
        pred = net.get_outputs()[0].asnumpy() #(seqlen * batch_size, 1)
        net.backward()

        norm_clipping(net._exec_group.grad_arrays, params.maxgradnorm)
        net.update()

        target = target.asnumpy().reshape((-1,)) # correct: 1.0; wrong 0.0; padding -1.0

        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()

    all_pred = np.concatenate(pred_list,axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    print("all_target", all_target)
    print("all_pred", all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params, q_data, qa_data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    seq_num = qa_data.shape[1]
    pred_list = []
    target_list = []
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    count = 0
    element_count = 0

    if os.path.exists('./analyse/analyse.csv'):
        os.remove('./analyse/analyse.csv')

    for idx in range(N):
        if params.show: bar.next()

        inds = np.arange(idx * params.batch_size, (idx + 1) * params.batch_size)
        q_one_seq = q_data.take(inds, axis=1, mode='wrap')
        qa_one_seq = qa_data.take(inds, axis=1, mode='wrap')

        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)
        target = qa_one_seq[:, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)
        target = mx.nd.array(target)
        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)

        # 需要把target和 input_qa 错一位
        target = target[1:]
        input_qa = input_qa[:-1]
        input_q = input_q[1:] # 取target的标签

        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[])
        net.forward(data_batch, is_train=False)
        pred = net.get_outputs()[0].asnumpy()
        target = target.asnumpy()

        # todo
        analyse_pred = pred.reshape((params.seqlen, params.batch_size))
        analyse_pred = analyse_pred.T
        analyse_target = target.T
        analyse_input_q = input_q.asnumpy().T

        # save_res(analyse_input_q, analyse_target, analyse_pred)

        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            target = target[:, :real_batch_size]
            pred = pred.reshape((params.seqlen, params.batch_size))[:, :real_batch_size]
            pred = pred.reshape((-1,))
            count += real_batch_size
        else:
            count += params.batch_size

        target = target.reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0
        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        #print avg_loss

        # todo: 保存预测结果用于分析
        #

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()
    assert count == seq_num

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


"""
生成热力图
"""


def get_mastery(net, params, input_q, input_qa):
    """
    仅传入一个样本，用户画热力图
    :param net:
    :param params:
    :param q_data: (1, seqlen)
    :param qa_data:(1, seqlen)
    :param label:
    :return:
    """

    unique_questions = list(set(input_q[0]))
    unique_questions.remove(0)

    input_q = input_q.T # Shape (seqlen+1, 1)
    input_qa = input_qa.T # Shape (seqlen+1, 1)
    target = input_qa[:, :]
    target = (target - 1) / params.n_question
    target = np.floor(target)


    # 需要把target和 input_qa 错一位
    target = target[1:]
    input_qa = input_qa[:-1]
    input_q = input_q[1:]

    # 给定前i-1个做题历史记录，预测第i次，在所有题目上的答题概率
    predict_matrix = np.zeros((len(unique_questions), len(input_q)))
    for i in range(len(input_q)):
        q = np.zeros((len(input_q), 1))
        qa = np.zeros((len(input_qa), 1))
        t = np.zeros((len(target), 1))
        qa[:i+1] = input_qa[:i+1]
        t[:i+1] = target[:i+1]
        q[:i] = input_q[:i]

        q = mx.nd.array(q)
        qa = mx.nd.array(qa)
        t = mx.nd.array(t)
        # 每一步预测unique_questions个概率
        for j in range(len(unique_questions)):
            question = unique_questions[j]
            q[i] = question
            data_batch = mx.io.DataBatch(data=[q, qa], label=[])
            net.forward(data_batch, is_train=False)
            pred = net.get_outputs()[0].asnumpy()
            predict_matrix[j][i] = pred[i]

    return predict_matrix, unique_questions


def save_res(inputs, targets, predicts):
    """

    :param inputs: (batch_size * seqlen)
    :param targets: (batch_size * seqlen)
    :param predicts: (batch_size * seqlen)
    :return:
    """
    with open('./analyse/analyse.csv', 'a') as f:
        for i in range(len(inputs)):
            f.write('question,' + ','.join([str(j) for j in inputs[i]]) + '\n')
            f.write('target,' + ','.join([str(j) for j in targets[i]]) + '\n')
            f.write('predict,' + ','.join([str(j) for j in predicts[i]]) + '\n')
    return


def construct_matrix(questions, predicts, targets):
    """
    形成一个n_question * seq_len长度的矩阵，用户画heatMap
    :param questions:
    :param predicts:
    :param targets:
    :return:
    """
    unique_questions = list(set(questions))
    matrix = np.zeros((len(unique_questions), len(questions)))
    for i in range(len(questions)):
        question = questions[i]
        predict = predicts[i]
        question_idx = unique_questions.index(question)
        matrix[question_idx][i] = predict
    return matrix


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


def show(matrix):
    fig = plt.figure()
    ax = sns.heatmap(matrix)
    plt.show()
    return


