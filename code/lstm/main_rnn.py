from log import *
import os
import argparse
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from load_data_rnn import DATA
from model_rnn import ModelRnn
from run_rnn import *

log = setlogger('./log/', 'DKVMN')


def find_file(dir_name, best_epoch):
    for dir, subdir, files in os.walk(dir_name):
        for sub in subdir:
            if sub[0:len(best_epoch)] == best_epoch and sub[len(best_epoch)] == "_":
                return sub


def load_params(prefix, epoch):
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def train_one_dataset(params, file_name, train_q_data, train_qa_data, valid_q_data, valid_qa_data):
    ### ================================== model initialization ==================================

    g_model = ModelRnn(n_question=params.n_question,
                       seqlen=params.seqlen,
                       batch_size=params.batch_size,
                       q_embed_dim=params.q_embed_dim,
                       qa_embed_dim=params.qa_embed_dim,
                       final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    net = mx.mod.Module(symbol=g_model.sym_gen(),
                        data_names=['q_data', 'qa_data'],
                        label_names=['target'],
                        context=params.ctx)


    # create memory by given input shapes
    net.bind(data_shapes=[mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
                          mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
             label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])
    # initial parameters with the default random initializer
    net.init_params(initializer=mx.init.Normal(sigma=params.init_std))
    # decay learning rate in the lr_scheduler
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=20 * (train_qa_data.shape[0] / params.batch_size), factor=0.667,
                                                   stop_factor_lr=1e-5)

    net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': params.lr, 'momentum': params.momentum,
                                                          'lr_scheduler': lr_scheduler})

    for parameters in net.get_params()[0]:
        log.info(str(parameters) + ' ' + str(net.get_params()[0][parameters].asnumpy().shape))
    log.info("\n")

    ### ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc = train(net, params, train_q_data, train_qa_data, label='Train')
        valid_loss, valid_accuracy, valid_auc = test(net, params, valid_q_data, valid_qa_data, label='Valid')

        log.info('epoch ' + str(idx + 1))
        log.info("valid_auc\t" + str(valid_auc) + "\ttrain_auc\t" + str(train_auc))
        log.info("valid_accuracy\t" + str(valid_accuracy) + "\ttrain_accuracy\t" + str(train_accuracy))
        log.info("valid_loss\t" + str(valid_loss) + "\ttrain_loss\t" + str(train_loss))

        if not os.path.isdir('model'):
            os.makedirs('model')
        if not os.path.isdir(os.path.join('model', params.save)):
            os.makedirs(os.path.join('model', params.save))
        net.save_checkpoint(prefix=os.path.join('model', params.save, file_name), epoch=idx + 1)

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = idx + 1

    if not os.path.isdir('./result'):
        os.makedirs('./result')
    if not os.path.isdir(os.path.join('./result', params.save)):
        os.makedirs(os.path.join('./result', params.save))
    f_save_log = open(os.path.join('./result', params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch):
    log.info("\n\nStart testing ......................\n Best epoch: " + str(best_epoch))
    g_model = ModelRnn(n_question=params.n_question,
                       seqlen=params.seqlen,
                       batch_size=params.batch_size,
                       q_embed_dim=params.q_embed_dim,
                       qa_embed_dim=params.qa_embed_dim,
                       final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    test_net = mx.mod.Module(symbol=g_model.sym_gen(),
                        data_names=['q_data', 'qa_data'],
                        label_names=['target'],
                        context=params.ctx)
    # create memory by given input shapes
    test_net.bind(data_shapes=[mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
                          mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
             label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])
    arg_params, aux_params = load_params(prefix=os.path.join('model', params.load, file_name),
                                         epoch=best_epoch)
    test_net.init_params(arg_params=arg_params, aux_params=aux_params,
                         allow_missing=False)
    test_loss, test_accuracy, test_auc = test(test_net, params, test_q_data, test_qa_data, label='Test')
    log.info("\ntest_auc\t" + str(test_auc))
    log.info("test_accuracy\t" + str(test_accuracy))
    log.info("test_loss\t" + str(test_loss))


def get_mastery_by_one_seq(params, file_name, q_data, qa_data, best_epoch):
    log.info("\n\nStart calculating mastery matrix giiven a seq ......................\n Best epoch: " + str(best_epoch))
    g_model = ModelRnn(n_question=params.n_question,
                       seqlen=params.seqlen,
                       batch_size=params.batch_size,
                       q_embed_dim=params.q_embed_dim,
                       qa_embed_dim=params.qa_embed_dim,
                       final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    test_net = mx.mod.Module(symbol=g_model.sym_gen(),
                             data_names=['q_data', 'qa_data'],
                             label_names=['target'],
                             context=params.ctx)
    # create memory by given input shapes
    test_net.bind(data_shapes=[mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
                               mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
                  label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])
    arg_params, aux_params = load_params(prefix=os.path.join('model', params.load, file_name),
                                         epoch=best_epoch)
    test_net.init_params(arg_params=arg_params, aux_params=aux_params,
                         allow_missing=False)
    predict_matrix, unique_questions = get_mastery(test_net, params, q_data, qa_data)
    return predict_matrix, unique_questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test KVMN.')
    parser.add_argument('--gpus', type=str, default='-1', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=50, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--dataset', type=str, default='assist2009_updated', help='dataset')
    parser.add_argument('--heatmap', type=bool, default=False, help='use single sequence data to draw heatmap')
    dataset = parser.parse_args().dataset  # synthetic / assist2009_updated / assist2015 / KDDal0506 / STATICS
    # dataset = "assist2009_updated"  # synthetic / assist2009_updated / assist2015 / KDDal0506 / STATICS

    if dataset == "synthetic":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=10, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=10, help='answer and question embedding dimensions')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=50, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/synthetic', help='data directory')
        parser.add_argument('--data_name', type=str, default='naive_c5_q50_s4000_v1', help='data set name')
        parser.add_argument('--load', type=str, default='synthetic/v1', help='model file to load')
        parser.add_argument('--save', type=str, default='synthetic/v1', help='path to save model')

    if dataset == "assist2009_updated":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/assist2009_updated', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
        parser.add_argument('--load', type=str, default='assist2009_updated', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')

    elif dataset == "assist2015":
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/assist2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')
        parser.add_argument('--load', type=str, default='assist2015', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2015', help='path to save model')
    elif dataset == "STATICS":
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=1223,
                            help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/STATICS', help='data directory')
        parser.add_argument('--data_name', type=str, default='STATICS', help='data set name')
        parser.add_argument('--load', type=str, default='STATICS', help='model file to load')
        parser.add_argument('--save', type=str, default='STATICS', help='path to save model')

    params = parser.parse_args()
    params.lr = params.init_lr
    params.dataset = dataset
    if params.gpus == '-1':
        ctx = mx.cpu()
        log.info("Training with cpu ...")
    else:
        ctx = mx.gpu(int(params.gpus))
        log.info("Training with gpu(" + str(params.gpus) + ") ...")
    params.ctx = ctx

    # Read data
    dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    seedNum = 224
    np.random.seed(seedNum)

    # 画heatmap图
    if params.heatmap:
        # q = [2,2,2,2,2,2,2,2,3,3,3,3,3,4,5,4,5,4,5,4,5,6,7,6,8,4,8,4,8,4,8,4,8,4,8,6,9,6,9,6,8,6,7,11,11,28,21,18,40,12,13,14,12,35,4,9,4,9,24,24,22,22,22,22,22,37,37,37,37,37,37,37,37,34,34,34,34,38,38,38,38,23,22,23,20,20,20,20,20,20,20,18,24,34,34,34,34,34,34,34,34,34,34,34,34,60,60,21,21,21,21,21,20,18,44,50,50,22,22,22,22,18,18,18,18,24,24,24,24,24,24,22,22,22,22,22,37,74,21,21,21,21,21,59,21,20,20,20,20,20,27,27,27,27,27,27,27,27,11,32,5,32,9,4,7,4,7,61,44,61,44]
        # targets = [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0]

        q = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 4, 5, 4, 5, 4, 5]
        targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]

        q_data = np.zeros(params.seqlen + 1)
        qa_data = np.zeros(params.seqlen + 1)
        q_data[:len(q)] = q[:]
        for i in range(len(q)):
            qa_data[i] = int(q_data[i]) + int(targets[i]) * params.n_question

        best_epoch = 21
        file_name = 'b' + str(params.batch_size) + \
                    '_q' + str(params.q_embed_dim) + \
                    '_qa' + str(params.qa_embed_dim) + \
                    '_std' + str(params.init_std) + \
                    '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                    '_f' + str(params.final_fc_dim) + '_s' + str(seedNum)

        # (unique_questions, seqlen)
        q_data = q_data.reshape(1, params.seqlen + 1)
        qa_data = qa_data.reshape(1, params.seqlen + 1)
        params.batch_size = 1
        predict_matrix, unique_questions = get_mastery_by_one_seq(params, file_name, q_data, qa_data, best_epoch)
        print("predict_matrix", predict_matrix)
        print("unique_questions", unique_questions)
        show(predict_matrix, unique_questions, q_data)



    # 训练
    if not params.test and not params.heatmap:
        d = vars(params)
        for key in d:
            log.info('\t' + str(key) + '\t' + str(d[key]))
        file_name = 'b' + str(params.batch_size) + \
                    '_q' + str(params.q_embed_dim) + \
                    '_qa' + str(params.qa_embed_dim) + \
                    '_std' + str(params.init_std) + \
                    '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                    '_f' + str(params.final_fc_dim) + '_s' + str(seedNum)
        train_data_path = params.data_dir + "/" + params.data_name + "_train1.csv"
        valid_data_path = params.data_dir + "/" + params.data_name + "_valid1.csv"
        train_q_data, train_qa_data = dat.load_data(train_data_path)
        valid_q_data, valid_qa_data = dat.load_data(valid_data_path)
        log.info("\n")
        log.info("train_qa_data.shape " + str(train_qa_data.shape) + ' ' + str(
            train_qa_data))  ###(3633, 200) = (#sample, seqlen)
        log.info("valid_qa_data.shape " + str(valid_qa_data.shape))  ###(1566, 200)
        log.info("\n")
        best_epoch = train_one_dataset(params, file_name, train_q_data, train_qa_data, valid_q_data, valid_qa_data)
        if params.train_test:
            test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
            test_q_data, test_qa_data = dat.load_data(test_data_path)
            test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch)

    # 测试
    if params.test:
        test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
        test_q_data, test_qa_data = dat.load_data(test_data_path)
        best_epoch = 30
        file_name = 'b' + str(params.batch_size) + \
                    '_q' + str(params.q_embed_dim) + \
                    '_qa' + str(params.qa_embed_dim) + \
                    '_std' + str(params.init_std) + \
                    '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                    '_f' + str(params.final_fc_dim) + '_s' + str(seedNum)
        test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch)
