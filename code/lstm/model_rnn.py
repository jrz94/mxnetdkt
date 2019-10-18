import mxnet as mx
import mxnet.ndarray as nd
import ast as ast


def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr


class LogisticRegressionMaskOutput(mx.operator.CustomOp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutput, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 1.0 / (1.0 + nd.exp(- in_data[0])))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        output = out_data[0].asnumpy()
        label = in_data[1].asnumpy()
        data_grad = (output - label) * (label != self.ignore_label)
        self.assign(in_grad[0], req[0], data_grad)


@mx.operator.register("LogisticRegressionMaskOutput")
class LogisticRegressionMaskOutputProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutputProp, self).__init__(need_top_grad=False)
        self.ignore_label = safe_eval(ignore_label)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogisticRegressionMaskOutput(ignore_label=self.ignore_label)


def logistic_regression_mask_output(data, label, ignore_label, name=None):
    return mx.sym.Custom(name=name,
                         op_type="LogisticRegressionMaskOutput",
                         ignore_label=ignore_label,
                         data=data,
                         label=label)


class ModelRnn(object):
    def __init__(self, n_question, seqlen, batch_size,
                 q_embed_dim,
                 qa_embed_dim,
                 final_fc_dim, name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.final_fc_dim = final_fc_dim
        self.name = name

    def sym_gen(self):
        q_data = mx.sym.Variable('q_data', shape=(self.seqlen, self.batch_size))  # (seqlen, batch_size)
        qa_data = mx.sym.Variable('qa_data', shape=(self.seqlen, self.batch_size))  # (seqlen, batch_size)
        target = mx.sym.Variable('target', shape=(self.seqlen, self.batch_size))  # (seqlen, batch_size)
        # todo: 传入pos_data, pos_data = 目前为止当前题目做过的次数/目前为止总题目个数
        pos_data = mx.sym.Variable('pos_data', shape=(self.seqlen, self.batch_size))  # (seqlen, batch_size)

        ### embedding
        qa_data = mx.sym.BlockGrad(qa_data)
        q_data = mx.sym.BlockGrad(q_data)
        pos_data = mx.sym.BlockGrad(pos_data)

        # (seqlen, batch_size, 1)
        pos_data = mx.sym.reshape(data=pos_data, shape=(self.seqlen, self.batch_size, 1))

        # (seqlen, batch_size, qa_embed_dim)
        qa_embed_data = mx.sym.Embedding(data=qa_data, input_dim=self.n_question * 2 + 1,
                                         output_dim=self.qa_embed_dim, name='qa_embed')
        # (seqlen, batch_size, q_embed_dim)
        q_embed_data = mx.sym.Embedding(data=q_data, input_dim=self.n_question + 1,
                                        output_dim=self.q_embed_dim, name='q_embed')

        # (seqlen, batch_size, qa_embed_dim + q_embed_dim + 1)
        concat_q_qa_data = mx.sym.Concat(q_embed_data, qa_embed_data, pos_data, num_args=3, dim=2)

        # lstm
        lstm = mx.gluon.rnn.GRU(50, num_layers=1, layout='TNC', prefix='RNN')
        # (seqlen, batch_size, qa_embed_dim + q_embed_dim + 1)
        output = lstm(concat_q_qa_data)
        # (seqlen * batch_size, qa_embed_dim + q_embed_dim + 1)
        output = mx.sym.reshape(data=output, shape=(-3, 0))
        # (seqlen * batch_size, 1)
        pred = mx.sym.FullyConnected(data=output, num_hidden=1, name='pred')
        pred_prob = logistic_regression_mask_output(data=mx.sym.Reshape(pred, shape=(-1,)),
                                                    label=mx.sym.Reshape(data=target, shape=(-1,)),
                                                    ignore_label=-1., name='final_pred')
        net = mx.sym.Group([pred_prob])
        # 网络结构可视化
        #mx.viz.plot_network(net).view()
        return net


