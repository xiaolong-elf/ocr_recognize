import tensorflow as tf
import abc
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, BasicDecoder, \
    ScheduledEmbeddingTrainingHelper, dynamic_decode, GreedyEmbeddingHelper, BeamSearchDecoder, tile_batch
from units.crnn_net import rnn_layers
from units.crnn_net import convnet_layers
mode = learn.ModeKeys.TRAIN


class CtcPlusAttModel(object):
    """
    Class CtcPlusAttModel
    """
    def __init__(self, vocab_size, positional_embeddings=False,
                 beam_width=1, alignment_history=False):
        """
        Initialize global variables and compute graph
        """
        # vocabulary parameters
        # input image
        self.beam_width = beam_width
        self.attention_mode = 0
        self.vocab_size = vocab_size
        self.learning_rate = tf.placeholder(tf.float32)
        self.input_image = tf.placeholder(tf.float32, shape=(None, 32, None, 1), name='img_data')
        self.batch_size = tf.shape(self.input_image)[0]

        # attention part placeholder
        self.att_label = tf.placeholder(tf.int32, shape=[None, None], name='att_label')
        self.att_train_length = tf.placeholder(tf.int32, shape=[None], name='att_train_length')
        # self.eight = tf.constant(8, dtype=tf.int32)
        # ctc part placeholder
        self.ctc_label = tf.sparse_placeholder(tf.int32, name='ctc_label')
        self.ctc_feature_length = tf.placeholder(tf.int32, shape=[None], name='ctc_feature_length')
        self.max_dec_iteration = tf.placeholder(tf.int32, shape=[1])
        self.enc_lstm_dim = 256
        self.dec_lstm_dim = 512
        self.embedding_size = 512
        self.ctc_loss_weights = 0.2
        self.att_loss_weights = 1 - self.ctc_loss_weights
        self.wd = 0.00002
        self.momentum = 0.9
        self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size])

        self.cnn_out, self.sequence_len = convnet_layers(self.input_image, self.ctc_feature_length, mode)
        self.enc_outputs = rnn_layers(self.cnn_out, self.sequence_len, self.enc_lstm_dim)

        attention_weights_depth = 2 * self.enc_lstm_dim
        attention_layer_size = 2 * self.enc_lstm_dim
        attention_states = tf.reshape(self.enc_outputs, [self.batch_size, -1, 2 * self.enc_lstm_dim])
        attention_states_tiled = tile_batch(attention_states, self.beam_width)  # For generalization

        attention_mechanism = BahdanauAttention(attention_weights_depth, attention_states_tiled)

        dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.dec_lstm_dim)
        self.cell = AttentionWrapper(cell=dec_lstm_cell,
                                     attention_mechanism=attention_mechanism,
                                     attention_layer_size=attention_layer_size,
                                     alignment_history=alignment_history)
        self.setup_decoder()
        self.final_outputs, self.final_state, _ = dynamic_decode(self.decoder,
                                                                 maximum_iterations=self.max_dec_iteration[0] - 1)
        self.ctc_loss_branch()
        self.finalize_model()

    @abc.abstractmethod
    def setup_decoder(self):
        pass

    @abc.abstractmethod
    def finalize_model(self):
        pass

    @ abc.abstractclassmethod
    def ctc_loss_branch(self):
        pass


class TrainModel(CtcPlusAttModel):
    def __init__(self, vocab_size):
        super(TrainModel, self).__init__(vocab_size, beam_width=1)

    def ctc_loss_branch(self):
        """
        Ctc loss compute graph
        :param rnn_features: encoded features and self.ctc_feature_length、self.ctc_label
        :return: loss matrix
        """
        logit_activation = tf.nn.relu
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        # logits = rnn_layers(rnn_features, sequence_len, self.vocab_size -4)
        logits = tf.layers.dense(inputs=self.enc_outputs,
                                 units=self.vocab_size - 4 + 1,  # num_class + 1
                                 activation=logit_activation,
                                 kernel_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 name='logits')
        self.ctc_loss = tf.nn.ctc_loss(labels=self.ctc_label,
                                  inputs=logits,
                                  sequence_length=self.sequence_len,
                                  ignore_longer_outputs_than_inputs=True,
                                  time_major=False)

    def setup_decoder(self):
        output_embed = tf.nn.embedding_lookup([self.embedding], self.att_label[:, :-1])
        decoder_lengths = tf.tile([self.max_dec_iteration[0] - 1], [self.batch_size])
        helper = ScheduledEmbeddingTrainingHelper(output_embed, decoder_lengths, self.embedding, 0.1)

        output_layer = Dense(units=self.vocab_size)
        self.decoder = BasicDecoder(cell=self.cell, helper=helper,
                                    initial_state=self.cell.zero_state(dtype=tf.float32,
                                    batch_size=self.batch_size), output_layer=output_layer)

    def finalize_model(self):
        final_outputs = self.final_outputs.rnn_output
        self.att_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_outputs,
                                                                  labels=self.att_label[:, 1:])
        correct_prediction = tf.equal(tf.to_int32(tf.argmax(final_outputs, 2)), self.att_label[:, 1:])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

        attention_loss = tf.reduce_mean(self.att_loss)

        # ctc part
        ctc_loss = tf.reduce_mean(self.ctc_loss)

        # merge part
        t_loss = attention_loss * self.att_loss_weights + ctc_loss * self.ctc_loss_weights

        l2_losses = []
        for var in tf.trainable_variables():
            # if var.op.name.find(r'DW') > 0:
            l2_losses.append(tf.nn.l2_loss(var))
        self.tt_loss = tf.multiply(self.wd, tf.add_n(l2_losses)) + t_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.tt_loss)

    # def build_model(self):
    #     """
    #     build compute graph
    #     :return: model
    #     """
    #     # attention part
    #     attention_loss = tf.reduce_mean(self.att_loss)
    #
    #     # ctc part
    #     ctc_loss = tf.reduce_mean(self.ctc_loss)
    #
    #     # merge part
    #     t_loss = attention_loss*self.att_loss_weights + ctc_loss*self.ctc_loss_weights
    #
    #     l2_losses = []
    #     for var in tf.trainable_variables():
    #         # if var.op.name.find(r'DW') > 0:
    #         l2_losses.append(tf.nn.l2_loss(var))
    #     tt_loss = tf.multiply(self.wd, tf.add_n(l2_losses)) + t_loss
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         train_step = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(tt_loss)
    #     return train_step, tt_loss, self.accuracy

    def train(self, sess, feed_dict):
        _, tt_loss, acc = sess.run([self.train_step, self.tt_loss, self.accuracy],
                                    feed_dict=feed_dict)
        return tt_loss, acc


class BeamSearchInfer(CtcPlusAttModel):
    def __init__(self, vocab_size, beam_width=5):
        super(BeamSearchInfer, self).__init__(vocab_size, beam_width=beam_width)
        self.beam_width = beam_width

    def setup_decoder(self):
        self.dec_init_state = self.cell.zero_state(self.batch_size * self.beam_width, dtype=tf.float32)
        self.decoder = BeamSearchDecoder(cell=self.cell,
                                         embedding=self.embedding,
                                         start_tokens=tf.tile([0], [self.batch_size]),
                                         end_token=-1,
                                         initial_state=self.dec_init_state,
                                         beam_width=self.beam_width,
                                         output_layer=tf.layers.Dense(self.vocab_size))

    def finalize_model(self):
        self.predicted_labels = self.final_outputs.predicted_ids[:, :, 0]
        correct_prediction = tf.equal(self.predicted_labels, self.att_label[:, 1:])
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, sess, feed_dict):
        return sess.run(self.predicted_labels, feed_dict=feed_dict)

