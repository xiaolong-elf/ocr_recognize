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


class ModelTest(object):
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
        self.input_image = tf.placeholder(tf.float32, shape=(None, 46, None, 1), name='img_data')
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
        self.ctc_loss_weights = 0.5
        self.att_loss_weights = 1 - self.ctc_loss_weights
        self.wd = 0.00002
        self.momentum = 0.9
        self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size])

        self.cnn_out, self.sequence_len = convnet_layers(self.input_image, self.ctc_feature_length, mode)
        self.setup_decoder()
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


class TrainTest(ModelTest):
    def __init__(self, vocab_size):
        super(TrainTest, self).__init__(vocab_size, beam_width=1)

    def train(self, sess, feed_dict):
        cnn_out, sequence_len = sess.run([self.cnn_out, self.sequence_len],
                                          feed_dict=feed_dict)
        print('@@@@@@@@', sequence_len, cnn_out.shape)
        assert sequence_len[0] == cnn_out.shape[1], print('#########', sequence_len, cnn_out.shape)
        return sequence_len

