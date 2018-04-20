import tensorflow as tf
import random, time
import os, sys
import numpy as np
import logging
from batch_data import DataBatch
from units.model import TrainModel, BeamSearchInfer
from units.lr_shedule import LRSchedule
from units.visualize import display_result
from setting import *
from units.model_test import *

sys.path.append(PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto(
    # device_count={'CPU': 1, 'GPU': 1},
    # gpu_options=gpu_option,
    allow_soft_placement=True,
    log_device_placement=False)


logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        datefmt='%d %b %Y %H:%M:%S',
        filename='./log.txt',
        filemode='w')


class TrainWork:
    def __init__(self, train_path=None, validate_path=None, test_path=None,
                 vocab_path=None, formulas_path=None, image_path=None, batch_size=1):
        self.batch_size = batch_size
        self.lr_init = 0.1
        self.lr_min = 0.00001
        self.epochs = 60
        self.start_epoch = 0
        self.save_id = './saved_models/model-'
        self.model = './saved_models/model-17-04-2018--10-48/'

        self.data_batch = DataBatch(train_path=train_path, validate_path=validate_path, test_path=test_path,
                                    vocab_path=vocab_path, formulas_path=formulas_path, image_path=image_path,
                                    batch_size=batch_size)
        train_data, val_data, test_data, self.vocab_size = self.data_batch.load_data()
        if train_data:
            self.train_data = self.data_batch.gen_training_data(train_data)
            self.total_step = len(self.train_data) * self.epochs
            random.shuffle(self.train_data)
        if val_data:
            self.val_data = self.data_batch.gen_training_data(val_data)
            random.shuffle(self.val_data)
        if test_data:
            self.test_data = self.data_batch.gen_training_data(test_data)

    def init_session(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.train_model = TrainModel(vocab_size=self.vocab_size)
            self.train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    def get_checkpoint(self):
        """Get the checkpoint path from the given model output directory"""
        ckpt = tf.train.get_checkpoint_state(self.model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
        else:
            raise RuntimeError('No checkpoint file found')
        return ckpt_path

    def restore_session(self, sess, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.train_saver.restore(sess, dir_model)

    def train_process(self, restore=False):
        lr_schedule = LRSchedule(lr_init=self.lr_init, lr_min=self.lr_min, total_step=self.total_step,
                                 lr_type='pow')
        with tf.Session(config=config, graph=self.train_graph) as sess:
            if restore:
                checkpoint = self.get_checkpoint()
                self.restore_session(sess, checkpoint)
            else:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                sess.run(init_op)
            for epoch in range(self.start_epoch, self.epochs):
                print('the epoch is:', epoch)
                epoch_start_time = time.time()
                random.shuffle(self.train_data)
                for j in range(len(self.train_data)):
                    input_data = self.train_data[j]
                    feed_dict = {self.train_model.input_image: input_data['input_image'],
                                 self.train_model.ctc_label: input_data['ctc_label'],
                                 self.train_model.ctc_feature_length: input_data['ctc_feature_length'],
                                 self.train_model.att_train_length: input_data['att_train_length'],
                                 self.train_model.att_label: input_data['att_labels'],
                                 self.train_model.max_dec_iteration: [input_data['att_labels'].shape[1]],
                                 self.train_model.learning_rate: lr_schedule.lr}
                    loss_print, accuracy = self.train_model.train(sess, feed_dict=feed_dict)

                    batch_no = epoch * len(self.train_data) + j
                    lr_schedule.update(batch_no=batch_no)

                    if batch_no % 20 == 0:
                        print("step: %d/%d, lr: %.6f, loss: %g, acc: %g" % (batch_no, self.total_step,
                                                                            lr_schedule.lr, loss_print, accuracy))
                        logging.info("step: %d/%d, lr: %.6f, loss: %g, acc: %g" % (batch_no, self.total_step,
                                                                                   lr_schedule.lr, loss_print,
                                                                                   accuracy))
                print("Time for epoch:%f mins" % ((time.time() - epoch_start_time) / 60))
                if (epoch + 1) % 2 == 0:
                    id = self.save_id + time.strftime("%d-%m-%Y--%H-%M")
                    if not os.path.exists(id):
                        os.makedirs(id)
                    self.train_saver.save(sess, id + '/model')

    def infer_process(self):
        infer_model = BeamSearchInfer(vocab_size=self.vocab_size)
        infer_saver = tf.train.Saver()
        infer_sess = tf.Session(config=config)
        checkpoint = self.get_checkpoint()
        infer_saver.restore(infer_sess, checkpoint)
        for j in range(len(self.test_data)):
            input_data = self.test_data[j]
            start = time.time()
            feed_dict = {infer_model.input_image: input_data['input_image'],
                         infer_model.ctc_label: input_data['ctc_label'],
                         infer_model.ctc_feature_length: input_data['ctc_feature_length'],
                         infer_model.att_train_length: input_data['att_train_length'],
                         infer_model.att_label: input_data['att_labels'],
                         infer_model.max_dec_iteration: [input_data['att_labels'].shape[1]]}
            predict_labels = infer_model.predict(infer_sess, feed_dict=feed_dict)
            print('the cost time is:', time.time() - start)
            display_result(input_data['input_image'][0], input_data['att_labels'][0], predict_labels[0])


if __name__ == "__main__":
    train_path = '../data/baidu_data/baidu.lst'
    # test_path = '../id_data/test_filter.lst'
    test_path = '../chinese_formula_data/tmp.lst'
    vocab_path = '../data/baidu_data/vocab.txt'
    image_path = '../data/baidu_data/process_image'
    formula_path = '../data/baidu_data/baidu.lst'
    mulwork = TrainWork(train_path=train_path, validate_path=None, test_path=None,
                        vocab_path=vocab_path, image_path=image_path, formulas_path=formula_path, batch_size=20)
    mulwork.init_session()
    mulwork.train_process(restore=False)
    # mulwork.infer_process()
