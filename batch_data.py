import numpy as np
import sys, os
import cv2
from setting import *
from data_process.image_normalize import *
# curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

STA = "_STA" # for start words
UNK = "_UNK" # for unknown words
PAD = "_PAD" # for padding
END = "_END"


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
        dtype: type of data
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int32)

    return indices, values, shape


def load_vocab(filename):
    """
    Args:
        filename: (string) path to vocab txt file one word per line
    Returns:
        dict: d[token] = id
    """
    vocab = dict()
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            vocab[token] = idx + 4

    # add pad and unk tokens
    vocab[STA] = 0
    vocab[PAD] = 1
    vocab[UNK] = 2
    vocab[END] = 3

    return vocab


class DataBatch:
    def __init__(self, train_path=None, validate_path=None, test_path=None, vocab_path=None, formulas_path=None,
                 image_path=None, batch_size=10):
        self.train_path = train_path
        self.validate_path = validate_path
        self.test_path = test_path
        self.vocab_path = vocab_path
        self.formulas_path = formulas_path
        self.image_path = image_path
        self.batch_size = batch_size

        self.vocab_to_idx = load_vocab(self.vocab_path)
        print('#######', len(self.vocab_to_idx))
        # formulas = open(self.formulas_path).read().strip().split('\t')[1]
        # self.formulas = [self.formula_to_indices(formula) for formula in formulas]

    def formula_to_indices(self, formula):
        # TODO: attention this.
        formula = formula.split()
        res = [self.vocab_to_idx['_STA']]
        for token in formula:
            assert token != '\n'
            if token in self.vocab_to_idx:
                res.append(self.vocab_to_idx[token])
            else:
                res.append(self.vocab_to_idx['_UNK'])
        return res

    def import_images(self, datum):
        datum = datum.strip().split('\t')
        if len(datum) < 2:
            return None, None
        path = self.image_path + '/' + datum[0]
        if not os.path.exists(path):
            print('the unreasonable path:', path)
            return None, None
        img = cv2.imread(path, 0)
        if type(img) == type(None):
            return None, None
        img = resize(img)
        img = pad_group_image(img)
        if img.shape[0] != 46 or img.shape[1] > 1600:
            return None, None
        assert img.shape[0] == 46 and img.shape[1] <= 1600, print(img.shape)
        return img, self.formula_to_indices(datum[1])

    def load_data(self):
        if self.train_path:
            train = open(self.train_path).read().split('\n')
            train = map(self.import_images, train)
        else:
            train = None
        if self.validate_path:
            validate = open(self.validate_path).read().split('\n')
            validate = map(self.import_images, validate)
        else:
            validate = None
        if self.test_path:
            test = open(self.test_path).read().split('\n')
            test = map(self.import_images, test)
        else:
            test = None
        return train, validate, test, len(self.vocab_to_idx)

    def gen_training_data(self, data):
        vocab_to_idx = self.vocab_to_idx
        res = {}  # save data in a dict by images shape , the dict keys is image shape.
        for datum in data:
            if datum[0] is None:
                continue
            if datum[0].shape not in res:
                res[datum[0].shape] = [datum]
            else:
                res[datum[0].shape].append(datum)
        batches = []
        # print('@@@@@@@', res[(50, 240)])
        for size in res:
            # batch by similar sequence length within each image-size group -- this keeps padding to a
            # minimum
            group = sorted(res[size], key=lambda x: len(x[1]))
            count = 0
            for i in range(0, len(group), self.batch_size):
                count += 1
                images = map(lambda x: np.expand_dims(np.expand_dims(x[0], 0), 3),
                             group[i:i + self.batch_size])  # add new dimension
                # print(list(images)[0].shape)
                batch_images = np.concatenate(list(images), 0)  # shape(batch_size, 50, 320, 1)

                ctc_feature_length = []
                for kk in range(batch_images.shape[0]):
                    ctc_feature_length.append(batch_images[kk, :, :, :].shape[1])
                ctc_feature_length = np.array(ctc_feature_length)
                seq_len = max(
                    [len(x[1]) for x in group[i:i + self.batch_size]])  # the bigget length label in batch_size

                def preprocess(x):
                    arr = np.array(x[1])
                    arr = np.pad(arr, (0, seq_len - arr.shape[0]), 'constant', constant_values=vocab_to_idx[PAD])
                    pad = np.pad(arr, (0, 1), 'constant', constant_values=vocab_to_idx[END])
                    return np.expand_dims(pad, 0)

                labels = map(preprocess,
                             group[i:i + self.batch_size])  # in batch_size , add the label lenth to equal seq_len + 1
                att_labels = np.concatenate(list(labels), 0)  # shape (batch_size, seq_len+1)
                ctc_labels = []
                for kk in att_labels:
                    tmp_ctc_label = list(kk)
                    tmp = tmp_ctc_label.copy()
                    for ii in tmp:
                        if ii in [0, 1, 2, 3]:
                            tmp_ctc_label.remove(ii)
                    for ii in range(4):
                        assert ii not in tmp_ctc_label, print(tmp_ctc_label)
                    ctc_labels.append([x - 4 for x in tmp_ctc_label])
                # print('~~~~~~`', len(ctc_labels[0]), len(ctc_labels[1]))
                sparse_y = sparse_tuple_from(ctc_labels)
                # print('!!!!!!', sparse_y)
                if batch_images.shape[0] == self.batch_size:
                    batches.append({'input_image': batch_images,
                                    'ctc_label': sparse_y,
                                    'ctc_feature_length': ctc_feature_length,
                                    'att_train_length': np.array([seq_len + 1] * self.batch_size),
                                    'att_labels': att_labels})
        return batches


if __name__ == '__main__':
    train_path = '../data/chinese_formula_data/label.txt'
    # test_path = '../id_data/test_filter.lst'
    test_path = '../chinese_formula_data/tmp.lst'
    vocab_path = '../data/chinese_formula_data/vocab.txt'
    image_path = '../data/chinese_formula_data/processed_image'
    formula_path = '../data/chinese_formula_data/label.txt'
    data = DataBatch(train_path=train_path, validate_path=None, test_path=None,
                                    vocab_path=vocab_path, formulas_path=formula_path, image_path=image_path,
                                    batch_size=2)
    train_data, val_data, test_data, vocab_size = data.load_data()
    g = data.gen_training_data(train_data)
    print(g[0]['ctc_label'])
