'''
1. data generate
    1.1 loads vaoca
    1.2 loads image features
    1.3 provide data for training
2. builds image caption model
3. train the model
'''

import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle
import numpy as np
import math
import random

tf.logging.set_verbosity(tf.logging.INFO)

input_descrption_file = os.path.abspath(os.path.join(os.getcwd(),'../flickr30k/results_20130124.token'))
input_img_feature_dir = os.path.abspath(os.path.join(os.getcwd(),'../download_inception_v3-feature'))
input_vocab_file = os.path.abspath(os.path.join(os.getcwd(),'../flickr30k/vocab.txt'))
output_dir = os.path.abspath(os.path.join(os.getcwd(),'../download_inception_v3_features/local_run'))

if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)


def get_default_param():
    return tf.contrib.training.HParams(
        num_vocab_word_threashold = 3,
        num_embedding_nodes = 32,
        num_timesteps = 10,
        num_lstm_nodes = [64,64],
        num_lstm_layers = 2,
        num_fc_nodes = 32,
        batch_size = 80,
        cell_type = 'lstm',
        clip_lstm_grads = 1.0,
        learning_rate = 0.001,
        keep_prob = 0.8,
        log_frequence = 100,
        save_frequence = 1000
    )

hps = get_default_param()
#print (hps.save_frequence)

class Vocab(object):
    def __init__(self, filename, word_num_threshold):
        self._id_to_word = {}
        self._word_to_id = {}
        self._unk = -1
        self._eos = -1
        self._word_num_threshold = word_num_threshold
        self._read_dict(filename)

    def _read_dict(self,filename):
        with gfile.GFile(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            word, occurence = line.strip('\r\n').split('\t')
            occurence = int(occurence)
            if occurence < self._word_num_threshold:
                continue
            idx = len(self._id_to_word)
            if word == '<UNK>':
                self._unk = idx
            elif word == '.':
                self._eos = idx

            if word in self._word_to_id or idx in self._id_to_word:
                raise Exception('duplicate words in vocab')

            self._word_to_id[word] = idx
            self._id_to_word[idx] = word

    @property
    def unk(self):
        return self._unk

    @property
    def eos(self):
        return self._eos

    def word_to_id(self,word):
        return self._word_to_id.get(word,self._unk)

    def id_to_word(self, word_id):
        return self._id_to_word.get(word_id, '<UNK>')

    def size(self):
        return len(self._id_to_word)

    def encode(self, sentence):
        return [self.word_to_id(word) for word in sentence.split(' ')]

    def decode(self, sentence_id):
        words =  [self.id_to_word(word_id) for word_id in sentence_id]
        return  ' '.join(words)

vocab = Vocab(input_vocab_file, hps.num_vocab_word_threashold)
vocab_size = vocab.size()
#logging.info("vocab_size: %d" %vocab_size)
#pprint.pprint(vocab.encode('I have a dream'))
#pprint.pprint(vocab.decode([5, 10, 9, 20]))


def parse_token_file(token_file):
    '''parse image description file'''
    img_name_to_tokens = {}
    with gfile.GFile(token_file,'r') as f:
        lines = f.readlines()

    for line in lines:
        img_id, descrption = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_tokens.setdefault(img_name, [])
        img_name_to_tokens[img_name].append(descrption)

    return img_name_to_tokens

def conver_token_to_id(img_name_to_tokens, vocab):
    '''convert tokens to each descrption of image to id'''
    img_name_to_tokens_id = {}
    for img_name in img_name_to_tokens:
        img_name_to_tokens_id.setdefault(img_name, [])
        for description in img_name_to_tokens[img_name]:
            token_ids = vocab.encode(description)
            img_name_to_tokens_id[img_name].append(token_ids)

    return img_name_to_tokens_id

img_name_to_tokens = parse_token_file(input_descrption_file)
img_name_to_tokens_id = conver_token_to_id(img_name_to_tokens, vocab)

'''
logging.info("num of all imgs: %d" %len(img_name_to_tokens))
pprint.pprint(img_name_to_tokens['2778832101.jpg'])
logging.info("num of all imgs: %d" %len(img_name_to_tokens_id))
pprint.pprint(img_name_to_tokens_id['2778832101.jpg'])
'''

class ImageCaptionData(object):
    '''provide data for image caption model'''
    def __init__(self,
                 img_name_to_tokens_id,
                 img_feature_dir,
                 num_timesteps,
                 vocab,
                 deterministic = False):
        self._vocab = vocab
        self._img_name_to_token_id = img_name_to_tokens_id
        self._num_timesteps = num_timesteps
        self._deterministic = deterministic
        self._indicator = 0

        self._img_feature_filenames = []
        self._img_feature_data = []

        self._all_img_features_filepaths = []

        for filename in gfile.ListDirectory(img_feature_dir):
            self._all_img_features_filepaths.append(os.path.join(img_feature_dir, filename))

        pprint.pprint(self._all_img_features_filepaths)

        self._load_img_feature_pickle()

        if not self._deterministic:
            self._random_shuffle()

    def _load_img_feature_pickle(self):
        '''loads image feature data from pickle'''
        for filepath in self._all_img_features_filepaths:
            logging.info('loadding %s' %filepath)
            with gfile.GFile(filepath, 'rb') as f:
                filenames, features = pickle.load(f)
                self._img_feature_filenames += filenames
                self._img_feature_data.append(features)

        #[#(1000,1,1,2048),(1000,1,2048)] ->(2000,1,1,2048)
        self._img_feature_data = np.vstack(self._img_feature_data)
        origin_shape = self._img_feature_data.shape
        self._img_feature_data = np.reshape(self._img_feature_data,
                                            (origin_shape[0],origin_shape[3]))
        self._img_feature_filenames = np.asarray(self._img_feature_filenames)
        print(self._img_feature_data.shape)
        print(self._img_feature_filenames)

    def size(self):
        return len(self._img_feature_filenames)

    def img_feature_size(self):
        return self._img_feature_data.shape[1]

    def _random_shuffle(self):
        '''shuffle data randomly'''
        p = np.random.permutation(self.size())
        self._img_feature_filenames = self._img_feature_filenames[p]
        self._img_feature_data = self._img_feature_data[p]

    def _img_desc(self, batch_filenames):
        '''get descrption for filenames in batch'''
        batch_sentence_ids = []
        batch_weights = []
        for filename in batch_filenames:
            token_ids_set = self._img_name_to_token_id[filename]
            chosen_token_ids = random.choice(token_ids_set)
            chosen_token_ids_length = len(chosen_token_ids)

            weight = [1 for i in range(chosen_token_ids_length)]
            if chosen_token_ids_length >= self._num_timesteps:
                chosen_token_ids = chosen_token_ids[0:self._num_timesteps]
                weight = weight[0:self._num_timesteps]
            else:
                remainging_length = self._num_timesteps - chosen_token_ids_length
                chosen_token_ids += [self._vocab.eos for i in range(remainging_length)]
                weight += [0 for i in range(remainging_length)]
            batch_sentence_ids.append(chosen_token_ids)
            batch_weights.append(weight)

        batch_sentence_ids = np.asarray(batch_sentence_ids)
        batch_weights = np.asarray(batch_weights)
        return batch_sentence_ids, batch_weights

    def nex_batch(self, batch_size):
        '''return next batch data'''
        end_indicator = self._indicator + batch_size
        if end_indicator> self.size():
            if not self._deterministic:
                self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size

        assert end_indicator < self.size()

        batch_filenames = self._img_feature_filenames[self._indicator: end_indicator]
        batch_img_features = self._img_feature_data[self._indicator: end_indicator]
        batch_sentence_ids, batch_weights = self._img_desc(batch_filenames)
        self._indicator = end_indicator
        #sentence_ids :[100,101,102,10,3,0,0,0] -> [1,1,1,1,1,0,0,0] batch_weights权重
        return batch_img_features, batch_sentence_ids, batch_weights, batch_filenames



caption_data = ImageCaptionData(img_name_to_tokens_id,
                                input_img_feature_dir,
                                hps.num_timesteps,
                                vocab)

img_feature_dim = caption_data.img_feature_size()
caption_data_size = caption_data.size()
'''
logging.info('img_feature_dim: %d' %img_feature_dim)
logging.info('caption_data_size: %d' %caption_data_size)

batch_img_features,batch_sentence_ids,batch_weights,batch_img_names = caption_data.nex_batch(5)
pprint.pprint(batch_img_features)
pprint.pprint(batch_sentence_ids)
pprint.pprint(batch_weights)
pprint.pprint(batch_img_names)
'''

def create_rnn_cell(hidden_dim, cell_type):
    '''return specific cell according to cell_type'''
    if cell_type == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_dim,
                                            state_is_tuple = True)
    elif cell_type == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_dim)
    else:
        raise Exception('%s type has not been supported' % cell_type)

def dropout(cell, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(cell,
                                         output_keep_prob = keep_prob)



def get_train_model(hps, vocab_size, img_feature_dim):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    img_feature = tf.placeholder(tf.float32, (batch_size, img_feature_dim))
    sentence = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    mask = tf.placeholder(tf.float32, (batch_size, num_timesteps))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(tf.zeros([], tf.int32),
                              name='global_step',
                              trainable = False)
    #prediction process
    #sentence :[a,b,c,d,e]
    #input: [img, a, b, c, d, e]
    #img_feature: [0.4,.03,10,2]
    #predict #1: img_feature --> embedding_img ->lstm-> (a)
    #predict #2: a->embedding_word->lstm->(b)
    #predict #3: b->embedding_word->lstm->(c)
    #..

    #set up embedding layer
    embedding_initializer = tf.random_uniform_initializer(-1.0,1.0)
    with tf.variable_scope('embedding',
                           initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding',
                                     [vocab_size, hps.num_embedding_nodes],
                                     tf.float32)
        embed_token_ids = tf.nn.embedding_lookup(embeddings,
                                                 sentence[:,0: num_timesteps-1])

    img_feature_embed_init = tf.uniform_unit_scaling_initializer(factor=1.0)

    with tf.variable_scope('img_feature_embed',
                           initializer=img_feature_embed_init):
        #img_feature: [batch_size, img_feature_dim]
        #embed_img :[batch_size,num_embedding_nodes]
        embed_img = tf.layers.dense(img_feature,
                                    hps.num_embedding_nodes)
        #embed_img: [batch_size,1,num_embedding_nodes]
        embed_img = tf.expand_dims(embed_img, 1)
        #embde_input: [batch_size, num_timesteps, num_embedding_nodes]
        embed_inputs = tf.concat([embed_img, embed_token_ids], axis=1 )

    #set up rnn network
    scale = 1.0/math.sqrt(hps.num_embedding_nodes+hps.num_lstm_nodes[-1])/3.0
    rnn_init = tf.random_uniform_initializer(-scale, scale)
    print(rnn_init)
    with tf.variable_scope('lstm_nn',
                           initializer=rnn_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = create_rnn_cell(hps.num_lstm_nodes[i], hps.cell_type)
            cell = dropout(cell, keep_prob)
            cells.append(cell)

        cell = tf.contrib.rnn.MultiRNNCell(cells)
        init_state = cell.zero_state(hps.batch_size, tf.float32)
        #rnn_outputs: [batch_size, num_timestep, hps.num_lstm_node[-1]]
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                           embed_inputs,
                                           initial_state=init_state)

    #set up fully-connected layer
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc',initializer=fc_init):
        rnn_output_2d = tf.reshape(rnn_outputs,
                                   [-1,hps.num_lstm_nodes[-1]])
        fc1 = tf.layers.dense(rnn_output_2d,
                              hps.num_fc_nodes,
                              name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        fc_relu = tf.nn.relu(fc1_dropout)
        logits = tf.layers.dense(fc_relu,
                                 vocab_size,
                                 name='logits')

    #calculates loss
    with tf.variable_scope('loss'):
        sentence_flatten = tf.reshape(sentence, [-1])
        mask_flattern = tf.reshape(mask, [-1])
        mask_sum = tf.reduce_sum(mask_flattern)

        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=sentence_flatten)
        weight_softmax_loss = tf.multiply(softmax_loss,
                                          tf.cast(mask_flattern,tf.float32))
        loss = tf.reduce_mean(weight_softmax_loss)/mask_sum

        prediction = tf.argmax(logits, 1, output_type=tf.int32)
        correction_prediction = tf.equal(prediction,
                                         sentence_flatten)
        weighted_correct_prediction = tf.multiply(tf.cast(correction_prediction, tf.float32),
                                                  mask_flattern)

        accuracy = tf.reduce_sum(weighted_correct_prediction)/mask_sum
        tf.summary.scalar('loss',loss)

    #defines train_op
    with tf.variable_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            logging.info('variable name: %s' %var.name)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),hps.clip_lstm_grads)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

    return ((img_feature, sentence, mask, keep_prob),
            (loss, accuracy, train_op),
            global_step)

placeholders, metrics, global_step = get_train_model(hps,
                                                     vocab_size,
                                                     img_feature_dim)
img_feature, sentence, mask, keep_prob = placeholders
loss, accuracy, train_op = metrics

summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=10)

training_steps = 1000

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(output_dir, sess.graph)
    for i in range(training_steps):
        (batch_img_features,
         batch_sentence_ids,
         batch_weights,_) = caption_data.nex_batch(hps.batch_size)
        input_vals = (batch_img_features,
                      batch_sentence_ids,
                      batch_weights,
                      hps.keep_prob)
        feed_dict = dict(zip(placeholders, input_vals))
        fetches = [global_step, loss, accuracy, train_op]
        should_log = (i+1) % hps.log_frequence == 0
        should_save = (i+1) % hps.save_frequence == 0

        if should_log:
            fetches += [summary_op]

        outputs = sess.run(fetches, feed_dict=feed_dict)
        global_step_val, loss_val, accuracy_val = outputs[0:3]

        if should_log:
            summary_str = outputs[-1]
            writer.add_summary(summary_str, global_step_val)
            logging.info('step: %5d, loss: %3.3f, accu: %3.3f'
                         %(global_step_val, loss_val, accuracy_val))

        if should_save:
            model_save_file = os.path.join(output_dir,'image_caption')
            logging.info('step: %5d, model saved:' %global_step_val)
            saver.save(sess,
                       model_save_file,
                       global_step = global_step_val)
