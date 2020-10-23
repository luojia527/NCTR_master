'''

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''

import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
import datetime

import pickle
import NCTR

tf.flags.DEFINE_string("word2vec", "../data/glove.6B.100d.txt", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data","../data/music/music.valid", " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/music/music.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/music/music.train", "Data for training")
tf.flags.DEFINE_string("test_data","../data/music/music.test", " Data for test")

# ==================================================

# Model Hyperparameters
#tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding ")
#tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.2, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularizaion lambda")
tf.flags.DEFINE_float("l2_reg_V", 0.0, "L2 regularizaion V")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size ")#100
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae


def dev_step(i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    #FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")

    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    item_length = para['item_length']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    valid_length = para['valid_length']
    test_length = para['test_length']
    i_text = para['i_text']
    item_voc = para['item_voc']

    np.random.seed(2019)
    random_seed = 2019
   
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = NCTR.NCTR(
                user_num=user_num,
                item_num=item_num,
                item_length=item_length,
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                fm_k=64,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=64)
            tf.set_random_seed(random_seed)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.005, initial_accumulator_value=1e-8).minimize(deep.loss)

            optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            '''optimizer=tf.train.RMSPropOptimizer(0.002)
            grads_and_vars = optimizer.compute_gradients(deep.loss)'''
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW_i = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                word2vec_dic = {}
                index2word = {}
                mean = np.zeros(FLAGS.embedding_dim)
                count = 0
                with open(FLAGS.word2vec, "rb") as f:
                    for line in f:
                        values = line.split()
                        word = values[0]
                        word_vec = np.array(values[1:], dtype = 'float32')
                        word2vec_dic[word] = word_vec
                        mean = mean + word_vec
                        index2word[count] = word
                        count = count + 1
                    mean = mean / count
                
                for word_i in vocabulary_item:
                   if word_i in word2vec_dic:
                       initW_i[vocabulary_item[word_i]] = word2vec_dic[word_i]
                   else:
                       initW_i[vocabulary_item[word_i]] = np.random.normal(mean, 0.1, size=FLAGS.embedding_dim)

                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))
                sess.run(deep.W2.assign(initW_i)) 
                print ("hello=")
                


            l = (train_length / FLAGS.batch_size) + 1
            print (l)
            ll = 0
            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0
            pkl_file = open(FLAGS.train_data, 'rb')

            train_data = pickle.load(pkl_file)

            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(FLAGS.valid_data, 'rb')

            valid_data = pickle.load(pkl_file)
            valid_data = np.array(valid_data)
            pkl_file.close()

            pkl_file = open(FLAGS.test_data, 'rb')

            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()            
            
            data_size_train = len(train_data)
            data_size_valid = len(valid_data)            
            data_size_test = len(test_data)
            print("test size:", data_size_test)
            batch_size = 100
            ll = int(len(train_data) / batch_size)
            print('Stating epoch training')
            for epoch in range(100):
                # Shuffle the data at each epoch

                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, y_batch = zip(*data_train)
                    #print("data_train.shape():", uid, iid, y_batch)

                    i_batch = []
                    for i in range(len(uid)):
                        i_batch.append(i_text[iid[i][0]])
                    i_batch = np.array(i_batch)

                    t_rmse, t_mae = train_step(i_batch, uid, iid, y_batch, batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae

                    if batch_num % 1000 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print (batch_num)
                        loss_valid = 0
                        accuracy_valid = 0
                        mae_valid = 0
                        ll_valid = int(len(valid_data) / batch_size)  # + 1
                        for batch_num2 in range(ll_valid):
                            start_index = batch_num2 * batch_size
                            end_index = min((batch_num2 + 1) * batch_size, data_size_valid)
                            data_valid = valid_data[start_index: end_index]

                            userid_valid, itemid_valid, y_valid = zip(*data_valid)
                            # print("valid_data.shape():", userid_valid, itemid_valid, y_valid)

                            i_valid = []
                            for i in range(len(userid_valid)):
                                i_valid.append(i_text[itemid_valid[i][0]])
                            i_valid = np.array(i_valid)

                            loss, accuracy, mae = dev_step(
                                i_valid, userid_valid, itemid_valid, y_valid)
                            loss_valid = loss_valid + len(i_valid) * loss
                            accuracy_valid = accuracy_valid + len(i_valid) * np.square(
                                accuracy)  # ,delta_user_attention,delta_item_attention  user_feas_weight, item_feas_weight,
                            mae_valid = mae_valid + len(i_valid) * mae
                        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_valid / valid_length,
                                                                                         np.sqrt(
                                                                                             accuracy_valid / valid_length),
                                                                                         mae_valid / valid_length))

                print (str(epoch) + ':\n')
                print("\nEvaluation:")
                print ("train:rmse,mae:", train_rmse / ll, train_mae / ll)
                train_rmse = 0
                train_mae = 0

                loss_test = 0
                accuracy_test = 0
                mae_test = 0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_test, itemid_test, y_test = zip(*data_test)
                    # print("test_data.shape():", userid_test[0])


                    i_test = []
                    for j in range(len(itemid_test)):
                        i_test.append(i_text[itemid_test[j][0]])
                    i_test = np.array(i_test)

                    loss, accuracy, mae = dev_step(
                        i_test, userid_test, itemid_test, y_test)
                    loss_test = loss_test + len(i_test) * loss
                    accuracy_test = accuracy_test + len(i_test) * np.square(accuracy)
                    mae_test = mae_test + len(i_test) * mae

                print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}".format(loss_test / test_length,
                                                                              np.sqrt(accuracy_test / test_length),
                                                                              mae_test / test_length))
                rmse = np.sqrt(accuracy_test / test_length)
                mae = mae_test / test_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print ('best rmse:', best_rmse)
            print ('best mae:', best_mae)

    print ('end')
