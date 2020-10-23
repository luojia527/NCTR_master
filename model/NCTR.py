'''
@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''


import tensorflow as tf


class NCTR(object):
    def __init__(
            self, item_length,item_vocab_size,fm_k,n_latent,user_num,item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,l2_reg_V=0.0):
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)



        pooled_outputs_i = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, item_length- filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i,3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_i= tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):
            Wi = tf.get_variable(
                "Wi",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            print("self.h_drop_i:", self.h_drop_i)
            self.i_fea = tf.nn.tanh(tf.matmul(self.h_drop_i, Wi) + bi)
            #self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        with tf.name_scope('ncf'):
            iidmf = tf.Variable(tf.random_uniform([item_num , n_latent], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num, n_latent], -0.1, 0.1), name="uidmf")
            self.uid = tf.nn.embedding_lookup(uidmf,self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf,self.input_iid)

            self.uid = tf.reshape(self.uid,[-1,n_latent])
            self.iid = tf.reshape(self.iid,[-1,n_latent])

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.i_feas = tf.multiply(self.iid, self.i_fea)
            print("，self.u_bias，self.i_feas:", self.uid, self.i_feas)
            self.FM = tf.concat([self.uid, self.i_feas], 1)

            #self.FM = tf.multiply(self.u_fea, self.i_feas)
            print("self.FM:", self.FM)


            Wii = tf.get_variable(
                "Wii",
                shape=[n_latent*2, n_latent*2],
                initializer=tf.contrib.layers.xavier_initializer())

            bii = tf.Variable(tf.constant(0.1, shape=[n_latent*2]), name="bii")
            self.FM = tf.nn.relu(tf.matmul(self.FM, Wii) + bii)
            #self.FM = tf.nn.relu(self.FM)

            Wi1 = tf.get_variable(
                "Wi1",
                shape=[n_latent * 2, n_latent * 2],
                initializer=tf.contrib.layers.xavier_initializer())

            bi1 = tf.Variable(tf.constant(0.1, shape=[n_latent * 2]), name="bi1")
            self.FM = tf.nn.relu(tf.matmul(self.FM, Wi1) + bi1)

            Wi2 = tf.get_variable(
                "Wi2",
                shape=[n_latent * 2, 1],
                initializer=tf.contrib.layers.xavier_initializer()) #n_latent * 2

            bi2 = tf.Variable(tf.constant(0.1, shape=[n_latent * 2]), name="bi2")
            #self.FM = tf.nn.relu(tf.matmul(self.FM, Wi2) + bi2)
            self.mul = tf.nn.relu(tf.matmul(self.FM, Wi2)+ bi2)


            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            #Wmul = tf.Variable(tf.random_uniform([n_latent*2, 1], -0.1, 0.1), name='wmul')
            #self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)


            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            l2_loss2 = 0
            l2_loss2 += tf.nn.l2_loss(self.uidW2)
            l2_loss2 += tf.nn.l2_loss(self.iidW2)
            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

