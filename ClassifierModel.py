import tensorflow as tf
import numpy as np

def fully_conn(input,output):
    ac_fn = None
    #weights_i = tf.random_normal_initializer(0.0, 0.02)
    #return tf.contrib.layers.fully_connected(input,output,activation_fn=ac_fn,weights_initializer=weights_i)
    return tf.contrib.layers.fully_connected(input,output,activation_fn=ac_fn)

def batch_norm(input,is_training):
    return tf.contrib.layers.batch_norm(input,decay=0.9,center=True,scale=True,updates_collections=None,is_training=is_training)


class BaseModel(object):
    def __init__(self):
        self.num_epoch = 40
        self.batch_size = 128
        self.log_step = 200
        self._build_model()

    def _model(self):
        print('-' * 5 + '  Sample model  ' + '-' * 5)

        print('intput layer: ' + str(self.X.get_shape()))

        with tf.variable_scope('fc1'):
            self.fc12 = fully_conn(self.X, 16)
            self.relu12 = tf.nn.relu(self.fc12)
            # self.relu1 = tf.nn.tanh(self.fc1)
            print("fc1 layer:" + str(self.relu12.get_shape()))
            # print('conv1 layer: ' + str(self.pool1.get_shape()))

        with tf.variable_scope('fc2'):
            self.fc2 = fully_conn(self.relu12, 8)
            self.relu2 = tf.nn.relu(self.fc2)
            print('h1r layer: ' + str(self.relu2.get_shape()))

        with tf.variable_scope('fc3'):

            self.fc4inp = tf.concat([self.relu2, self.feat], axis=1)
            self.fc3 = fully_conn(self.fc4inp, 3)
            print('fc3 layer: ' + str(self.fc3.get_shape()))

        # Return the last layer
        return self.fc3

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 33])
        self.feat = tf.placeholder(tf.float32, [None, 2])
        # self.f2 = tf.placeholder(tf.float32, [None, 3])
        self.Y = tf.placeholder(tf.int64, [None])



    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 200, 0.95, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_op, global_step=global_step)


    def _loss(self, labels, logits):
        # Softmax cross entropy loss 'self.loss_op'
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


    def _build_model(self):
        # Define input variables
        self._input_ops()

        # Convert Y to one-hot vector
        labels = tf.one_hot(self.Y, 3)

        # Build a model and get logits
        logits = self._model()

        # Compute loss
        self._loss(labels, logits)

        # Build optimizer
        self._build_optimizer()

        # Compute accuracy
        predict = tf.argmax(logits, 1)
        self.pred = predict
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        epochList = []
        num_training = X_train.shape[0]
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                # print(X_train.shape)
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size, :33]
                # print(X_.shape)
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]
                f_ = X_train[i * self.batch_size:(i + 1) * self.batch_size, 33:]
                # print(f_.shape)

                # feed_dict = {self.X1:X_[:,0,:],self.X2:X_[:,1,:],self.X3:X_[:,2,:],self.Y:Y_,self.feat:f_}
                feed_dict = {self.X: X_, self.Y: Y_, self.feat: f_}
                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                # print(loss)
                losses.append(loss)
                accuracies.append(accuracy)
                # print(score)

                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                          (step, loss, accuracy))
                    # print("score is:- ",score)
                step += 1

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy, _ = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_iter = 0
        pred_list = []
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size, :33]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]
            f_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size, 33:]


            feed_dict = {self.X: X_, self.Y: Y_, self.feat: f_}

            fetches = [self.accuracy_op, self.pred]
            accuracy, preds = sess.run(fetches, feed_dict=feed_dict)
            pred_list = pred_list + list(preds)
            eval_accuracy += accuracy
            eval_iter += 1
        ###Remaining data
        X_ = X_eval[(i + 1) * self.batch_size:, :33]
        Y_ = Y_eval[(i + 1) * self.batch_size:]
        f_ = X_eval[(i + 1) * self.batch_size:, 33:]

        # feed_dict = {self.X1:X_[:,0,:],self.X2:X_[:,1,:],self.X3:X_[:,2,:],self.Y:Y_,self.feat:f_}
        feed_dict = {self.X: X_, self.Y: Y_, self.feat: f_}

        fetches = [self.accuracy_op, self.pred]
        accuracy, preds = sess.run(fetches, feed_dict=feed_dict)
        pred_list = pred_list + list(preds)
        eval_accuracy += accuracy
        eval_iter += 1

        return eval_accuracy / eval_iter, pred_list
