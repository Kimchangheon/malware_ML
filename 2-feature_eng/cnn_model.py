import tensorflow as tf
import numpy as np
import random
import glob

from PIL import Image
from keras.utils import np_utils


class CNN_tensor():

  def load_images(self):

    nb_classes = 2

    b = glob.glob('../images/normal/*thumb.png')
    total_len = len(b)
    BEN_TRAIN = int(round(total_len * 0.8))
    BEN_TEST = total_len - BEN_TRAIN

    X_train_benign = np.empty((BEN_TRAIN, 28, 28, 1), dtype = "float32")
    y_train_benign = np.empty((BEN_TRAIN,), dtype = "uint8")
    X_test_benign = np.empty((BEN_TEST, 28, 28, 1), dtype = "float32")
    y_test_benign = np.empty((BEN_TEST,), dtype = "uint8")
    cnt = 0

    for i in b:
      im = Image.open(i).convert("L")
      out = im.resize((28,28))
  
      if cnt < BEN_TRAIN: 
        X_train_benign[cnt,:,:,0] = out
        y_train_benign[cnt,] = 0 
      else:
        X_test_benign[cnt-BEN_TRAIN,:,:,0] = out
        y_test_benign[cnt-BEN_TRAIN,] = 0 

      cnt = cnt+1
      if cnt == (BEN_TRAIN+BEN_TEST):
        break

    m = glob.glob('../images/malware/*thumb.png')
    total_len = len(m)
    MAL_TRAIN = int(round(total_len * 0.8))
    MAL_TEST = total_len - MAL_TRAIN

    X_train_malware = np.empty((MAL_TRAIN, 28, 28, 1), dtype = "float32")
    y_train_malware = np.empty((MAL_TRAIN,), dtype = "uint8")
    X_test_malware = np.empty((MAL_TEST, 28, 28, 1), dtype = "float32")
    y_test_malware = np.empty((MAL_TEST,), dtype = "uint8")
    cnt = 0

    for i in m:
      im = Image.open(i).convert("L")
      out = im.resize((28,28))

      if cnt < MAL_TRAIN:
        X_train_malware[cnt,:,:,0] = out
        y_train_malware[cnt,] = 1 # malware
      else:
        X_test_malware[cnt-MAL_TRAIN,:,:,0] = out
        y_test_malware[cnt-MAL_TRAIN,] = 1 # malware

      cnt = cnt+1
      if cnt == (MAL_TRAIN+MAL_TEST):
        break

    X_train = np.empty(((BEN_TRAIN+MAL_TRAIN), 28, 28, 1), dtype = "float32")
    y_train = np.empty(((BEN_TRAIN+MAL_TRAIN),), dtype = "uint8")
    X_test = np.empty(((BEN_TEST+MAL_TEST),  28, 28, 1), dtype = "float32")
    y_test = np.empty(((BEN_TEST+MAL_TEST),), dtype = "uint8")

    y_train_benign = np.zeros(BEN_TRAIN,)
    y_test_benign = np.zeros(BEN_TEST,)

    X_train = np.concatenate((X_train_benign, X_train_malware), axis=0)
    y_train = np.append(y_train_benign, y_train_malware)
    X_test = np.concatenate((X_test_benign, X_test_malware), axis=0)
    y_test = np.append(y_test_benign, y_test_malware)

    X_train = X_train.astype("float32")
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    X_test = X_test.astype("float32")
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    self.x_train  = X_train
    self.x_test   = X_test
    self.y_train  = Y_train
    self.y_test   = Y_test


  def do_cnn(self):

    learning_rate = 0.001
    training_epochs = 30
    batch_size = 100
    keep_prob = tf.placeholder(tf.float32)

    # input place holders
    X = tf.placeholder(tf.float32, [None, 28, 28 ,1])
    Y = tf.placeholder(tf.float32, [None, 2])  

    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

    W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([625]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    W5 = tf.get_variable("W5", shape=[625, 2], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(L4, W5) + b5

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(self.y_train) / batch_size)

        for i in range(total_batch):

            batch_xs = self.x_train[i*batch_size:(i+1)*batch_size]
            batch_ys = self.y_train[i*batch_size:(i+1)*batch_size]

            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}

            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={X: self.x_test, Y: self.y_test, keep_prob: 1})

    return acc
