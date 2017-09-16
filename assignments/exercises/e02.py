import tensorflow as tf
import csv
import numpy as np

file_path = '../../data/heart.csv'

# Create the linear regress model
# head 400 as train; others as test
TRAIN_SET_NUM  = 400
TEST_SET_NUM   = 62
BATCH_SIZE = 100
HIDDEN_SIZE= 500
LOGIT_SIZE = 2
class model:
    def __init__(self,file_path,batch_size,input_size,hidden_size,logit_size,lr,train_set_num,test_set_num):
        self.file_path=file_path
        self.batch_size = batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.logit_size=logit_size
        self.lr = lr
        self.train_set_num=train_set_num
        self.test_set_num=test_set_num

    def _create_data_label(self):
        # read data and label from heart.csv
        data = np.array(np.zeros(shape=[462, 9]))
        label = np.array(np.zeros(shape=[462, 1]))
        with open(self.file_path, 'r') as file:
            lines = csv.reader(file)
            for idx, row in enumerate(lines):
                if (idx > 0):
                    for index, i in enumerate(row):
                        if (i == 'Present'):
                            i = 1.0
                        elif (i == 'Absent'):
                            i = 0.0
                        if (index >= 0 and index < 9):
                            data[idx - 1, index] = float(i)
                        elif (index == 9):
                            label[idx - 1, 0] = float(i)
        label = tf.one_hot(label, 2, 1, 0, -1)
        self.x_train = data[0:self.train_set_num,:]
        self.x_test  = data[self.train_set_num:self.train_set_num+self.test_set_num,:]
        self.y_train = label[0:self.train_set_num, :]
        self.y_test =  label[self.train_set_num:self.train_set_num + self.test_set_num, :]
    def _create_variable(self):
        self.X = tf.placeholder(tf.float32,shape=[self.batch_size,self.input_size])
        self.y = tf.placeholder(tf.float32,shape=[self.batch_size,self.out_size])

        self.W1 = tf.Variable(tf.truncated_normal(shape=[self.input_size,self.hidden_size],))
        self.b1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))

        self.W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.logit_size]))
        self.b2 = tf.Variable(tf.zeros(shape=self.logit_size))

    def _create_optimizer(self):
        hidden = tf.matmul(self.X,self.W1)+self.b1
        logit = tf.matmul(hidden,self.W2)+self.b2
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logit))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def _create_graph(self):
        self._create_data_label()
        self._create_variable()
        self._create_optimizer()

    def _train_model(self,epoch,print_steps=3):
        for e in range(epoch):
            for step in range(self.train_set_num/self.input_size)
                with tf.Session() as sess:
                    feed_dict = {self.X:self.x_train[(step)*self.batch_size:(step+1)*self.batch_size],
                                 self.y:self.y_train[(step)*self.batch_size:(step+1)*self.batch_size]}
                    sess.run(self.optimizer,feed_dict=feed_dict)


    def _test_model(self):