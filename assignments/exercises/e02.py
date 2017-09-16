import tensorflow as tf
import csv
import numpy as np

FILE_PATH = '../../data/heart.csv'
# Create the linear regress model
# head 400 as train; others as test
TRAIN_SET_NUM  = 400
TRAIN_SET_DIM  = 9
TEST_SET_NUM   = 62

BATCH_SIZE = 100
HIDDEN_SIZE= 4
LOGIT_SIZE = 2
LEARNING_RATE = 0.1
class lrmodel:
    def __init__(self,file_path,batch_size,train_set_dim,hidden_size,logit_size,lr,train_set_num,test_set_num):
        self.file_path=file_path
        self.batch_size = batch_size
        self.train_set_dim=train_set_dim
        self.hidden_size=hidden_size
        self.logit_size=logit_size
        self.lr = lr
        self.train_set_num=train_set_num
        self.test_set_num=test_set_num
        self.data_set_num = self.train_set_num+self.test_set_num

    def _create_data_label(self):
        # read data and label from heart.csv
        data = np.array(np.zeros(shape=[self.data_set_num, self.train_set_dim]))
        #label is in one hot style
        label = np.array(np.zeros(shape=[self.data_set_num, self.logit_size]))
        with open(self.file_path, 'r') as file:
            lines = csv.reader(file)
            for idx, row in enumerate(lines):
                if (idx > 0):
                    for index, i in enumerate(row):
                        if (i == 'Present'):
                            i = 1.0
                        elif (i == 'Absent'):
                            i = 0.0
                        if (index >= 0 and index < self.train_set_dim):
                            data[idx - 1, index] = float(i)
                        elif (index == self.train_set_dim):
                            label[idx - 1, int(i)] = 1

        self.x_train = data[0:self.train_set_num,:]
        self.x_test  = data[self.train_set_num:self.train_set_num+self.test_set_num,:]
        self.y_train = label[0:self.train_set_num, :]
        self.y_test =  label[self.train_set_num:self.train_set_num + self.test_set_num, :]
        # because there is noly 62 element in test_set, add 0 to test_set
        self.x_test = np.row_stack((self.x_test,np.zeros([39,9])))
        self.y_test = np.row_stack((self.y_test,np.zeros([39,2])))


    def _create_variable(self):
        self.X = tf.placeholder(tf.float32,shape=[self.batch_size,self.train_set_dim])
        self.y = tf.placeholder(tf.float32,shape=[self.batch_size,self.logit_size])

        self.W1 = tf.Variable(tf.truncated_normal(shape=[self.train_set_dim,self.hidden_size],stddev=0.1))
        self.b1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))

        self.W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.logit_size],stddev=0.1))
        self.b2 = tf.Variable(tf.zeros(shape=self.logit_size))

    def _create_optimizer(self):
        hidden = tf.matmul(self.X,self.W1)+self.b1
        logit = tf.matmul(hidden,self.W2)+self.b2
        # logit = tf.nn.relu(logi)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=logit)
        self.loss = tf.reduce_mean(entropy)
        # for train
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # for train_acc
        preds = tf.nn.softmax(logits=logit)
        correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(self.y,1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        # for debug
        self.debug = correct_preds


    def build_graph(self):
        self._create_data_label()
        self._create_variable()
        self._create_optimizer()

    def train_model(self,epoch,print_steps=4, test=1):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)
            sess.run(tf.global_variables_initializer())
            for e in range(epoch):
                total_loss = 0
                total_train_acc=0
                for step in range(4):
                    xxx = self.x_train[step * self.batch_size: (step + 1) * self.batch_size,:]
                    yyy = self.y_train[step * self.batch_size: (step + 1) * self.batch_size,:]
                    feed_dict = {self.X:xxx, self.y:yyy}

                    [loss,_,debug,train_acc]=sess.run([self.loss,self.optimizer,self.debug,self.accuracy],feed_dict=feed_dict)
                    # debug
                    # print(debug)
                    total_loss+=loss
                    total_train_acc+=train_acc
                if(1):
                    print("epoch:{} loss:{} train_acc:{}".format(e,total_loss,total_train_acc/400))
                    pass
                # if(test==1):
                #     xxx = self.x_test[0:-1, :]
                #     yyy = self.y_test[0:-1, :]
                #     feed_dict = {self.X: xxx, self.y: yyy}
                #     [acc] = sess.run([self.accuracy], feed_dict=feed_dict)
                #     # print("test accuracy: {.3f}".format(acc/self.test_set_num))
            writer.close()



    def test_model(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            xxx = self.x_test[0:-1, :]
            yyy = self.y_test[0:-1, :]
            feed_dict = {self.X: xxx, self.y: yyy}
            [acc] = sess.run([self.accuracy],feed_dict=feed_dict)
            print("test accuracy: {}".format(acc/self.test_set_num))
        pass

def main():
    model = lrmodel(FILE_PATH,BATCH_SIZE,TRAIN_SET_DIM,HIDDEN_SIZE,LOGIT_SIZE,LEARNING_RATE,TRAIN_SET_NUM,TEST_SET_NUM)
    model.build_graph()
    model.train_model(epoch=100,test=1)
    # model.test_model()

if __name__ == '__main__':
    main()