import tensorflow as tf
import numpy as np
from trademodel.stock import MarketEnv
import random
'''
class test():

    def __init__(self):
        self.state_dim = 60
        self.nodes = 128
        self.test_data = np.random.rand(1000, self.state_dim)
        self.time_step = 0
        self.initial_reply = []
        self.batch_size = 10

        self.create_network()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('saved_network/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
        global summary_writer
        summary_writer = tf.train.SummaryWriter('testlogs', graph=self.session.graph)

    def create_network(self):
        self.state_input = tf.placeholder('float', [None, self.state_dim])
        with tf.name_scope('layer'):
            W = tf.Variable(tf.truncated_normal([self.state_dim, self.nodes]))
            b = tf.Variable(tf.truncated_normal([self.nodes]))
            self.output = tf.nn.sigmoid(tf.matmul(self.state_input, W) + b)
        with tf.name_scope('virtual_layer'):
            W = tf.Variable(tf.truncated_normal([self.nodes, self.state_dim]))
            b = tf.Variable(tf.truncated_normal([self.state_dim]))
            self.virtur_output = tf.nn.sigmoid(tf.matmul(self.output, W) + b)
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.square(self.virtur_output - self.state_input))
            tf.scalar_summary('cost', self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.cost)

    def train_data(self, initial_data):
        self.time_step += 1
        self.initial_reply.append(initial_data)
        if len(self.initial_reply) > 30:
            self.train_network()

    def train_network(self):
        minibatch = random.sample(self.initial_reply, self.batch_size)
        output = self.output.eval(feed_dict={self.state_input: minibatch})
        self.optimizer.run(feed_dict = {
            self.state_input: minibatch
        })
        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.state_input: minibatch
        })
        summary_writer.add_summary(summary_str, self.time_step)
        if self.time_step % 100 ==0:
            self.saver.save(self.session, 'saved_network/mymodel')

aa = test()
for i in aa.test_data:
    aa.train_data(i)
'''
a = tf.placeholder('float', [None, 10])
w = tf.Variable(tf.truncated_normal([10, 3]))
b = tf.nn.sigmoid(tf.matmul(a, w))
h = tf.placeholder('float', [None, 3])
action = tf.Variable([0.20, 0.30, 0.50])
actionprob = tf.squeeze(action)
aa = tf.argmax(actionprob, 0)
actiongather = tf.gather(action, 1)
with tf.Session() as sess:
    np.random.seed(11)
    sess.run(tf.initialize_all_variables())
    print sess.run(actiongather)
    print sess.run(aa)

