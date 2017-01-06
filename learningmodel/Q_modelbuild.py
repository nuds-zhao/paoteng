import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import pdb
from learningmodel.modelbuild_helper import *

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
REPLAY_SIZE = 20000  # experience replay buffer size
BATCH_SIZE = 64  # size of minibatch
INITIAL_BATCHSIZE = 10

class Modelbuild():
    # DQN Agent
    def __init__(self, data_dictionary):
        # init some parameters
        self.state_dim = data_dictionary['state_dim']
        self.action_dim = data_dictionary['action_dim']
        self.hidden_nodes = data_dictionary['hidden_nodes']
        self.hidden_layers = data_dictionary['hidden_layers']
        self.parameter_u = data_dictionary['parameter_u']
        self.historic_delta = deque([0,1,0])
        self.initial_reply = deque()
        self.replay_buffer = deque()
        self.initial_timestep = 0
        self.time_step = 0
        self.initial_data = data_dictionary['initial_data']
        self.parameter_n = data_dictionary['parameter_n']
        self.epsilon = 0.05

        self.create_network(data_dictionary)

        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_Q_network/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('logs', graph=self.session.graph)

    def create_network(self, data_dictionary):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # fuzzy repre
        fuzzy_output = self.state_input
        # deep trans
        with tf.name_scope('input_layer'):
            with tf.name_scope('W'):
                W = self.weight_variable([self.state_dim, self.hidden_nodes])
            with tf.name_scope('b'):
                b = self.bias_variable([self.hidden_nodes])
            h_layer = tf.nn.relu(tf.matmul(self.state_input, W) + b)
        for i in xrange(self.hidden_layers):
            h_layer1 = h_layer
            with tf.name_scope('hidden_layer' + str(i)):
                with tf.name_scope('W'):
                    W = self.weight_variable([self.hidden_nodes, self.hidden_nodes])
                with tf.name_scope('b'):
                    b = self.bias_variable([self.hidden_nodes])
                h_layer = tf.nn.relu(tf.matmul(h_layer1, W) + b)
        self.h_layer = h_layer

        # initial

        with tf.name_scope('virtual_layer'):
            self.W_virtual = tf.Variable(tf.truncated_normal([self.hidden_nodes, self.state_dim]))
            b = tf.Variable(tf.truncated_normal([self.state_dim]))
            variable_summaries(self.W_virtual, 'initial/W')
            self.virtual_output = tf.nn.sigmoid(tf.matmul(self.h_layer, self.W_virtual) + b)

        # DDR

        with tf.name_scope('layer_action'):
            with tf.name_scope('W'):
                W = self.weight_variable([self.hidden_nodes, self.action_dim])
            with tf.name_scope('b'):
                b = self.bias_variable([self.action_dim])
        self.Q_value = tf.nn.sigmoid(tf.matmul(self.h_layer, W) + b)

    def create_training_method(self):
        # initial method
        self.initial_cost = tf.reduce_mean(tf.square(self.virtual_output - self.state_input))
        variable_summaries(self.initial_cost, 'initial_cost')
        self.initial_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.initial_cost)

        # training method
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        tf.scalar_summary("loss", self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        self.time_step += 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        self.historic_delta.append(one_hot_action)
        if len(self.historic_delta) > 20000:
            self.historic_delta.popleft()
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > 2000:
            self.train_network()

    def create_initial_data(self, initial_state):
        self.initial_timestep += 1
        self.initial_reply.append(initial_state)
        if len(self.initial_reply) > 200:
            self.initial_method()

    def initial_method(self):
        minibatch = random.sample(self.initial_reply, INITIAL_BATCHSIZE)

        self.initial_optimizer.run(feed_dict={
            self.state_input: minibatch
        })
        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.state_input: minibatch
        })
        summary_writer.add_summary(summary_str, self.initial_timestep)
        # pdb.set_trace()

        # save network every 1000 iteration
        if self.initial_timestep % 100 == 0:
            self.saver.save(self.session, 'saved_Q_network/mymodel')

    def train_network(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        # pdb.set_trace();
        Q_value_batch = self.Ut.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        summary_writer.add_summary(summary_str, self.time_step)
        # pdb.set_trace()

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_Q_network/mymodel')

    def egreedy_action(self, state):
        Q_value = self.Ut.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # print(self.time_step)
        # print(self.epsilon)
        if self.time_step > 200000:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 1000000
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 10000  # Episode limitation
STEP = 9  # Steps in an episode
TEST = 10  # The number of experiment test every 100 episode
ITERATION = 20

