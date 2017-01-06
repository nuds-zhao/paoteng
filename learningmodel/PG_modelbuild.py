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

        self.create_policy_network(data_dictionary)

        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_PG_network/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('logs', graph=self.session.graph)

    def create_policy_network(self, data_dictionary):
        # Policy Estimator
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        self.next_state_input = tf.placeholder("float", [None, self.state_dim])
        self.action_input = tf.placeholder(tf.int32, [None, self.action_dim])
        self.delta_t_1 = tf.placeholder('float', [None, self.action_dim])
        with tf.name_scope('Policy_Estimator'):
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

            # initial

            with tf.name_scope('virtual_layer'):
                self.W_virtual = tf.Variable(tf.truncated_normal([self.hidden_nodes, self.state_dim]))
                b = tf.Variable(tf.truncated_normal([self.state_dim]))
                variable_summaries(self.W_virtual, 'initial/W')
                self.virtual_output = tf.nn.sigmoid(tf.matmul(h_layer, self.W_virtual) + b)

            # DDR

            self.delta_t = tf.nn.softmax(tf.matmul(h_layer, W) + b + self.parameter_u * self.delta_t_1)
            self.action_probs = tf.squeeze(self.delta_t)

            self.action = tf.argmax(self.delta_t, 0)
            self.picked_action_prob = tf.gather(self.action_probs, self.action_input)


    def create_training_method(self):
        # initial method
        self.initial_cost = tf.reduce_mean(tf.square(self.virtual_output - self.state_input))
        variable_summaries(self.initial_cost, 'initial_cost')
        self.initial_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.initial_cost)
        # training method
        self.y_input = tf.placeholder('float', [None, self.action_dim])
        self.loss = tf.reduce_mean(-tf.log(self.picked_action_prob) * self.y_input)
        tf.scalar_summary("loss", self.loss)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def perceive(self, states, targets, action):
        self.time_step += 1
        temp = []
        for index, value in enumerate(states):
            temp.append([states[index], targets[index], action[index]])
        self.replay_buffer += temp

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
            self.saver.save(self.session, 'saved_PG_network/mymodel')

    def train_network(self):

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        targets_batch = [data[1] for data in minibatch]
        action_batch = [data[2] for data in minibatch]

        self.optimizer.run(feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch,
            self.y_input: targets_batch
        })
        summary_str = self.session.run(merged_summary_op, feed_dict={
            self.state_input:state_batch,
            self.y_input: targets_batch,
            self.action_input: action_batch
        })
        summary_writer.add_summary(summary_str, self.time_step)
        # pdb.set_trace()

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_PG_network/mymodel')

    def policy_forward(self, state):
        prob = self.delta_t_1.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # print(self.time_step)
        # print(self.epsilon)
        if self.time_step > 200000:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 1000000
        if random.random() <= self.epsilon:
            action = np.random.choice(self.action_dim, 1)[0]
        else:
            action = np.random.choice(self.action_dim, 1, p=prob)[0]
        y = np.zeros([self.action_dim])
        self.time_step += 1
        y[action] = 1
        return y, action

    def action(self, state):
        prob = self.delta_t_1.eval(feed_dict = {self.state_input:[state]})[0]
        action = np.argmax(prob)
        y = np.zeros([self.action_dim])
        y[action] = 1
        return y, action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    def discounted_rewards(self,rewards):
        reward_discounted = np.zeros_like(rewards)
        track = 0
        for index in reversed(xrange(len(rewards))):
            track = track * GAMMA + rewards[index]
            reward_discounted[index] = track
        return reward_discounted
# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 10000  # Episode limitation
STEP = 9  # Steps in an episode
TEST = 10  # The number of experiment test every 100 episode
ITERATION = 20

