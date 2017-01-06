from learningmodel.PG_modelbuild import Modelbuild
from trademodel.stock import MarketEnv
import numpy as np
from pandas import Series
from trademodel import plotting
import collections

initialdata = MarketEnv(dir_path='/home/zhangbw/Documents/stock_data/', stockid='SH600000', parameterc=0.01,
                start_date='2000-1-1', end_date='2015-1-1')
trainingdata = MarketEnv(dir_path='/home/zhangbw/Documents/stock_data/', stockid='SH600000', parameterc=0.01,
                start_date='2000-1-1', end_date='2015-1-1')
testdata = MarketEnv(dir_path='/home/zhangbw/Documents/stock_data/', stockid='SH600000', parameterc=0.01,
                start_date='2000-1-1', end_date='2015-1-1')
# make initial data
initial_data = []
count = 0
while count <= 500:
    initialdata.currentTargetIndex += 1
    initialdata.defineState()
    initial_data.append(initialdata.state)
    count+=1
# make model
data_directory = {}
data_directory['state_dim'] = 60
data_directory['action_dim'] = 3
data_directory['hidden_nodes'] = 128
data_directory['hidden_layers'] = 4
data_directory['parameter_u'] = 1
data_directory['initial_data'] = np.array(initial_data)
data_directory['parameter_n'] = 1
ITERATION = 1000
INITIAL_EPOCH = 50
TRAINING_EPOCH = 500
STEP = 100
BATCH_SIZE = 32
EPOCH = 50000
# make model
agent = Modelbuild(data_directory)


for iter in xrange(ITERATION):
    print iter
    # initialize tase
    # Train
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(EPOCH),
        episode_rewards=np.zeros(EPOCH)
    )
    for episode in xrange(EPOCH):
        transition = collections.namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'done'])
        portfolio = 0
        portfolio_value = 0
        state = trainingdata.reset()
        for step in xrange(STEP):
            episodedata = []
            action_probs1, action1 = agent.policy_forward(state)
            next_state, reward, _ = trainingdata.step(action)
            state = next_state
            action_probs2, action2 = agent.policy_forward(state)
            next_state, reward2, _, _ = trainingdata.step(action2)
            episodedata.append(transition(state = state, action1 = action1, action2=action2, reward = reward2, next_state =
                                          next_state, done = done))
            state = next_state
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = step
            if done:
                for t, transition in enumerate(episodedata):
                    total_return = sum(t.reward for i, t in enumerate(episodedata[t: ]))
                    agent.perceive(transition.states, transition.targets, transition.action)
                if episode % BATCH_SIZE == 0 and episode > 1:
                    agent.train_network()
                break
        if episode % 100 == 0 and episode > 1:
            total_reward = 0
            for i in xrange(10):
                for step in xrange(STEP):
                    action_probs, action = agent.policy_forward(state)
                    next_state, reward, done, info = trainingdata.step(action)
                    # pdb.set_trace();
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / 10
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
    plotting.plot_episode_stats(stats, smoothing_window = 25)
