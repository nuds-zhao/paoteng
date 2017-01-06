from trademodel.stock import MarketEnv
import learningmodel.kmeans as kmeans
from learningmodel.Q_modelbuild import Modelbuild
import numpy as np
from pandas import Series

initialdata = MarketEnv(dir_path='/home/zhangbw/Documents/stock_data/', stockid='SH600000', parameterc=1,
                start_date='2000-1-1', end_date='2015-1-1')
trainingdata = MarketEnv(dir_path='/home/zhangbw/Documents/stock_data/', stockid='SH600000', parameterc=1,
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
data_directory['train_data'] = trainingdata
model = Modelbuild(data_directory)
INITIAL_EPOCH = 50
TRAINING_EPOCH = 500
# initial network
'''
for _ in range(INITIAL_EPOCH):
    for i in data_directory['initial_data']:
        model.create_initial_data(i)
'''
# train network
for _ in range(TRAINING_EPOCH):
    Done = False
    reward = 0
    input_state = trainingdata.reset()
    total_reward = []
    total_action = []
    while Done == False:
        input_dim = input_state
        action = model.egreedy_action(input_dim)
        input_state, Rt, done, info = trainingdata.step(action)
        reward += Rt
        total_reward.append(Rt)
        total_action.append(action)
        # save reward, action
        rewardseries = Series(total_reward)
        rewardseries.to_csv('/home/zhangbw/Documents/results/reward.csv')
        actionseries = Series(total_action)
        actionseries.to_csv('/home/zhangbw/Documents/results/action.csv')
        model.perceive(input_dim, action, reward, input_state, done)


