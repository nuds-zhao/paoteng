from random import random
import numpy as np
import math
from pandas import Series
from pandas import DataFrame
import datetime
import pandas

import gym
from gym import spaces


class MarketEnv(gym.Env):
    PENALTY = 1  # 0.999756079

    def __init__(self, dir_path, stockid, parameterc, start_date, end_date, scope=60):
        self.parameter_c = parameterc
        self.stockid = stockid
        self.startDate = start_date
        self.endDate = end_date
        self.scope = scope

        np.random.seed(7)
        self.dir_path = dir_path
        dates = pandas.read_csv(self.dir_path + stockid + '.csv', usecols=[0]).values
        dataset = pandas.read_csv(self.dir_path + stockid + '.csv', usecols=[4])
        dates = [date[0] for date in dates]
        dates = dates[self.scope:]
        zt = [0]
        dataset = dataset.values
        for i in range(1, len(dataset)):
            zt.append(round((dataset[i]-dataset[i-1])[0], 4))
        self.zt = zt[self.scope:]
        dataset = dataset[self.scope:]
        self.fullyzt = zt
        self.Rt = {}
        data = {}
        for i in range(len(dates)):
            data[dates[i]] = dataset[i]
        self.target = data
        self.dataset = dataset
        self.dates = dates
        self.actions = ['long', 'neutral', 'short']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.scope)

        self.reset()
        self._seed()

    def _step(self, action):
        if self.done:
            return self.state, self.Rt[self.currentTargetIndex-1], self.done, {}

        if self.actions[action] == "long":
            self.boughts.append(1.0)
        elif self.actions[action] == "short":
            self.boughts.append(-1.0)
        elif self.actions[action] =='neutral':
            self.boughts.append(0.0)
        else:
            pass

        self.Rt[self.currentTargetIndex] = self.boughts[self.currentTargetIndex-1]*self.zt[self.currentTargetIndex]
        self.currentTargetIndex += 1
        self.defineState()
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
            self.currentTargetIndex]:
            self.done = True

        return self.state, self.Rt[self.currentTargetIndex-1], self.done, {"dt": self.targetDates[self.currentTargetIndex]}

    def _reset(self):
        self.targetDates = sorted(self.target.keys())
        self.currentTargetIndex = 0
        self.boughts = []
        self.done = False
        self.defineState()

        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    def _seed(self):
        return int(random() * 100)

    def defineState(self):

        budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
        size = math.log(max(1., len(self.boughts)), 100)
        position = 1. if sum(self.boughts) > 0 else 0.

        subject = []
        for i in xrange(self.scope):
            try:
                subject.append(self.fullyzt[self.currentTargetIndex + self.scope - 1 - i])
            except Exception, e:
                print self.currentTargetIndex, i, len(self.targetDates)
                self.done = True
        self.state = subject