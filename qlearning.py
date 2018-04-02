# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 20:48:48 2018

@author: yang
calcuate Q-learning in this mod
"""
import pandas as pd
import numpy as np

class Qtable:
    def __int__(self,all_actions,state,alpha=0.01,gamma=0.9,eplison=0.8):
        self.all_actions=all_actions#动作集合
        self.gamma=gamma
        self.alpha=alpha#学习率，默认0.01
        self.eplison=eplison#贪婪率，默认0.8
        self.q_table=pd.Dataframe(colums=self.all_actions,dtype=np.float64)#定义Q表
    #-----------------------我-是-分-隔-线------------------------------
    def choose_action(self,state_now):
        if np.random.uniform<self.eplison:
            #贪婪
            actions=self.q_table.loc[state_now,:]#Q表中取当前状态所有的动作
            actions=actions.reindex(np.random.permuation(actions.index))#重新排列,防止出现过个最值
            action=actions.idxmax()
        else:
            #随机
            actions=np.random.choice(self.all_actions)
        return action
    #-----------------------我-是-分-隔-线------------------------------
    def learn(self,state_now,state_next,action,reward):
        if state_now=='terminal':
            g=reward
        else:
            g=reward+self.gamma*self.q_table.loc[state_next,:].max()
        self.q_table.loc[state_now,action]+=self.alpha*(g-self.q_table[state_now,action])
    #-----------------------我-是-分-隔-线------------------------------
    def check_state_exist(self,state_now):#检测当前状态是否在Q表中，不在就补全Q表
        if state_now not in self.q_table.index:
            self.q_table=self.q_table.append(pd.Series([0]*len(self.all_actions),#对应行补零
                                             index=self.q_table.colums,#补索引
                                             name=state_now)#补索引名
                                             )