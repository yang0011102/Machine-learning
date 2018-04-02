clc
clear all
close all
disp('GO')
%初始化参数
alpha=0.1;%学习率
gamma=0.1;%折扣率
goal=25;
Epslion=0.8;%贪婪阈值
max_episodes=2000;%最大步数
%初始化Q表
Q=zeros(25,4);
%--------------------------开始训练------------------------
episode=1;
while episode<max_episodes
    step=1;%计步初始化
    state=unidrnd(24);%随机初始状态
    while state~=goal
        action=chose_action(Epslion,state,goal,step);
        [state_next,R]=Reward(state,action,goal);
        G=gamma*max(Q(state_next,:))-Q(state,action);%估计
        Q(state,action)=Q(state,action)+alpha*(R+G);
        state=state_next;
        step=step+1;
    end
    episode=episode+1;
end
disp('Training Finished')
%--------------------------读Q表----------------------------
state=1;
step=1;
while state~=goal
    path(step)=state;
    [~,act]=max(Q(state,:));
    [s_next,~]=Reward(state,act,goal);
    state=s_next;
    step=step+1;
end
disp(['路径为：',num2str(path)]);