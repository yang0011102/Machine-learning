clc
clear all
close all
disp('GO')
%��ʼ������
alpha=0.1;%ѧϰ��
gamma=0.1;%�ۿ���
goal=25;
Epslion=0.8;%̰����ֵ
max_episodes=2000;%�����
%��ʼ��Q��
Q=zeros(25,4);
%--------------------------��ʼѵ��------------------------
episode=1;
while episode<max_episodes
    step=1;%�Ʋ���ʼ��
    state=unidrnd(24);%�����ʼ״̬
    while state~=goal
        action=chose_action(Epslion,state,goal,step);
        [state_next,R]=Reward(state,action,goal);
        G=gamma*max(Q(state_next,:))-Q(state,action);%����
        Q(state,action)=Q(state,action)+alpha*(R+G);
        state=state_next;
        step=step+1;
    end
    episode=episode+1;
end
disp('Training Finished')
%--------------------------��Q��----------------------------
state=1;
step=1;
while state~=goal
    path(step)=state;
    [~,act]=max(Q(state,:));
    [s_next,~]=Reward(state,act,goal);
    state=s_next;
    step=step+1;
end
disp(['·��Ϊ��',num2str(path)]);