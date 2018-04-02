function [action_taken]=chose_action(Epslion,state,goal,step)
% epslion greedy贪婪策略
%1 上/2 下/3 左/4 右/5 不动
ep=rand(1);
%有效动作
switch state
    case 1
        action_valid=[2,4];
    case 5
        action_valid=[2,3];
    case 21
        action_valid=[1,4];
    case {2,3,4}
        action_valid=[2,3,4];
    case {6,11,16}
        action_valid=[1,2,4];
    case {22,23,24}
        action_valid=[1,3,4];
    case {10,15,20}
        action_valid=[1,2,3];
    case {7,8,9,12,13,14,17,18,19}
        action_valid=[1,2,3,4];
end
X=Epslion+step*0.0001;
if ep<X
    %贪婪原则   
    for i=1:length(action_valid)
        [~,r(i)]=Reward(state,action_valid(i),goal);
    end%提取有效动作对应的回报
    action_temp=action_valid(find(r==max(r)));%防止出现多个最值
    action_taken=action_temp(unidrnd(length(action_temp)));
%     action_taken=action_valid(find(max(action_valid)));
else
    %随机原则
    action_taken=action_valid(unidrnd(length(action_valid)));
end

