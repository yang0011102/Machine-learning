function [state_next,R]=Reward(state,action,goal)
%1 上/2 下/3 左/4 右/5 不动

switch action
    case 1 %上
        state_next=state-5;
    case 2 %下
        state_next=state+5;
    case 3 %左
        state_next=state-1;
    case 4 %右
        state_next=state+1;
%     case 5 %不动
%         state_next=state;
end

switch state_next
    case goal
        R=100;
    case {10,13,18,22}
        R=-10;
    otherwise
        R=1;
end