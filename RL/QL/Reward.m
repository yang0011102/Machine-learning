function [state_next,R]=Reward(state,action,goal)
%1 ��/2 ��/3 ��/4 ��/5 ����

switch action
    case 1 %��
        state_next=state-5;
    case 2 %��
        state_next=state+5;
    case 3 %��
        state_next=state-1;
    case 4 %��
        state_next=state+1;
%     case 5 %����
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