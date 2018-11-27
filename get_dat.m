close all; clear;


batchSize = 32;
timesteps = 400;
dim = 10;
noClasses = 3;

%train
l_tr = 10;
X_train = cell(l_tr,1);
T_train = cell(l_tr,1);
for i = 1:l_tr
    X_train{i} = rand([batchSize,timesteps,dim]);
    T_train{i} = randi(noClasses,batchSize,timesteps,'uint32')-1;
end
    
%val
l_vl = 10;
x_val = cell(l_vl,1);
t_val = cell(l_vl,1);
for i = 1:l_vl
    x_val{i} = rand([batchSize,timesteps,dim]);
    t_val{i} = randi(noClasses,batchSize,timesteps,'uint32')-1;
end
save pyDat.mat X_train T_train x_val t_val  

%test
l_te = 10;
x_test = cell(l_te,1);
t_test = cell(l_te,1);
for i = 1:l_te
    x_test{i} = rand([batchSize,timesteps,dim]);
    t_test{i} = randi(noClasses,batchSize,timesteps,'uint32')-1;
end
save pyDat_test.mat x_test t_test
    
