__author__ = "Shreyas Seshadri, shreyas.seshadri@aalto.fi"
import numpy as np
import tensorflow as tf
import time
import hdf5storage

tf.reset_default_graph() # debugging, clear all tf variables
#tf.enable_eager_execution() # placeholders are not compatible

import model_convNet 
import scipy.io


_FLOATX = tf.float32 



## LOAD DATA
mainFile = '/Users/seshads1/Documents/code/ACLEW/cnn_wavenet_like/1/'
#mainFile = '/l/seshads1/code/syllCount/syll_cout/'
loadFile = mainFile + 'pyDat.mat'
loaddata = hdf5storage.loadmat(loadFile)
X = loaddata['X_train'] # training features
T = loaddata['T_train'] # target labels
x_val = loaddata['x_val'] # validation features
t_val = loaddata['t_val'] # validation target labels


## PARAMETERS
residual_channels = 256
filter_width = 5
dilations = [1, 1, 1, 1, 1, 1, 1]
input_channels = X[0][0].shape[2]
no_classes = 3
postnet_channels= 256


S = model_convNet.CNET(name='S', 
                       input_channels=input_channels,
                       output_channels = no_classes,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=None,
                       do_postproc=True,
                       do_GU=True)


# optimizer parameters
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

num_epochs = 3#3#200

# data placeholders of shape (batch_size, timesteps, feature_dim)
x = tf.placeholder(shape=(None, None, input_channels), dtype=_FLOATX)
y = tf.placeholder(shape=(None, None), dtype=tf.int32)
logits = S.forward_pass(x)

#y = tf.one_hot(y,no_classes,axis = -1)

#loss_function
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,logits))
y_pred = tf.nn.softmax(logits)

#optimization
opt=tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(loss)

#initialize variables
init=tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)

saveFile1 = mainFile + 'res_train.mat'
model_path = mainFile + 'model.ckpt'

saver = tf.train.Saver()
epoch=0
dontStop=1
loss_val = np.ones((num_epochs,1)) * np.inf
stpCrit_win = 5
stpCrit_min = 30

with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 
    
    while (epoch<num_epochs) & (dontStop):
        
        # Train discriminator
        idx = np.random.permutation(X.shape[0])
        saver = tf.train.Saver(max_to_keep=0)
        t = time.time()
        for batch_i in range(X.shape[0]):            
            _, lossD = sess.run([opt,loss], feed_dict={x: X[idx[batch_i]][0], y: T[idx[batch_i]][0]})
            elapsed = time.time() - t       
            print("Errors for epoch %d, batch %d: %f, and took time: %f" % (epoch, batch_i,lossD, elapsed)) 
        
        # validation        
        no_utt = x_val.shape[0]
        loss_test = np.ones((no_utt,1))
        for n_val in range(no_utt):
            loss_test[n_val] = sess.run([loss], feed_dict={x: x_val[n_val][0],y:t_val[n_val][0]})  
        loss_val[epoch] = np.mean(loss_test)   
        if epoch>stpCrit_min:
            tmp = loss_val[epoch:epoch-stpCrit_win-1:-1]
            tmp = tmp[1:]-tmp[0]
            if ((tmp < 0).sum() == tmp.size).astype(np.int):
                dontStop = 0
              
        print("Validation errors for epoch %d: %f , and took time: %f" % (epoch, loss_val[epoch], elapsed))
        epoch += 1            
    save_path = saver.save(sess, model_path)
    
    #X
    no_utt = X.shape[0]
    t_train_pred = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        t_train_pred[n_val] = sess.run([y_pred], feed_dict={x: X[n_val][0],y:T[n_val][0]})
    
    #x
    no_utt = x_val.shape[0]
    t_val_pred = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        t_val_pred[n_val] = sess.run([y_pred], feed_dict={x: x_val[n_val][0],y:t_val[n_val][0]})        

scipy.io.savemat(saveFile1,{"t_train_pred":t_train_pred,"t_test_pred":t_val_pred})