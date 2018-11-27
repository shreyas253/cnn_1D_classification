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
loadFile = mainFile + 'pyDat_test.mat'
loaddata = hdf5storage.loadmat(loadFile)
x_test = loaddata['x_test'] # test features
t_test = loaddata['t_test'] # test target labels


## PARAMETERS
residual_channels = 8
filter_width = 5
dilations = [1]
input_channels = X[0][0].shape[2]
no_classes = 3
postnet_channels= 8


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

saveFile1 = mainFile + 'res_test.mat'
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
    saver.restore(sess, model_path)
    
       #x
    no_utt = x_test.shape[0]
    t_test_pred = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        t_test_pred[n_val] = sess.run([y_pred], feed_dict={x: x_test[n_val][0],y:t_test[n_val][0]})        

scipy.io.savemat(saveFile1,{"t_test_pred":t_test_pred})