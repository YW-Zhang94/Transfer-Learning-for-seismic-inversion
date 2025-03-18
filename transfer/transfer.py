import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import random
import math
#import segyio
import tensorflow as tf
import keras
from keras import Model
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import UpSampling2D, Conv2D, ZeroPadding2D, Reshape, Concatenate, Dropout , MaxPooling2D, Flatten
from keras.layers import Dense, Input, LeakyReLU, InputLayer
from keras import regularizers
from keras import backend as K
from keras.callbacks import TensorBoard
#from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint


###bandpass filter
def trace_fcos(f1,f2,f3,f4,nt,dt,tr):
    #f1=2
    #f2=8
    #f3=18
    #f4=36

    #nt=301
    #dt=0.002
    df=1/((nt-1)*dt)

    if nt%2 == 0: nth = int((nt-2)/2)
    if nt%2 == 1: nth = int((nt-1)/2)

    if1=math.floor(f1/df)+1
    if2=math.floor(f2/df)+1
    if3=math.floor(f3/df)+1
    if4=math.floor(f4/df)+1

    filt = np.zeros(nt)
    filt[0:nth+1] = 1

    if if1>1: filt[0] = 0

    if if2>2 and if2>if1:
        if if1>0:
            filt[0:if1] = 0
            istart=if1-1
        else:
            istart=0

        #print(istart,if2)
        tap = np.sin(math.pi/2*(np.array(list(range(istart,if2)))-if1)/(if2-if1))
        filt[istart:if2] = filt[istart:if2] * (np.array(tap).T ** 2)

    if if1<nth+1 and if1<if4:
        if if4<nth+1:
            filt[if4-1:nth+1] = 0
            iend=if4
        else:
            iend=nth+1

        tap = np.cos(math.pi/2*(np.array(list(range(if3,iend)))-if3)/(if4-if3))
        filt[if3:iend] = filt[if3:iend] * (np.array(tap).T ** 2)

    filt[nt-nth:nt] = filt[nth+1:1:-1]
    ftr = np.fft.fft(tr)*filt
    trf = np.fft.ifft(ftr).real


###read waveform
nx=72
nz=301
dl=0.5
X=[]

Vz_bin = np.fromfile('../Data/A11v8/bin_A11v8_Nm8192_Nr72_Nt301_Ns3/Vz.bin',dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(8192, nx, nz, 3, order='F').copy()
X=Vz


###bandpass filter
f1=2
f2=8
f3=18
f4=30

nt=361
dt=0.002

tap1 = np.sin(math.pi/2*np.array(list(range(30)))/30)
tap2 = np.cos(math.pi/2*np.array(list(range(30)))/30)

X_test_filtered = []
for i in range(len(X_test)):
    if i%5000 == 0: print(i)
    xx=[]
    for ii in range(len(X_test[i])):
        x=[]
        for iii in range(3):
            t=X_test[i,ii,:,iii]
            t[:30]=tap1*t[:30]
            t[-30:]=tap2*t[-30:]
            t=np.pad(t,(30,30),mode='constant',constant_values=0)
            t=trace_fcos(f1,f2,f3,f4,nt,dt,t)
            x.append(t/max(abs(t)))
        xx.append(x)
    X_test_filtered.append(xx)
X_test_filtered=np.array(X_test_filtered).transpose(0,1,3,2)[:,:,30:-30,:]
X_transfer_train=X_test_filtered[:1024].copy()
X_transfer_unseen=X_test_filtered[2000:2100].copy()
print(X_transfer_train.shape)
print(X_transfer_unseen.shape)

###add 25% noise
np.random.seed(48)
noise=np.random.rand(1024,72,301,3)*0.25
X_transfer_train_noise=X_transfer_train+noise
print(X_transfer_train_noise.shape)


###read mask
nx=241
nz=81
dl=0.5
Y_mask = []
mask_bin = np.fromfile('../Data/A11v8/models_masks/mask_combined_smth.bin',dtype=np.float32, count=-1, sep='')
Y_mask = mask_bin.reshape(8192, nx, nz, 1, order='F').copy()
print(np.array(Y_mask).shape)

Y_mask=np.array(Y_mask[:,49:191,0:41])
print(np.array(Y_mask).shape)

Y_mask[Y_mask<0]=0
for i in range(len(Y_mask)):
    if Y_mask[i].max()>1.8:
        Y_mask[i]=Y_mask[i]/Y_mask[i].max()
Y_mask[Y_mask>1]=1

Y_mask=np.concatenate((Y_mask,1.0-Y_mask),axis=-1)

Y_mask_transfer_train=Y_mask[:1024]
Y_mask_transfer_unseen=Y_mask[2000:2100]
print(Y_mask_transfer_train.shape)
print(Y_mask_transfer_unseen.shape)


###read Vs model
Y_velocity=[]
Vs_bin = np.fromfile('../Data/A11v8/models_masks/vs_combined.bin',dtype=np.float32, count=-1, sep='')
Y_velocity = Vs_bin.reshape(8192, nx, nz, 1, order='F').copy()/100
Y_velocity=np.array(Y_velocity[:,49:191,0:41])
Y_velocity_transfer_train=Y_velocity[:1024]
Y_velocity_transfer_unseen=Y_velocity[2000:2100]
print(Y_velocity_transfer_train.shape)
print(Y_velocity_transfer_unseen.shape)


###CNN model
input_shape=Input(shape=(72, 301, 3),name='in_sh')

#conv 1-1 (72,301) to (72,301)
model=Conv2D(kernel_size=(3,3), filters=64, 
             strides=(1), padding='same')(input_shape)
model=LeakyReLU(alpha=0.05)(model)

#conv 1-2 (72,301) to (72,299)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=64, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 2-1 (72,299) to (36,149)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=128, 
             strides=(2))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 2-2 (36,149) to (36,147)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=128, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 3-1 (36,147) to (18,73)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(2))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 3-2 (18,73) to (18,71)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 4-1 (18,71) to (9,35)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(2))(model)
model=LeakyReLU(alpha=0.05)(model)

#conv 4-2 (9,35) to (9,32)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,4), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

model=Flatten()(model)
model=Dense(32)(model)
model=Dense(73728)(model)
model=Reshape((9,32,256))(model)

#deconv 1-1 (9,32) to (18,30)
model=UpSampling2D(size=(2,1))(model)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#deconv 1-2 (18,30) to (18,28)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#deconv 2-1 (18,28) to (36,26)
model=UpSampling2D(size=(2,1))(model)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#deconv 2-2 (36,26) to (36,24)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=256, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)


#deconv 3-1 (36,26) to (72,24)
model=UpSampling2D(size=(2,1))(model)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=128, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#deconv 3-2 (72,24) to (72,22)
model=ZeroPadding2D(padding=(1,1))(model)
model=Conv2D(kernel_size=(3,3), filters=128, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#deconv 4-1 (72,22) to (142,42)
model=UpSampling2D(size=(2,2))(model)
model=ZeroPadding2D(padding=(1,0))(model)
model=Conv2D(kernel_size=(3,3), filters=128, 
             strides=(1))(model)
model=LeakyReLU(alpha=0.05)(model)

#output 1
output_1=ZeroPadding2D(padding=(0,1))(model)
output_1=Conv2D(kernel_size=(3,4), filters=64, 
             strides=(1))(output_1)
output_1=LeakyReLU(alpha=0.05)(output_1)
output_1=Conv2D(kernel_size=(1,1),filters=2,
                strides=(1),activation='softmax')(output_1)

#output 2
output_2=ZeroPadding2D(padding=(0,1))(model)
output_2=Conv2D(kernel_size=(3,4), filters=64, 
             strides=(1))(output_2)
output_2=LeakyReLU(alpha=0.05)(output_2)
output_2=Conv2D(kernel_size=(1,1),filters=1,
                strides=(1))(output_2)

model_final = Model(inputs=[input_shape], outputs=[output_1, output_2],name='model_anomaly')
model_final.compile(loss=[keras.losses.categorical_crossentropy],
                    optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                    metrics=['accuracy'])
model_final.summary()


###transfer learning
model_final.load_weights('../models/paper_before_transfer.h5')
batch_size=64
batch_size_val=64
def generator(xx,yy1,batch_size):
    num_samples = len(xx)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield xx[batch_indices], yy1[batch_indices]
            
def generator_val(xxv,yyv1,batch_size_val):
    num_samples = len(xxv)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size_val):
            end_idx = min(start_idx + batch_size_val, num_samples)
            #print(start_idx,end_idx)
            batch_indices = indices[start_idx:end_idx]
            yield xxv[batch_indices], yyv1[batch_indices]
              
H=model_final.fit(generator(X_transfer_train_noise,Y_velocity_transfer_train,batch_size), 
                epochs=1, 
                steps_per_epoch=len(X_transfer_train_noise)//batch_size)

model_final.save_weights('CNN_after_transfer.h5')








