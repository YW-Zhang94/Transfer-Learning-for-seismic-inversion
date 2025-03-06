import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
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

    return trf


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
model_final.load_weights('../models/CNN_paper_before_transfer.h5')


###Figure 5 in the paper
#read waveform
#n=1 for Figure 5a, b and c
#n=29 for Figure 5g, e and f
n=1
nx=72
nz=301
dl=0.5
Vz_bin = np.fromfile('../Data/bin_A12v1_02_Nm8192_Nr72_Nt301_Ns3/Vz.bin',dtype=np.float32, count=-1, sep='') 
Vz = Vz_bin.reshape(8192, nx, nz, 3, order='F').copy()
X_test=np.array(Vz)
print(np.array(X_test).shape)

#bandpass filter
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
    if i%10000 == 0: print(i)
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
print(X_test_filtered.shape)

#read Vs model of Figure 5a, b, and c
r='../Data/A12v1_8192mod_no_Fault_FS72recs_T0.6_SmthMask/Rand_Models_Masks_smthMask/models_with_void_02/'+str(n).zfill(6)+'/proc000000_vs.bin'
nx=241
nz=81
Vz_bin = np.fromfile(r,dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(1, nx, nz, 1, order='F').copy()/100
Vs_1=np.array(Vz[:,49:191,0:41])
print(np.array(Vs_1).shape)

#CNN output for Figure 5b and c
result_1=model_final.predict(X_test_filtered[n:n+1])

#read Vs model of Fgiure 5d
n=29
r='../Data/A12v1_8192mod_no_Fault_FS72recs_T0.6_SmthMask/Rand_Models_Masks_smthMask/models_with_void_02/'+str(n).zfill(6)+'/proc000000_vs.bin'
nx=241
nz=81
Vz_bin = np.fromfile(r,dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(1, nx, nz, 1, order='F').copy()/100
Vs_2=np.array(Vz[:,49:191,0:41])
print(np.array(Vs_2).shape)

#CNN output for Figure 5e and f
result_2=model_final.predict(X_test_filtered[n:n+1])

#read waveform
#n=22 for Figure 5g, h, and i
n=22
nx=72
nz=301
dl=0.5
Vz_bin = np.fromfile('../Data/bin_A12v2_02_Nm8192_Nr72_Nt301_Ns3/Vz.bin',dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(8192, nx, nz, 3, order='F').copy()
X_test=np.array(Vz)
print(np.array(X_test).shape)

#bandpass filter
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
    if i%10000 == 0: print(i)
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
print(X_test_filtered.shape)

#read Vs model for Figure 5g
r='../Data/A12v2_8192mod_Fault_FS72recs_T0.6_SmthMask/Rand_Models_Masks_smthMask/models_with_void_02/'+str(n).zfill(6)+'/proc000000_vs.bin'
nx=241
nz=81
Vz_bin = np.fromfile(r,dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(1, nx, nz, 1, order='F').copy()/100
Vs_3=np.array(Vz[:,49:191,0:41])
print(np.array(Vs_3).shape)

#CNN output for Figure 5h and i
result_3=model_final.predict(X_test_filtered[n:n+1])

#read waveform
#n=28 for Figure 5i, k, and l
n=28
nx=72
nz=301
dl=0.5
Vz_bin = np.fromfile('../Data/bin_A12v2_01_Nm8192_Nr72_Nt301_Ns3/Vz.bin',dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(8192, nx, nz, 3, order='F').copy()
X_test=np.array(Vz)
print(np.array(X_test).shape)

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
    if i%10000 == 0: print(i)
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
print(X_test_filtered.shape)

#read Vs model of Figure 5j
r='../Data/A12v2_8192mod_Fault_FS72recs_T0.6_SmthMask/Rand_Models_Masks_smthMask/models_with_void_01/'+str(n).zfill(6)+'/proc000000_vs.bin'
nx=241
nz=81
Vz_bin = np.fromfile(r,dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(1, nx, nz, 1, order='F').copy()/100
Vs_4=np.array(Vz[:,49:191,0:41])
print(np.array(Vs_4).shape)

#CNN output for Figure 5k and l
result_4=model_final.predict(X_test_filtered[n:n+1])

#read waveform
#n=10 for Figure 5m, n, and o
n=10
nx=72
nz=301
dl=0.5
Vz_bin = np.fromfile('../Data/bin_A12v2_00_Nm8192_Nr72_Nt301_Ns3/Vz.bin',dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(8192, nx, nz, 3, order='F').copy()
X_test=np.array(Vz)
print(np.array(X_test).shape)

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
    if i%10000 == 0: print(i)
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
print(X_test_filtered.shape)

#read Vs model of Figure 5m
r='../Data/A12v2_8192mod_Fault_FS72recs_T0.6_SmthMask/Rand_Models_Masks_smthMask/models_with_void_00/'+str(n).zfill(6)+'/proc000000_vs.bin'
nx=241
nz=81
Vz_bin = np.fromfile(r,dtype=np.float32, count=-1, sep='')
Vz = Vz_bin.reshape(1, nx, nz, 1, order='F').copy()/100
Vs_5=np.array(Vz[:,49:191,0:41])
print(np.array(Vs_5).shape)

#CNN output for Figure 5n and o
result_5=model_final.predict(X_test_filtered[n:n+1])


###Figure 5 in the paper
fig = plt.figure(figsize=(16,16))
ax_v_1 = fig.add_subplot(5, 3, 1)    #a
ax_nv_1 = fig.add_subplot(5, 3, 2)   #b
ax_m_1 = fig.add_subplot(5, 3, 3)    #c

ax_v_2 = fig.add_subplot(5, 3, 4)    #d
ax_nv_2 = fig.add_subplot(5, 3, 5)   #e
ax_m_2 = fig.add_subplot(5, 3, 6)    #f

ax_v_3 = fig.add_subplot(5, 3, 7)    #g
ax_nv_3 = fig.add_subplot(5, 3, 8)   #h
ax_m_3 = fig.add_subplot(5, 3, 9)    #i

ax_v_4 = fig.add_subplot(5, 3, 10)   #j
ax_nv_4 = fig.add_subplot(5, 3, 11)  #k
ax_m_4 = fig.add_subplot(5, 3, 12)   #l

ax_v_5 = fig.add_subplot(5, 3, 13)   #m
ax_nv_5 = fig.add_subplot(5, 3, 14)  #n
ax_m_5 = fig.add_subplot(5, 3, 15)   #o

cmap=plt.cm.get_cmap('Spectral')

ax_v_1.plot(0,0)
im1=ax_v_1.imshow(np.fliplr(np.rot90(Vs_1[0,:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_v_1.text(24,-2,"(a)",size=11)
ax_v_1.text(94.5,-1.6,"Vs (m/s)",size=11)
ax_v_1.text(17,10,"Depth (m)",ha="center",va="center",rotation=90,size=11)
#ax_v_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_nv_1.plot(0,1)
im2=ax_nv_1.imshow(np.fliplr(np.rot90(result_1[1][0][:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_nv_1.text(24,-2,"(b)",size=11)
ax_nv_1.text(94.5,-1.6,"Vs (m/s)",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_m_1.plot(0,2)
im3=ax_m_1.imshow(np.fliplr(np.rot90(result_1[0][0][:,:,0],k=-1)), cmap=cmap, vmin=0, vmax=1, extent=[24,95,20.5,0])
ax_m_1.text(24,-2,"(c)",size=11)
ax_m_1.text(92,-1.6,"Probability",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_v_2.plot(1,0)
im4=ax_v_2.imshow(np.fliplr(np.rot90(Vs_2[0,:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_v_2.text(24,-2,"(d)",size=11)
ax_v_2.text(94.5,-1.6,"Vs (m/s)",size=11)
ax_v_2.text(17,10,"Depth (m)",ha="center",va="center",rotation=90,size=11)
#ax_v_2.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_nv_2.plot(1,1)
im5=ax_nv_2.imshow(np.fliplr(np.rot90(result_2[1][0][:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_nv_2.text(24,-2,"(e)",size=11)
ax_nv_2.text(92,-1.6,"Vs (m/s)",size=11)
#ax_m_2.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_2.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_m_2.plot(1,2)
im6=ax_m_2.imshow(np.fliplr(np.rot90(result_2[0][0][:,:,0],k=-1)), cmap=cmap, vmin=0, vmax=1, extent=[24,95,20.5,0])
ax_m_2.text(24,-2,"(f)",size=11)
ax_m_2.text(92,-1.6,"Probability",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_v_3.plot(2,0)
im7=ax_v_3.imshow(np.fliplr(np.rot90(Vs_3[0,:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_v_3.text(24,-2,"(g)",size=11)
ax_v_3.text(94.5,-1.6,"Vs (m/s)",size=11)
ax_v_3.text(17,10,"Depth (m)",ha="center",va="center",rotation=90,size=11)
#ax_v_3.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_nv_3.plot(2,1)
im8=ax_nv_3.imshow(np.fliplr(np.rot90(result_3[1][0][:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_nv_3.text(24,-2,"(h)",size=11)
ax_nv_3.text(92,-1.5,"Vs (m/s)",size=11)
#ax_m_3.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_3.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_m_3.plot(2,2)
im9=ax_m_3.imshow(np.fliplr(np.rot90(result_3[0][0][:,:,0],k=-1)), cmap=cmap, vmin=0, vmax=1, extent=[24,95,20.5,0])
ax_m_3.text(24,-2,"(i)",size=11)
ax_m_3.text(92,-1.5,"Probability",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_v_4.plot(3,0)
im10=ax_v_4.imshow(np.fliplr(np.rot90(Vs_4[0,:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_v_4.text(24,-2,"(j)",size=11)
ax_v_4.text(94.5,-1.5,"Vs (m/s)",size=11)
ax_v_4.text(17,10,"Depth (m)",ha="center",va="center",rotation=90,size=11)
#ax_v_4.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_nv_4.plot(3,1)
im11=ax_nv_4.imshow(np.fliplr(np.rot90(result_4[1][0][:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_nv_4.text(24,-2,"(k)",size=11)
ax_nv_4.text(92,-1.5,"Vs (m/s)",size=11)
#ax_m_4.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_4.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_m_4.plot(3,2)
im12=ax_m_4.imshow(np.fliplr(np.rot90(result_4[0][0][:,:,0],k=-1)), cmap=cmap, vmin=0, vmax=1, extent=[24,95,20.5,0])
ax_m_4.text(24,-2,"(l)",size=11)
ax_m_4.text(92,-1.5,"Probability",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

ax_v_5.plot(4,0)
im13=ax_v_5.imshow(np.fliplr(np.rot90(Vs_5[0,:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_v_5.text(24,-2,"(m)",size=11)
ax_v_5.text(94.5,-1.5,"Vs (m/s)",size=11)
ax_v_5.text(17,10,"Depth (m)",ha="center",va="center",rotation=90,size=11)
ax_v_5.text(59,27,"Distance (m)",ha="center",va="center",size=11)

ax_nv_5.plot(4,1)
im14=ax_nv_5.imshow(np.fliplr(np.rot90(result_5[1][0][:,:,0]*100,k=-1)), cmap=cmap, vmin=0, vmax=700, extent=[24,95,20.5,0])
ax_nv_5.text(24,-2,"(n)",size=11)
ax_nv_5.text(92,-1.5,"Vs (m/s)",size=11)
#ax_m_5.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
ax_nv_5.text(59,27,"Distance (m)",ha="center",va="center",size=11)

ax_m_5.plot(3,2)
im15=ax_m_5.imshow(np.fliplr(np.rot90(result_5[0][0][:,:,0],k=-1)), cmap=cmap, vmin=0, vmax=1, extent=[24,95,20.5,0])
ax_m_5.text(24,-2,"(o)",size=11)
ax_m_5.text(92,-1.5,"Probability",size=11)
ax_m_5.text(59,27,"Distance (m)",ha="center",va="center",size=11)
#ax_m_1.text(-5,10,"Depth (m)",ha="center",va="center",rotation=90,size=15)
#ax_m_1.text(35,25,"offset (m)",ha="center",va="center",size=15)

fig.colorbar(im1, cax=fig.add_axes([0.353,0.699,0.0075,0.065]))
fig.colorbar(im2, cax=fig.add_axes([0.63,0.699,0.0075,0.065]))
fig.colorbar(im3, cax=fig.add_axes([0.905,0.699,0.0075,0.065]))
fig.colorbar(im4, cax=fig.add_axes([0.353,0.581,0.0075,0.065]))
fig.colorbar(im5, cax=fig.add_axes([0.63,0.581,0.0075,0.065]))
fig.colorbar(im6, cax=fig.add_axes([0.905,0.581,0.0075,0.065]))
fig.colorbar(im7, cax=fig.add_axes([0.353,0.462,0.0075,0.065]))
fig.colorbar(im8, cax=fig.add_axes([0.63,0.462,0.0075,0.065]))
fig.colorbar(im9, cax=fig.add_axes([0.905,0.462,0.0075,0.065]))
fig.colorbar(im10, cax=fig.add_axes([0.353,0.343,0.0075,0.065]))
fig.colorbar(im11, cax=fig.add_axes([0.63,0.343,0.0075,0.065]))
fig.colorbar(im12, cax=fig.add_axes([0.905,0.343,0.0075,0.065]))
fig.colorbar(im13, cax=fig.add_axes([0.353,0.225,0.0075,0.065]))
fig.colorbar(im14, cax=fig.add_axes([0.63,0.225,0.0075,0.065]))
fig.colorbar(im15, cax=fig.add_axes([0.905,0.225,0.0075,0.065]))
plt.subplots_adjust(wspace=0.25, hspace=-0.6)

plt.show()












