#test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,cross_validate,train_test_split
import tensorflow as tf
from sklearn.preprocessing import RobustScaler,StandardScaler
from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,\
GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D,Activation,Flatten,Conv2D,MaxPool2D,GlobalAveragePooling2D,\
AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
import keras
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import RepeatedKFold
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder
# from keras.optimizers import adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss
from keras.models import Model

class model():
    
    # def __init__(self,cap_size):
    #     self.cap_size = cap_size
    
    # def labelMake(epoch,hot=True):
    #     type_eeg = str(type(epoch))
    #     print('label  ',type_eeg)
    #     if "<class 'list'>" in type_eeg: 
    #         trials = int(len(epoch))/3
    #     elif "<class 'numpy.ndarray'>" in type_eeg:
    #         trials = int((epoch.shape[0])/3)
    #     print('Total trials : ',trials)
        
    #     if hot :
    #         p_label = [[1,0,0]]*trials
    #         f_label = [[0,1,0]]*trials
    #         g_label = [[0,0,1]]*trials
    #         #Use for custom CNN
    #     else:
    #         p_label = [0]*trials
    #         f_label = [1]*trials
    #         g_label = [2]*trials
    #         #Use for tuner CNN
    #     labels = np.array(p_label+f_label+g_label)
    #     # print(labels.shape,'....',np.array(epoch).shape)
    #     return labels
    
    def split_scale_label(data,labels,scale,hot=True,val=False):
        #labels = labelMake(data,hot)
        x_val_scl = []
        y_val = []

        if scale == 'robust':
            data = np.array([RobustScaler().fit_transform(data[i]) for i in range(len(data))])
        elif scale == 'standard':
            data = np.array([StandardScaler().fit_transform(data[i]) for i in range(len(data))])


        if val:
            x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=.20,random_state=69)
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=.25,random_state=69)
        else:
            x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=.20,random_state=69)  

        return x_train,x_test,x_val,y_train,y_test,y_val

    def cnn(timeLength,channel=124):
        clear_session()
        model=Sequential()
        model.add(Conv2D(filters=5,kernel_size=3,strides=1,input_shape=(timeLength, channel,1)))#1
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPool2D(pool_size=2,strides=2))#1
        model.add(Conv2D(filters=5,kernel_size=3,strides=1))#2
        model.add(LeakyReLU())
        model.add(MaxPool2D(pool_size=2,strides=2))#2
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=5,kernel_size=3,strides=1))#3
        model.add(LeakyReLU())
        model.add(AveragePooling2D(pool_size=2,strides=2))#3
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=5,kernel_size=3,strides=1))#4
        model.add(LeakyReLU())
        model.add(AveragePooling2D(pool_size=2,strides=2))#4
        model.add(Conv2D(filters=5,kernel_size=3,strides=1))#5
        model.add(LeakyReLU())
        model.add(GlobalAveragePooling2D())#5
        #model.add(Flatten())
        model.add(Dense(60,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(30,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20,activation='relu'))
        model.add(Dense(3,activation='softmax'))
    
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=30e-3),loss='categorical_crossentropy',metrics=['accuracy']) #optimizer=tf.keras.optimizers.Adam(learning_rate=10e-3)
        return model
    
    

    def vgg16_model(img_rows, img_cols, num_classes=None,channel=1):

        model = VGG16(weights='imagenet') #include_top=fowlkes_mallows_score

        model.layers.pop()

        model.outputs = [model.layers[-1].output]

        model.layers[-1].outbound_node = []

        x=Dense(num_classes, activation='softmax')(model.output)

        model=Model(model.input,x)

    #To set the first 8 layers to non-trainable (weights will not be updated)

        for layer in model.layers[:8]:
            layer.trainable = False

    # Learning rate is changed to 0.001
        #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=30e-3), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    
    def rkf(model,data,labels,splits=5,repeats=2,epoch=10,batch_size=20,hot=True,simple=True):
        print('Entered train')
        # data = data
        # labels = labels
        rkf = RepeatedKFold(n_splits=splits,n_repeats=repeats,random_state=20)
        # data = np.array([StandardScaler().fit_transform(data[i]) for i in range(len(data))])
        # x_val,x = train_test_split(data,test_size = 0.1,random_state=99)
        cvscores=[]
        for trn_indx, test_indx in rkf.split(data):
          # print(trn_indx,test_indx)
            # if simple:
            #   val_indx = test_indx[(len(test_indx))//3:]
            #   test_indx = test_indx[:(len(test_indx))//3]
            #   x_train, x_test, x_val = data[trn_indx], data[test_indx], data[val_indx]
            #   y_train, y_test, y_val = labels[trn_indx], labels[test_indx], labels[val_indx]
            # else:
            #   x_train,x_test,x_val,y_train,y_test,y_val = m.split_scale_label(data,labels,scale = 'standard', val= True)

            # x_train_ = np.array([i.reshape((i.shape[0], i.shape[1], 1)) for i in x_train])
            # x_test_ = np.array([i.reshape((i.shape[0], i.shape[1], 1)) for i in x_test])
            # x_val_ = np.array([i.reshape((i.shape[0], i.shape[1], 1)) for i in x_val])
            x_train,x_left,y_train,y_left = train_test_split(data,labels,random_state=50,test_size=.3)
            x_test,x_val,y_test,y_val = train_test_split(x_left,y_left,random_state=20,test_size=.5)
            model.fit(x_train,y_train,epochs=epoch,batch_size=batch_size,validation_data=(x_val,y_val))
            score = model.evaluate(x_test, y_test)
            cvscores.append(score[1] * 100)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        return model,cvscores

    