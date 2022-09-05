import time
import mne
from mne import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from mne.io import read_raw_eeglab,read_raw_cnt
from mne.io import eeglab
import numpy as np
import os
import pickle as pk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,cross_validate,train_test_split


class clean_filter_epoch():
    
    def __init__(self,dirName,pickle_path):
        # self.cap_size = cap_size
        self.dirName = dirName

    def dataLoad(self,epochs):
        EEG_I_txt = epochs[0]
        EEG_I_pic = epochs[1]
        EEG_I_sound = epochs[2]
        EEG_P_txt = epochs[3]
        EEG_P_pic = epochs[4]
        EEG_P_sound = epochs[5]

        return EEG_I_txt,EEG_I_pic,EEG_I_sound,EEG_P_txt,EEG_P_pic,EEG_P_sound
    def dataCreate(self,EEG,class_name):
      #class_lb
        EEG = EEG.equalize_event_counts(class_name) # handeling class imbalance
        
        p_Array = EEG[0][class_name[0]].get_data()
        f_Array = EEG[0][class_name[1]].get_data()
        g_Array = EEG[0][class_name[2]].get_data()
        
        if p_Array.shape[0] == f_Array.shape[0] == g_Array.shape[0] :
            trials = p_Array.shape[0]
        
        
        
        p_label = [[1,0,0]]*trials
        f_label = [[0,1,0]]*trials
        g_label = [[0,0,1]]*trials
        labels = np.array(p_label+f_label+g_label)
        
        data_list = p_Array.tolist()+f_Array.tolist()+g_Array.tolist()
        
        merged_data_list = np.stack(data_list)
        merged_data_list =np.moveaxis(merged_data_list,1,2) 
        
        # EEG_p_p = EEG_p_p.get_data()
        # nchannels,nsamples = EEG_p_p.shape[1:]
        #channel_names = perc_pic_epoch.ch_names
        #event_onsets = np.array(events[:,0])
        #event_codes = events[:,2]
        #sample_rate = int(perc_pic_epoch.info['sfreq'])
        #cl_lab = cl_lab
        #nclasses = len(cl_lab)
        return merged_data_list,labels
    
    def its_Trim_Time(self,epochs,epochs_of_interest):
        epochs_sub = epochs[epochs_of_interest]
        imagine_orthographic_epochs = epochs_sub[['imag_flower_t', 'imag_guitar_t', 'imag_penguin_t']].crop(tmin=-2, tmax =4)
        imagine_pictorial_epochs = epochs_sub[['imag_flower_p', 'imag_guitar_p', 'imag_penguin_p']].crop(tmin=-2, tmax =4)
        imagine_audio_epochs = epochs_sub[['imag_flower_s', 'imag_guitar_s', 'imag_penguin_s']].crop(tmin=-2, tmax =4)
        perception_orthographic_epochs = epochs_sub[['perc_flower_t', 'perc_guitar_t', 'perc_penguin_t']].crop(tmin=-2, tmax =3)
        perception_pictorial_epochs = epochs_sub[['perc_flower_p', 'perc_guitar_p', 'perc_penguin_p']].crop(tmin=-2, tmax =3)
        perception_audio_epochs = epochs_sub[['perc_flower_s', 'perc_guitar_s', 'perc_penguin_s']].crop(tmin=-2, tmax =4)
        epochs = [imagine_orthographic_epochs,imagine_pictorial_epochs,imagine_audio_epochs,perception_orthographic_epochs,perception_pictorial_epochs,
                     perception_audio_epochs]
        return epochs
    
    def mergeEpoch(self,epochs,event_list): 
        mne.epochs.combine_event_ids(epochs, event_list[0], {'imag_flower_t': 300}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[1], {'imag_penguin_t': 301}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[2], {'imag_guitar_t': 302}, copy = False)
    
        mne.epochs.combine_event_ids(epochs, event_list[3], {'imag_flower_p': 303}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[4], {'imag_penguin_p': 304}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[5], {'imag_guitar_p': 305}, copy = False)
    
        mne.epochs.combine_event_ids(epochs, event_list[6], {'imag_flower_s': 306}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[7], {'imag_penguin_s': 307}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[8], {'imag_guitar_s': 308}, copy = False)
    
    
        mne.epochs.combine_event_ids(epochs, event_list[9], {'perc_flower_t': 309}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[10], {'perc_penguin_t': 310}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[11], {'perc_guitar_t': 311}, copy = False)
    
        mne.epochs.combine_event_ids(epochs, event_list[12], {'perc_flower_p': 312}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[13], {'perc_penguin_p': 313}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[14], {'perc_guitar_p': 314}, copy = False)
    
        mne.epochs.combine_event_ids(epochs, event_list[15], {'perc_flower_s': 315}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[16], {'perc_penguin_s': 316}, copy = False)
        mne.epochs.combine_event_ids(epochs, event_list[17], {'perc_guitar_s': 317}, copy = False)
    
        epochs_of_interest  = ['imag_flower_t','imag_penguin_t', 'imag_guitar_t', 'imag_flower_p','imag_guitar_p',
                     'imag_penguin_p','imag_flower_s', 'imag_guitar_s', 'imag_penguin_s',
                    'perc_flower_t', 'perc_guitar_t', 'perc_penguin_t', 'perc_flower_p',
                    'perc_guitar_p', 'perc_penguin_p', 'perc_flower_s', 'perc_guitar_s', 'perc_penguin_s']
        return epochs,epochs_of_interest
    
    
    def epoch(self,raw):
        events, event_ids = mne.events_from_annotations(raw, verbose = False)
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_ids, preload=True, tmin = -0.2,tmax =4,baseline=None, event_repeated='merge')
        print('--------------Epoched--------------------')
        return epochs
    
    def cat(self,raw):
        events, event_ids = mne.events_from_annotations(raw, verbose = False)
        imag_flower_t = []
        imag_penguin_t = []
        imag_guitar_t = []
        imag_flower_p = []
        imag_guitar_p = []
        imag_penguin_p = []
        imag_flower_s = []
        imag_guitar_s = []
        imag_penguin_s = []

        perc_flower_t = []
        perc_guitar_t = []
        perc_penguin_t = []
        perc_flower_p = []
        perc_guitar_p = []
        perc_penguin_p = []
        perc_flower_s = []
        perc_guitar_s = []
        perc_penguin_s = []

        for id in event_ids:
            if 'Imagination' in id:
                if 'flower' in id:
                    if '_t_' in id:
                        imag_flower_t.append(id)
                    elif '_image_'in id:
                        imag_flower_p.append(id)
                    elif '_a_' in id:
                        imag_flower_s.append(id)
                    else:
                        print(id)
                elif 'guitar' in id:
                    if '_t_' in id:
                        imag_guitar_t.append(id)
                    elif '_image_'in id:
                        imag_guitar_p.append(id)
                    elif '_a_'in id:
                        imag_guitar_s.append(id)
                    else:
                        print(id)
                elif 'penguin' in id:
                    if '_t_' in id:
                        imag_penguin_t.append(id)
                    elif '_image_'in id:
                        imag_penguin_p.append(id)
                    elif '_a_'in id:
                        imag_penguin_s.append(id)
                    else:
                        print(id)



            if 'Perception' in id:
                if 'flower' in id:
                   # print(id)
                    if '_t_' in id:
                        perc_flower_t.append(id)
                    elif '_image_' in id:
                        perc_flower_p.append(id)
                    elif 'a_'in id or 'audio' in id:
                        perc_flower_s.append(id)
                    else:
                        print(id)
                elif 'guitar' in id:
                    if '_t_' in id:
                        perc_guitar_t.append(id)
                    elif '_image_'in id:
                        perc_guitar_p.append(id)
                    elif 'tiona_'in id or 'audio' in id:
                        perc_guitar_s.append(id)
                    else:
                        print(id)
                elif 'penguin' in id:
                    if '_t_' in id:
                        perc_penguin_t.append(id)
                    elif '_image_'in id:
                        perc_penguin_p.append(id)
                    elif 'tiona_'in id or 'audio' in id:
                        perc_penguin_s.append(id)
                    else:
                        print(id)

        event_list = [imag_flower_t,imag_penguin_t, imag_guitar_t, 
                      imag_flower_p,imag_penguin_p,imag_guitar_p,
                      imag_flower_s,imag_penguin_s,imag_guitar_s,
                      perc_flower_t,perc_penguin_t,perc_guitar_t,
                      perc_flower_p,perc_penguin_p,perc_guitar_p,
                      perc_flower_s,perc_penguin_s,perc_guitar_s]
        print("There should be 18 conditions: ", len(event_list))
    #     for li in event_list:
    #         print("Amount of different types in each condition ",len(li))
        print ('------------Categorised-----------')
        return event_list
    
    def filtering(self,raw):
        
        notches = np.arange(50,100,150)
        raw.notch_filter(notches, picks = 'eeg', filter_length = 'auto', phase = 'zero-double', fir_design = 'firwin')
        # raw.plot_psd(area_mode='range', tmax=10.0, picks=picks, average=False)

        # Filter the data to remove low-frequency drifts (i.e. DC drifts):
        raw.filter(l_freq=1., h_freq=None)

        # Frequency analysis
        #%matplotlib qt
        #filtered_psd = raw.plot_psd(fmin = 0,fmax=100, n_fft=2048, spatial_colors=True) # y is decibals and x is frequency
        #filtered_psd.savefig(ppt_num+'_'+session+'_filtered_psd.png')

        reject = dict(mag=5e-12, grad=4000e-13) # this prevents ICA being fit on extreme environmental factors
        amount_variance_explain = .99
        num_components = 20
        ica = mne.preprocessing.ICA(n_components=num_components, random_state=97, max_iter=800)#, method='picard')
        ica.fit(raw, reject = reject)

        from mne.preprocessing import create_ecg_epochs, create_eog_epochs
        
        eog_evoked = preprocessing.create_eog_epochs(raw, ch_name = ['Fp1', 'Fp2']).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        #eog_evoked.plot_joint()
        
        #Here we use the EOG components we estimated earlier, and remove them from the data
        ica_z_thresh = 1.96 
        eog_indices, eog_scores = ica.find_bads_eog(eog_evoked, 
                                                    ch_name=['Fp1', 'Fp2'], 
                                                    threshold=ica_z_thresh)
        ica.exclude = eog_indices
        #ica.exclude = ecg_indices #ica.exclude = muscle_indices
        #ica.plot_scores(eog_scores) # rejected ones are in red
        
        dimension_to_keep = 124
        ica.apply(raw, n_pca_components = dimension_to_keep)
        
        print('--------------FILTERED--------------------')
        return raw

    
    def clean(self,raw,cap_size='medium'):
        #%matplotlib qt 
        #raw.plot(block=True)
        if cap_size == 'large':
            raw.info['bads'] += ['CCP1h'] # ['names of channels to remove']
        picks = mne.pick_types(raw.info, exclude='bads')  # picks can then be taken into epochs
        raw.interpolate_bads(reset_bads=False)  # interpolate for the bad channel to fill in the missing data
        print('--------------CLEANED--------------------')
        return raw   
        
    def dataRead(self,path,ppt_num,session,cap_size = 'medium'):
        #dirName = 'H:\\ray\\Dessertation\\DATA\\sub_'+str(ppt_num)+'_sess_'+str(session)  
        #os.getcwd()+'DATA\\sub'+ppt_num+'_sess'+session
        #files = os.listdir(dirName)
        cap_size = cap_size # either 'small', 'medium' or 'large'
        session = str(session) #either 1, 2 or 3
        ppt_num = str(ppt_num)
        eog =  ['VEOGL', 'VEOGU', 'HEOGR', 'HEOGL']
        path = path+'sub_'+ppt_num+'_sess_'+session+'/sub_'+ppt_num+'_sess_'+session+'.set'
        print('This path : ',path)
        raw = read_raw_eeglab(path,preload=True, eog=eog)
    #     for i in files:
    #         sp = i.split('.')
    #         if sp[-1]=='set':
    #             path = 'sub'+ppt_num+'_sess'+session+'/'+'sub'+ppt_num+'_sess'+session+'.set'
    #             raw = read_raw_eeglab(path,preload=True, eog=eog)
    #         elif sp[-1]=='cnt':
    #             path = 'sub'+ppt_num+'_sess'+session+'/'+'sub'+ppt_num+'_sess'+session+'.cnt' 
    #             raw = read_raw_eeglab(path,preload=True, eog=eog)
        raw.pick_types(eog=False, eeg=True) # ignore eog channels
        print('Sample rate of the data is:', raw.info['sfreq'], 'Hz. It should be 1024 Hz')
        return raw
    
    
    def pre_process_main(self):
        print(self.dirName,'     success')
        p_p_class_name = ['perc_penguin_p','perc_flower_p', 'perc_guitar_p']
        p_s_class_name = ['perc_penguin_s','perc_flower_s', 'perc_guitar_s']
        p_t_class_name = ['perc_penguin_t','perc_flower_t', 'perc_guitar_t']
        i_p_class_name = ['imag_penguin_p','imag_flower_p', 'imag_guitar_p']
        i_s_class_name = ['imag_penguin_s','imag_flower_s', 'imag_guitar_s']
        i_t_class_name = ['imag_penguin_t','imag_flower_t', 'imag_guitar_t']
        
        cl_lab = ['penguin','flower','guitar']
        
        # #dirName = '/content/drive/MyDrive/Colab Notebooks/EEG/DATA/'
        # dirName = "H:\\ray\\Dessertation\\DATA\\new\\"
        # #pickle_path = '/content/drive/MyDrive/Colab Notebooks/EEG/DATA/'
        # pickle_path = 'H:\\ray\\Dessertation\\pickle'
        fileList = os.listdir(self.dirName)
        
        for i in fileList:
            print('enter 1  ',i)
            j = os.listdir(self.dirName+i)
            for i in j:
                print('enter 2')
                # try:
                if '.set' in i:
                    print('enter 3')
                    k=i.split('.')
                    k=k[0].split('_')
                    print(k)
                    subject = k[1]
                    sess = k[3]
                    f_dict = {}
    
                    name_list = ['EEG_I_txt_'+subject+'_'+sess, 'EEG_I_pic_'+subject+'_'+sess, 'EEG_I_sound_'+subject+'_'+sess,
                                  'EEG_P_txt_'+subject+'_'+sess,'EEG_P_pic_'+subject+'_'+sess,'EEG_P_sound_'+subject+'_'+sess
                                ]
    
                    data = self.dataRead(self.dirName,subject,sess)
    
                    # print('gonna sleep 1')
                    time.sleep(5)
    
                    raw = self.clean(data)
                    del(data)
                    print('data del, RAM released')
                    # print('gonna sleep 1')
                    time.sleep(5)
    
                    raw = self.filtering(raw)
                    # print('gonna sleep 1')
                    time.sleep(5)
    
                    event_list = self.cat(raw) #1 time use j
                    epochs = self.epoch(raw)
                    del(raw)
                    print('raw deleated, RAM Released')
                    # print('gonna sleep 1')
                    time.sleep(5)
    
                    epochs,epochs_of_interest = self.mergeEpoch(epochs,event_list)
                    epochs = self.its_Trim_Time(epochs,epochs_of_interest)
    
                    name_list[0],name_list[1],name_list[2],name_list[3],name_list[4],name_list[5] = self.dataLoad(epochs)
    
                    X_i_t , Y_i_t = self.dataCreate(name_list[0],i_t_class_name)
                    X_i_p , Y_i_p = self.dataCreate(name_list[1],i_p_class_name)
                    X_i_s , Y_i_s = self.dataCreate(name_list[2],i_s_class_name)
                    X_p_t , Y_p_t = self.dataCreate(name_list[3],p_t_class_name)
                    X_p_p , Y_p_p = self.dataCreate(name_list[4],p_p_class_name)
                    X_p_s , Y_p_s = self.dataCreate(name_list[5],p_s_class_name)
    
    
    
                    pre_merge_X = {'X_p_p': X_p_p,'X_p_s' : X_p_s, 'X_p_t' : X_p_t,'X_i_p' : X_i_p, 'X_i_s' : X_i_s,'X_i_t' : X_i_t}
                    pre_merge_Y = {'Y_p_p' : Y_p_p,'Y_p_s' : Y_p_s,'Y_p_t' : Y_p_t,'Y_i_p' : Y_i_p, 'Y_i_s' : Y_i_s,'Y_i_t' : Y_i_t}
                    f_dict = {'pre_merge_X' : pre_merge_X, 'pre_merge_Y' : pre_merge_Y}
                    with open(self.dirName+'processed_numpy_epochs_'+str(subject)+'_'+str(sess)+'.pkl','wb') as f:
                        pk.dump(f_dict,f)
                    with open(self.dirName+'og_epochs'+str(subject)+'_'+str(sess)+'.pkl','wb') as f:
                        pk.dump(epochs,f)    
                    print('Dump successfull : ', os.listdir(self.dirName))
                    del(f_dict) 
                else:
                    print('index error maybe')

                # except os.error:
                #     print('index error maybe  ',os.error)
    
    
    

    


    