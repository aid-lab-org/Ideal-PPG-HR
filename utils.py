import os
from glob import glob
import pandas as pd
from pathlib import Path
import scipy
import numpy as np
import h5py
import pickle
from biosppy.signals import tools as tools
from filters import filter_ppg, filter_ecg, normalize

def getSlice_norm(t1,df,offset):
    t2 = t1 + offset
    if t1+offset>df.index.max():
        return [1]
    slice = df.loc[t1:t2][0]
    slice = normalize(slice)
    slice[np.isnan(slice)] = 1
    return np.array(slice)

def process(resampled_ppg,resampled_ecg,window_size):
    filtered_ppg = filter_ppg(resampled_ppg,128)
    filtered_ecg = filter_ecg(resampled_ecg,128)

    df_ppg_overlap = ([getSlice_norm(t1,pd.DataFrame(filtered_ppg),(window_size-1)) for t1 in range(pd.DataFrame(filtered_ppg).index.min(),pd.DataFrame(filtered_ppg).index.max(),int(window_size*1.0))])[:-1]

    df_ecg_overlap = ([getSlice_norm(t1,pd.DataFrame(filtered_ecg),(window_size-1)) for t1 in range(pd.DataFrame(filtered_ecg).index.min(),pd.DataFrame(filtered_ecg).index.max(),int(window_size*1.0))])[:-1]

    return df_ppg_overlap,df_ecg_overlap
    
def BIDMC(split, data_path = "data/physionet.org/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/*Signal*", sample_frq = 125, window_size=512):
    labels_to_keep =  [' PLETH', ' II',] ; rename = {' PLETH': 'pleth_6',' II':'ecg'} 
    df_ppg_overlap_multi = []; df_ecg_overlap_multi = []
    all_files = glob(pathname=data_path)
    file_count = len(all_files)
    count = 0

    for filename in all_files:
        count +=1
        #print(f'[Processing file {count} of {file_count} at {datetime.now()}. {time_elapsed(start)} elapsed.]')

        name_csv = os.path.split(filename)[1].rstrip('.csv')
        #print(f'Creating temporary data frame for {name_csv}, {time_elapsed(start)} elapsed.')
        tmpdf = pd.read_csv(filename) 
        if count in split:
            try:
                tmpdf = tmpdf[labels_to_keep]
                tmpdf  = tmpdf .rename(columns=rename)
                tmpdf['name'] = name_csv

                resampled_ppg = scipy.signal.resample(tmpdf.pleth_6, int(len(tmpdf.pleth_6)/sample_frq*128), t=None, axis=0, window=None, domain='time')
                resampled_ecg = scipy.signal.resample(tmpdf.ecg, int(len(tmpdf.ecg)/sample_frq*128), t=None, axis=0, window=None, domain='time')

                df_ppg_overlap,df_ecg_overlap = process(resampled_ppg,resampled_ecg,window_size)

                df_ppg_overlap_multi+=df_ppg_overlap; df_ecg_overlap_multi+=df_ecg_overlap

            except Exception as e: 
                print('error in',name_csv, e)

        #print(f'Deleting temporary dataframe {time_elapsed(start)} elapsed.')
        del(tmpdf)
    return df_ppg_overlap_multi, df_ecg_overlap_multi

def CAPNO(split, data_path = "data/dataverse_files capno/data/mat/*8min*", sample_frq = 300, window_size=512):
    df_ppg_overlap_multi = []; df_ecg_overlap_multi = []
    all_files = glob(pathname=data_path)
    file_count = len(all_files)
    df_ppg_overlap = []; df_ecg_overlap = []; count = 0
    
    for filename in all_files:
        count +=1
        #print(f'[Processing file {count} of {file_count} at {datetime.now()}. {time_elapsed(start)} elapsed.]')
        if count in split:
            try:
                f = h5py.File(filename,'r')
                ecg = np.array(f.get('signal/ecg/y'))[0]
                pleth_6 = np.array(f.get('signal/pleth/y'))[0]

                resampled_ppg = scipy.signal.resample(pleth_6, int(len(pleth_6)/sample_frq*128), t=None, axis=0, window=None, domain='time')
                resampled_ecg = scipy.signal.resample(ecg, int(len(ecg)/sample_frq*128), t=None, axis=0, window=None, domain='time')

                df_ppg_overlap,df_ecg_overlap = process(resampled_ppg,resampled_ecg,window_size)

                df_ppg_overlap_multi+=df_ppg_overlap; df_ecg_overlap_multi+=df_ecg_overlap
            except Exception as e: print('error in',filename,e)

        #print(f'Deleting temporary dataframe {time_elapsed(start)} elapsed.')
        else: pass

    return df_ppg_overlap_multi, df_ecg_overlap_multi

def WESAD(split, data_path = "data/WESAD/S", ppg_sampling_rate = 64, ecg_sampling_rate = 700, window_size =512):
    df_ppg_overlap_multi = []; df_ecg_overlap_multi = []

    for sub_num in split:
        
        file_pkl= data_path+str(sub_num)+"/S"+str(sub_num)+".pkl"
        with open(file_pkl, 'rb') as f:
            try:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()

                pleth_6 = data['signal']['wrist']['BVP']
                ecg = data['signal']['chest']['ECG']
                
                resampled_ppg = scipy.signal.resample(pleth_6, int(len(pleth_6)/ppg_sampling_rate*128), t=None, axis=0, window=None, domain='time')
                resampled_ecg = scipy.signal.resample(ecg, int(len(ecg)/ecg_sampling_rate*128), t=None, axis=0, window=None, domain='time')
                
                df_ppg_overlap,df_ecg_overlap = process(resampled_ppg.flatten(),resampled_ecg.flatten(),window_size)

                df_ppg_overlap_multi+=df_ppg_overlap; df_ecg_overlap_multi+=df_ecg_overlap
            
            except EOFError:
                print('EOF error in S'+str(sub_num))
                u = list() 
    
    return df_ppg_overlap_multi, df_ecg_overlap_multi

def DALIA(split, data_path = "data/PPG-DaLiA/PPG_FieldStudy/S", ppg_sampling_rate = 64, ecg_sampling_rate = 700, window_size =512):
    df_ppg_overlap_multi = []; df_ecg_overlap_multi = []
    for sub_num in split:
        
        file_pkl= data_path+str(sub_num)+"/S"+str(sub_num)+".pkl"
        with open(file_pkl, 'rb') as f:
            try:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()

                pleth_6 = data['signal']['wrist']['BVP']
                ecg = data['signal']['chest']['ECG']
                
                resampled_ppg = scipy.signal.resample(pleth_6, int(len(pleth_6)/ppg_sampling_rate*128), t=None, axis=0, window=None, domain='time')
                resampled_ecg = scipy.signal.resample(ecg, int(len(ecg)/ecg_sampling_rate*128), t=None, axis=0, window=None, domain='time')
                
                df_ppg_overlap,df_ecg_overlap = process(resampled_ppg.flatten(),resampled_ecg.flatten(),window_size)
                
                df_ppg_overlap_multi+=df_ppg_overlap; df_ecg_overlap_multi+=df_ecg_overlap
            
            except EOFError:
                print('EOF error in S'+str(sub_num))
                u = list()
    
    return df_ppg_overlap_multi, df_ecg_overlap_multi

def PTT(data_path = "data/physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/*s*", sample_frq=500, window_size = 512):
    labels_to_keep =  ['pleth_6', 'ecg']; rename =  {'pleth_6': 'pleth_6','ecg':'ecg'}
    df_ppg_overlap_multi = []; df_ecg_overlap_multi = []
    all_files = glob(pathname=data_path)
    file_count = len(all_files)
    count = 0

    for filename in all_files:
        count +=1
        #print(f'[Processing file {count} of {file_count} at {datetime.now()}. {time_elapsed(start)} elapsed.]')

        name_csv = os.path.split(filename)[1].rstrip('.csv')
        #print(f'Creating temporary data frame for {name_csv}, {time_elapsed(start)} elapsed.')
        tmpdf = pd.read_csv(filename) 

        try:
            tmpdf = tmpdf[labels_to_keep]
            tmpdf  = tmpdf .rename(columns=rename)
            tmpdf['name'] = name_csv
            
            resampled_ppg = scipy.signal.resample(tmpdf.pleth_6, int(len(tmpdf.pleth_6)/sample_frq*128), t=None, axis=0, window=None, domain='time')
            resampled_ecg = scipy.signal.resample(tmpdf.ecg, int(len(tmpdf.ecg)/sample_frq*128), t=None, axis=0, window=None, domain='time')

            df_ppg_overlap,df_ecg_overlap = process(resampled_ppg,resampled_ecg,window_size)

            df_ppg_overlap_multi+=df_ppg_overlap; df_ecg_overlap_multi+=df_ecg_overlap
            
        except Exception as e: 
            print('error in',name_csv, e)

        #print(f'Deleting temporary dataframe {time_elapsed(start)} elapsed.')
        del(tmpdf)
    return df_ppg_overlap_multi, df_ecg_overlap_multi

train_splits = {'BIDMC' : list(range(1,4))+list(range(15,53)), 
                'CAPNO' : list(range(9,43)),
                'DALIA' : [5,6,7,8,9,10,11,12,13,14,15],
                'WESAD' : [2,3,5,7,8,9,11,13,14,15,16,17]}
test_splits = { 'BIDMC' : list(range(4,15)), 
                'CAPNO' : list(range(1,9)),
                'DALIA' : [1,2,3,4],
                'WESAD' : [4,6,10]}

def train_test_split(dataset, split, window_size):
    '''
    Load train and test data used for SynthPPG
    Takes arguments to select train or test splits as required:
        - train
        - BIDMC_test
        - CAPNO_test
        - DALIA_test
        - WESAD_test
        - PTT_test
    Returns:
        - Dataframe of train or test split, for test splits corresponding ECG segments are also returned for MAE-HR calculation
    '''
    print(f'Using dataset: {dataset} with {window_size} second windows')
    if dataset == 'train':
        BIDMC_train_ppg, BIDMC_train_ecg = BIDMC(split = train_splits['BIDMC'], window_size=128*window_size)
        CAPNO_train_ppg, CAPNO_train_ecg = CAPNO(split = train_splits['CAPNO'], window_size=128*window_size)
        DALIA_train_ppg, DALIA_train_ecg = DALIA(split = train_splits['DALIA'], window_size=128*window_size)
        WESAD_train_ppg, WESAD_train_ecg = WESAD(split = train_splits['WESAD'], window_size=128*window_size)
        train_dataset = np.concatenate([BIDMC_train_ppg, CAPNO_train_ppg, DALIA_train_ppg, WESAD_train_ppg],axis=0)
        
        return train_dataset
    
    if dataset == 'BIDMC_test':
        BIDMC_test_ppg, BIDMC_test_ecg = BIDMC(split = test_splits['BIDMC'], window_size=128*window_size)
        
        return np.vstack(BIDMC_test_ppg), np.vstack(BIDMC_test_ecg)

    if dataset == 'CAPNO_test':
        CAPNO_test_ppg, CAPNO_test_ecg = CAPNO(split = test_splits['CAPNO'], window_size=128*window_size)
        
        return np.vstack(CAPNO_test_ppg), np.vstack(CAPNO_test_ecg)
    
    if dataset == 'DALIA_test':
        DALIA_test_ppg, DALIA_test_ecg = DALIA(split = test_splits['DALIA'], window_size=128*window_size)
        
        return np.vstack(DALIA_test_ppg), np.vstack(DALIA_test_ecg)
    
    if dataset == 'WESAD_test':
        WESAD_test_ppg, WESAD_test_ecg = WESAD(split = test_splits['WESAD'], window_size=128*window_size)
        
        return np.vstack(WESAD_test_ppg), np.vstack(WESAD_test_ecg)
    
    if dataset == 'PTT_test':
        if split=='all': datapath = "data/physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/*s*.csv"
        else: datapath = f"data/physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/*s*_{split}.csv"
        PTT_test_ppg, PTT_test_ecg   = PTT(data_path=datapath, window_size=128*window_size)

        return np.vstack(PTT_test_ppg), np.vstack(PTT_test_ecg)




