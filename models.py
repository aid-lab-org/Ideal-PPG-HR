import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
import neurokit2 as nk
import biosppy.signals.ecg as ecg

from filters import filter_ppg

weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

def DeConv1D(filters, kernel_size, strides=1, padding='valid', use_bias=True):
    return tf.keras.layers.Conv2DTranspose(filters= filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                                         output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias,
                                         kernel_initializer=weights_initializer, bias_initializer='zeros', kernel_regularizer=None, 
                                         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)



def buildGenerator(input_shape=512, 
                      filter_size=[64, 128, 256],
                      kernel_size=[16, 16, 16]):
    
    def _downsample(ip, filter_size, kernel_size,stride_size=2):
        
        ip = layers.Conv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        ip = keras.layers.LayerNormalization()(ip)
        ip = tf.keras.activations.relu(ip, alpha=0.2, max_value=None, threshold=0)
        
        return ip
    
    def _upsample(ip, filter_size, kernel_size, stride_size=2, drop_rate = 0.5, apply_dropout=False):
        
        ip = DeConv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        ip = keras.layers.LayerNormalization()(ip)
        if apply_dropout:
            ip = layers.Dropout(rate=drop_rate)
        ip = tf.keras.activations.relu(ip, alpha=0.0, max_value=None, threshold=0)  
        
        return ip
        
    h = inputs = keras.Input(shape=input_shape)
    h = tf.expand_dims(h, axis=1)
    h = tf.expand_dims(h, axis=3) 

    connections = []
    for k in range(3): 
        
        if k==0:
            h =  _downsample(h, filter_size[k], kernel_size[k])
        else:
            h =  _downsample(h, filter_size[k], kernel_size[k])
        connections.append(h)
    h = _upsample(h, filter_size[k], kernel_size[k], stride_size=1)

    for l in range(1, 3):
        h  = _upsample(h, filter_size[k-l], kernel_size[k-l])
    h = DeConv1D(filters=1, kernel_size=kernel_size[k-l], strides=2, padding='same')(h)
    h = tf.keras.activations.tanh(h)
    h = tf.squeeze(h, axis=1)
    h = tf.squeeze(h, axis=2)

    return keras.Model(inputs=inputs, outputs=h)

def buildDiscriminator():
    a = layers.Input(shape=(512,1))
    x = layers.Conv1D(512, 16, strides=2, padding='valid', use_bias=True)(a)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 16, strides=2, padding='valid', activation='LeakyReLU', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 16, strides=2, padding='valid', activation='LeakyReLU', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 16, strides=2, padding='valid', activation='LeakyReLU', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=1)(x)

    model = tf.keras.Model(a,x)
    return model
    
def load_model():
    generator = buildGenerator()
    checkpoint_path = "Checkpoints/gen/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    generator.load_weights(latest)
    return generator

def get_iqr(difference,upper,lower):
    q75, q25 = np.percentile(difference, [upper ,lower])
    iqr = q75 - q25
    return iqr,q75,q25

def average_hr_PPG(ppg_seg,sampling_rate=128):
    '''
    Calculate average heart rate for an PPG segment
    '''

    rpeaks = nk.ppg_findpeaks(ppg_seg, sampling_rate=sampling_rate)['PPG_Peaks']
    differences  = (rpeaks[1:]-rpeaks[:-1])/sampling_rate
    if len(differences)==0:return 0
    hrs = np.reciprocal(np.mean(differences))
    average_hr = hrs*60
    return average_hr

def average_hr_ECG(ecg_seg,sampling_rate=128):
    '''
    Calculate average heart rate for an ECG segment
    '''

    rpeaks = list(ecg.hamilton_segmenter(signal=ecg_seg, sampling_rate=sampling_rate))[0]
    differences  = (rpeaks[1:]-rpeaks[:-1])/sampling_rate
    if len(differences)==0:return 0
    hrs = np.reciprocal(np.mean(differences))
    average_hr = hrs*60
    return average_hr

def get_HR(ecg, ppg, window_size, sampling_rate=128):
    average_hr_og = []; average_hr_gn = []
    for i in range(ppg.shape[0]):

        a = average_hr_ECG(ecg[i],sampling_rate)
        b = average_hr_PPG(ppg[i],sampling_rate)

        average_hr_og.append(a)
        average_hr_gn.append(b)
    average_hr_og = np.array(average_hr_og); average_hr_gn = np.array(average_hr_gn)
    return average_hr_og, average_hr_gn

def MAE_HR(ecg,ppg,window_size,sampling_rate=128):
    '''
    Calculate MAE-HR between PPG and ECG
    '''
    average_hr_og = []; average_hr_gn = []
    for i in range(ppg.shape[0]):

        a = average_hr_ECG(ecg[i],sampling_rate)
        b = average_hr_PPG(ppg[i],sampling_rate)

        average_hr_og.append(a)
        average_hr_gn.append(b)
    
    iqr,q75,q25 = get_iqr(average_hr_og,75,25)
    average_hr_gn =np.array(average_hr_gn)[average_hr_og < (q75 + 1.5*iqr)] ; average_hr_og=np.array(average_hr_og)[average_hr_og < (q75 + 1.5*iqr)]
    average_hr_gn =np.array(average_hr_gn)[(q25- 1.5*iqr) < average_hr_og ] ; average_hr_og=np.array(average_hr_og)[(q25- 1.5*iqr) < average_hr_og]

    mae = mean_absolute_error(average_hr_og, average_hr_gn)
    print(f"Mean absolute error for {window_size} second window:", mae)
    return mae


def heartrates(ecg,ppg,test_ppg,window_size,sampling_rate=128):
    '''
    Calculate MAE-HR between PPG and ECG
    '''
    average_hr_og = []; average_hr_gn = []
    for i in range(ppg.shape[0]):

        a = average_hr_ECG(ecg[i],sampling_rate)
        b = average_hr_PPG(ppg[i],sampling_rate)

        average_hr_og.append(a)
        average_hr_gn.append(b)
    
    iqr,q75,q25 = get_iqr(average_hr_og,75,25);print(len(average_hr_gn),len(average_hr_og))
    average_hr_gn =np.array(average_hr_gn)[average_hr_og < (q75 + 1.5*iqr)] ; test_ppg =np.array(test_ppg)[average_hr_og < (q75 + 1.5*iqr)]; ppg =np.array(ppg)[average_hr_og < (q75 + 1.5*iqr)] ; ecg=np.array(ecg)[average_hr_og < (q75 + 1.5*iqr)]
    average_hr_og=np.array(average_hr_og)[average_hr_og < (q75 + 1.5*iqr)];    
    average_hr_gn =np.array(average_hr_gn)[(q25- 1.5*iqr) < average_hr_og ] ; test_ppg =np.array(test_ppg)[(q25- 1.5*iqr) < average_hr_og ] ;  ppg =np.array(ppg)[(q25- 1.5*iqr) < average_hr_og ] ; ecg=np.array(ecg)[(q25- 1.5*iqr) < average_hr_og]
    average_hr_og=np.array(average_hr_og)[(q25- 1.5*iqr) < average_hr_og];     
    
    mae = mean_absolute_error(average_hr_og, average_hr_gn)
    print(f"Mean absolute error for {window_size} second window:", mae)
    return average_hr_og,average_hr_gn,ppg,ecg,test_ppg

def evaluate(inp):
    '''
    Generate PPG representations using SynthPPG

    Returns:
        - Generated PPG Representations
        - Evaluated heart rates for each PPG segment
    '''
    generator = load_model()
    ip = generator(inp.reshape(-1,512)).numpy().reshape(inp.shape)
    op = np.zeros(inp.shape)
    heart_rates = np.zeros(ip.shape[0])
    for i in range(ip.shape[0]):
        op[i] = filter_ppg(signal=ip[i],sampling_rate=128,frequency=[0.5,8])
        heart_rates[i] = average_hr_PPG(ppg_seg = op[i], sampling_rate = 128)
    return op, heart_rates