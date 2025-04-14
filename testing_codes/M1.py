import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU, Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Dropout, Reshape, TimeDistributed, add, Bidirectional
from tensorflow.keras import Model, Sequential
from sklearn.manifold import TSNE as tsne
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa
from scipy.io import wavfile
import csv
import random
import os,sys
import math
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import mir_eval
import gc
import matplotlib.pyplot as plt
import time
import pdb
import keras
from tqdm import tqdm
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.stats import norm

os.environ["CUDA_VISIBLE_DEVICES"]="2" #0
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#################################################

test_audio_files = ' ' #path of audio data
test_pitch_files = ' ' #path of pitch data in Hz

#################################################
batch_size = 16
#################################################

Nfft = 2048
win_len = 2048
hop_len = 160
win_size = 100

freq_min = 51.91
freq_max = 830.61
B = 96  # no. of semitones

num_semitones = int(B*np.log2(freq_max/freq_min))
bin_borders = []

for i in range(num_semitones+1):
    bin_borders.append(freq_min*np.power(2,i/B))
bin_borders_log = [np.log2(i/freq_min) for i in bin_borders]

sigma = bin_borders_log[1]-bin_borders_log[0]
sigma = np.round(sigma,5)

prepend_values = [bin_borders_log[0] - sigma * i for i in range(50, 0, -1)]
bin_borders_log = prepend_values + bin_borders_log

num_bins = len(bin_borders_log)
zero_rep_bin = bin_borders_log[0]

#################################################

def preprocess_wav(wav_path):
    """
    Loads a spectrogram from a .npy file.

    Parameters
    ----------
    wav_path : tf.Tensor
        A TensorFlow string tensor representing the path to a .npy file containing the spectrogram.

    Returns
    -------
    np.ndarray
        A NumPy array containing the spectrogram data.
    """

    X = np.load(wav_path.numpy().decode())
    return X

def preprocess_pitch(pitch_path):
    """
    Loads pitch values in Hz from a .npy file.

    Parameters
    ----------
    pitch_path : tf.Tensor
        A TensorFlow string tensor representing the path to a .npy file containing pitch values.

    Returns
    -------
    np.ndarray
        A NumPy array containing pitch values in Hertz (Hz).
    """
    
    X = np.load(pitch_path.numpy().decode())
    return X

#################################################

class ResNet_block(Model):
    """
    A custom ResNet-style convolutional block with four convolutional layers, 
    batch normalization, LeakyReLU activations, and a residual connection.

    The block performs the following operations:
    - 1x1 Conv -> BN -> LeakyReLU
    - 3x3 Conv -> BN -> LeakyReLU
    - 3x3 Conv -> BN -> LeakyReLU
    - 1x1 Conv -> BN
    - Add residual connection (after first BN) to output of final BN
    - Final LeakyReLU activation and max pooling along the width (1x4)

    Parameters
    ----------
    filters : int
        The number of filters to use in each convolutional layer.

    Methods
    -------
    call(input_tensor)
        Forward pass through the ResNet block.
    
    build_graph(raw_shape)
        Builds and returns a Keras Model with the given input shape.
    """

    def __init__(self,filters):
        super().__init__()
        self.conv1 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU(0.01) 
        self.conv2 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU(0.01) 
        self.conv3 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn3 = BatchNormalization()
        self.act3 = LeakyReLU(0.01) 
        self.conv4 = Conv2D(filters, (1, 1), padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn4 = BatchNormalization()
        self.act4 = LeakyReLU(0.01) 
        self.add = tf.keras.layers.Add()
        
        
    def call(self,input_tensor):
        x = self.conv1(input_tensor)
        shortcut = self.bn1(x)
        x = self.act1(shortcut)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = self.add([x, shortcut])
        x = self.act4(x)
        x = MaxPooling2D((1, 4))(x)
        return x

    def build_graph(self,raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x))

     
class melody_extraction(Model):
    """
    A melody extraction model using stacked ResNet blocks and a TimeDistributed dense output layer.

    This model processes an input spectrogram through a series of convolutional ResNet blocks to extract
    high-level features. These features are reshaped and passed through a TimeDistributed Dense layer 
    with softmax activation to produce frame-wise pitch probability distributions.

    Architecture:
    -------------
    - 4 ResNet-style convolutional blocks with increasing filter sizes.
    - Reshape layer to flatten spatial dimensions for each time window.
    - Optional Bidirectional LSTM for temporal modeling (currently commented out).
    - TimeDistributed Dense layer for outputting probability distribution over frequency bins.

    Returns both:
    - Final pitch probability predictions (softmax over `num_bins - 1`).
    - Intermediate features prior to classification.

    Attributes
    ----------
    rb1, rb2, rb3, rb4 : ResNet_block
        Convolutional residual blocks with 32, 64, 128, and 256 filters respectively.
        
    td1 : TimeDistributed
        TimeDistributed Dense layer with softmax activation for pitch classification.

    Methods
    -------
    call(x)
        Forward pass through the model. Returns predicted pitch probabilities and intermediate features.
    
    build_graph(raw_shape)
        Builds and returns a Keras model instance using the defined architecture, 
        useful for visualizing model summaries and plotting.
    """

    def __init__(self):
        super().__init__()
        self.rb1 = ResNet_block(32)
        self.rb2 = ResNet_block(64)
        self.rb3 = ResNet_block(128)
        self.rb4 = ResNet_block(256)
        self.td1 = TimeDistributed(Dense(num_bins-1,activation='softmax'))
       
    def call(self,x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        
        P = x.shape[2] * x.shape[3]
        intermediate = Reshape((win_size, P))(x)
        x = self.td1(intermediate)
        return x,intermediate

    def build_graph(self,raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x))


model = melody_extraction()
model.build_graph([win_size,int(Nfft/2)+1,1])#.summary()
model.load_weights('../model_weights/M1/weights')

#################################################

l_rate = 1.e-4
loss_fn = keras.losses.CategoricalCrossentropy()
test_acc_metric = keras.metrics.CategoricalAccuracy()

#################################################

def findpeaks(fx):
    """
    Identifies and returns the indices of peaks in a 1D array.

    A peak is defined as a point that is higher than its immediate neighbors. 
    This function prepends a 0 to the input array to align indexing, then adjusts 
    the indices of the detected peaks accordingly.

    Parameters
    ----------
    fx : np.ndarray
        A 1D NumPy array of numerical values in which to find peaks.

    Returns
    -------
    np.ndarray
        An array of indices corresponding to the positions of the peaks in the original input array.
    """

    fx = np.insert(fx,0,0)
    peaks, _ = find_peaks(fx, height=0)
    peaks = peaks-1
    return peaks

def find_max_peak(indx,pred_fx):
    """
    Finds the index of the highest peak from a list of peak indices.

    Parameters
    ----------
    indx : np.ndarray
        An array of indices representing the positions of detected peaks.
    
    pred_fx : np.ndarray
        A 1D array of values (e.g., prediction amplitudes) from which the peaks were identified.

    Returns
    -------
    int
        The index corresponding to the peak with the maximum value in `pred_fx`.
    """

    indxs = indx[np.argmax(pred_fx[indx])]
    return indxs


def get_scaled_up_probabilties(f, indices_to_zero):
    """
    Sets specified indices in a probability distribution to zero and rescales the remaining values to sum to 1.

    This function is useful when certain probabilities (e.g., at specific indices) need to be excluded 
    from consideration, and the remaining values should be re-normalized to maintain a valid probability distribution.

    Parameters
    ----------
    f : np.ndarray
        A 1D NumPy array representing a probability distribution (should sum to 1 before modification).
    
    indices_to_zero : array-like
        Indices in `f` that should be set to zero (i.e., excluded from the distribution).

    Returns
    -------
    np.ndarray
        A new probability distribution where the specified indices are zeroed out and 
        the remaining values are scaled to sum to 1.

    Raises
    ------
    ValueError
        If the sum of the remaining (non-zeroed) values is zero, making normalization impossible.
    """

    f[indices_to_zero] = 0  # make q at ksup = 0  
    remaining_sum = np.sum(f)  # 1-p(ksup)
    if remaining_sum > 0:
        f = f / remaining_sum
    else:
        raise ValueError("The sum of the remaining probabilities is zero.")
    return f

    
def find_pruned_expected_val(fx):
    """
    Computes the pruned expected frequency values in Hz from predicted probability distributions over frequency bins.

    This function processes a 3D array of predicted bin probabilities, identifies peaks in each distribution,
    prunes undesired regions based on peak comparisons, rescales the remaining probabilities, and computes the 
    expected frequency in the log domain. The final expected values are then converted to frequencies in Hz.

    Peak pruning logic is applied based on whether the peak at index 0 has greater, lesser, or equal probability 
    compared to the maximum peak. The corresponding region around the less relevant peaks is zeroed out and the 
    distribution is renormalized.

    Parameters
    ----------
    fx : tf.Tensor
        A 3D TensorFlow tensor of shape (batch_size, time_steps, num_bins) representing predicted 
        probability distributions over frequency bins for each frame.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (batch_size, time_steps) containing the predicted frequency values in Hz 
        after pruning and expected value computation.

    Raises
    ------
    ValueError
        If, during pruning, the sum of the remaining probabilities becomes zero (handled inside get_scaled_up_probabilities).
    """

    #fx --> predicted bin prob, expected_val --> predicted log freq, ypred --> predicted freq hz

    fx = fx.numpy()
    expected_val = np.zeros((np.shape(fx)[0],np.shape(fx)[1]),dtype=float)

    for k in range(np.shape(fx)[0]):
        for j in range(np.shape(fx)[1]):
            peaks = findpeaks(fx[k][j])

            if 0 in peaks and len(peaks)>1:
                max_peak = find_max_peak(peaks,fx[k][j])

                if fx[k][j][0]>fx[k][j][max_peak]:
                    fx[k][j] = get_scaled_up_probabilties(fx[k][j],np.arange(max_peak-10,max_peak+10))

                elif fx[k][j][0]<fx[k][j][max_peak]:
                    fx[k][j] = get_scaled_up_probabilties(fx[k][j],np.arange(10))
                
                elif fx[k][j][0]==fx[k][j][max_peak]:
                    for p in peaks[1:]:
                        if 2<=p<=10:
                            fx[k][j] = get_scaled_up_probabilties(fx[k][j],np.arange(p-1,p+10))
                        elif 400<p<435:
                            fx[k][j] = get_scaled_up_probabilties(fx[k][j],np.arange(p-10,p))
                        else:
                            fx[k][j] = get_scaled_up_probabilties(fx[k][j],np.arange(p-10,p+10))

            expected_val[k,j] = sum([(bin_borders_log[i])*f for i,f in enumerate(fx[k][j])])
                    
    freq_hz = freq_min * np.power(2,expected_val)  ## convert log values to hz
    threshold = freq_min*np.power(2,-0.5/B)
    freq_hz[freq_hz<threshold] = 0
    return freq_hz


def find_expected_val(fx):
    """
    Computes the expected frequency values in Hz from predicted probability distributions over frequency bins.

    This function calculates the expected value for each probability distribution over frequency bins 
    (in the log-frequency domain), converts it to Hz, and applies a lower threshold to remove 
    unrealistically low frequencies.

    Parameters
    ----------
    fx : tf.Tensor
        A 3D TensorFlow tensor of shape (batch_size, time_steps, num_bins), where each [k, j, :] slice 
        represents a probability distribution over frequency bins for a given frame.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (batch_size, time_steps) containing the expected frequency values in Hz.
    """


    fx = fx.numpy()
    expected_val = np.zeros((np.shape(fx)[0],np.shape(fx)[1]),dtype=float)
    for k in range(np.shape(fx)[0]):
        for j in range(np.shape(fx)[1]):
            expected_val[k,j] = sum([(bin_borders_log[i])*f for i,f in enumerate(fx[k][j])]) 
    
    expected_val = freq_min * np.power(2,expected_val)  ## convert log values to hz
    threshold = freq_min*np.power(2,-0.5/B)
    expected_val[expected_val<threshold] = 0
    return expected_val
    

def test_step(x):
    with tf.GradientTape() as tape:
        pred_fx,_ = model.call(x)
        # efv = find_pruned_expected_val(pred_fx) # can predict before and after pruning P(M1)
        efv = find_expected_val(pred_fx)
    return efv


def calc_rpa(y,yp):
    rpa = []
    rca = []
    oa = []
    for i in range(tf.shape(y)[0]):
        gfv = np.array(y[i])
        efv = np.array(yp[i])  
        
        t = np.array([i*0.01 for i in range(len(gfv))])
        
        (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(t,gfv,t,efv)

        RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
        RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
        OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)
        rpa.append(RPA)
        rca.append(RCA)
        oa.append(OA)
    rpa=np.mean(np.array(rpa))    
    rca=np.mean(np.array(rca))    
    oa=np.mean(np.array(oa))  
    return rpa,rca,oa
    
################################################# 

mean = np.load('../total_mean.npy')
std = np.load('../total_std.npy')

#################################################

rpa_test = []
rca_test = []
oa_test = []

test_dataset = tf.data.Dataset.from_tensor_slices((test_audio_files,test_pitch_files)) 
test_dataset = test_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32), tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

for step,batch in enumerate(test_dataset):
    x,gfv = batch
    x = (x-mean)/std        
    x = x[:,:,:,tf.newaxis]
    efv = test_step(x)
    rpa,rca,oa = calc_rpa(gfv,efv)

    rpa_test.append(rpa)
    rca_test.append(rca)
    oa_test.append(oa)

print(f'Metrics on the test dataset:  RPA-{np.mean(np.array(rpa_test))} RCA-{np.mean(np.array(rca_test))} OA-{np.mean(np.array(oa_test))}')
