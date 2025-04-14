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

os.environ["CUDA_VISIBLE_DEVICES"]="2" #0
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#################################################

test_audio_files = ' ' #path of audio data
test_pitch_files = ' ' #path of pitch data in Hz

#################################################

batch_size = 32

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
num_bins = len(bin_borders_log)
print(f'Number of bins...{num_bins}')

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

     
class melody(Model):
    """
    A dual-branch neural network model for melody extraction, predicting both voicing and pitch class.

    This model processes input spectrogram-like features using stacked ResNet blocks to extract 
    hierarchical representations. It then reshapes the features and feeds them into two parallel 
    branches:

    - **Voicing Branch**: A Dense layer with sigmoid activation that predicts the probability 
      of whether a pitch is voiced (i.e., present) for each frame.
    
    - **F0 Branch**: A TimeDistributed Dense layer with softmax activation that outputs the 
      probability distribution over pitch bins (excluding unvoiced).

    Architecture:
    -------------
    - 4 ResNet-style convolutional blocks with increasing filter sizes (32 → 256).
    - Reshape layer to convert convolutional outputs to a sequence format.
    - Voicing branch for binary classification (voiced/unvoiced).
    - Pitch (f0) branch for multi-class pitch prediction.

    Attributes
    ----------
    rb1, rb2, rb3, rb4 : ResNet_block
        Residual convolutional blocks for feature extraction.

    voicing_branch : tf.keras.layers.Dense
        Dense layer with sigmoid activation to predict voicing per time frame.

    f0_branch : tf.keras.layers.TimeDistributed
        TimeDistributed Dense layer with softmax activation to predict pitch class probabilities.

    Methods
    -------
    call(x)
        Runs a forward pass through the model and returns:
        - `vd`: Voicing predictions (shape: [batch_size, time_steps, 1])
        - `pd`: Pitch bin predictions (shape: [batch_size, time_steps, num_bins - 1])
    
    build_graph(raw_shape)
        Utility method to create and return a `tf.keras.Model` instance for visualization or summary.

    """
    def __init__(self):
        super().__init__()
        self.rb1 = ResNet_block(32)
        self.rb2 = ResNet_block(64)
        self.rb3 = ResNet_block(128)
        self.rb4 = ResNet_block(256)
        self.voicing_branch = Dense(1,activation='sigmoid')
        self.f0_branch = TimeDistributed(Dense(num_bins-1,activation='softmax'))
       
    def call(self,x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        
        P = x.shape[2] * x.shape[3]
        intermediate = Reshape((win_size, P))(x)
        vd = self.voicing_branch(intermediate)
        pd = self.f0_branch(intermediate)
        return vd, pd

    def build_graph(self,raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x))
    

model = melody()
model.build_graph([win_size,int(Nfft/2)+1,1])#.summary()
model.load_weights('../model_weights/M3/weights')
#################################################

l_rate = 1.e-4
optimizer = keras.optimizers.Adam(learning_rate=l_rate) 
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()

#################################################

def calculate_expected_value(pred_vd,pred_probs):
    """
    Computes the expected frequency (in Hz) from predicted pitch class probabilities,
    conditioned on the voicing detection output.

    This function calculates the expected value over frequency bins (in log2 domain) using
    a weighted sum of predicted pitch class probabilities. The final expected frequency is
    converted to Hz and masked based on the voicing predictions — only values with 
    voicing probability ≥ 0.5 are retained.

    Parameters
    ----------
    pred_vd : tf.Tensor
        A tensor containing voicing predictions (values in [0, 1]).

    pred_probs : tf.Tensor
        A 3D tensor of shape (batch_size, time_steps, num_bins - 1), representing the
        predicted pitch class probability distribution per time frame.

    Returns
    -------
    tf.Tensor
        A 2D tensor of shape (batch_size, time_steps) containing expected frequency values in Hz.
        Frequencies are set to 0 where voicing probability is below 0.5.
    """

    pred_vd = tf.reshape(pred_vd, [tf.shape(pred_vd)[0], -1]) 
    
    weighted_sum = tf.reduce_sum(bin_borders_log[:-1] * pred_probs, axis=-1) 
    # Compute expected value using vectorized operations
    exp_val = freq_min * tf.pow(2.0, weighted_sum)

    # Apply mask for pred_vd >= 0 - voicing detection
    mask = tf.cast(pred_vd >= 0.5, tf.float32)
    exp_val = exp_val * mask
    return exp_val


def test_step(x):
    vd_pred,pd_probs_pred = model.call(x)
    efv = calculate_expected_value(vd_pred,pd_probs_pred)
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
    return rpa, rca, oa
    
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


for batch in tqdm(test_dataset):
    x,gfv = batch
    x = (x-mean)/std 
    x = x[:,:,:,tf.newaxis]
    efv = test_step(x)
    rpa,rca,oa = calc_rpa(gfv,efv)

    rpa_test.append(rpa)
    rca_test.append(rca)
    oa_test.append(oa)

print(f'Metrics:  RPA-{np.mean(np.array(rpa_test))} RCA-{np.mean(np.array(rca_test))} OA-{np.mean(np.array(oa_test))}')


