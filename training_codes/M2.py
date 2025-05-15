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
import tensorflow_probability as tfp
from scipy.signal import find_peaks

tfd = tfp.distributions

os.environ["CUDA_VISIBLE_DEVICES"]="1" #0
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#################################################

train_audio_files = ' ' #path of audio data 
train_pitch_files = ' ' #path of pitch data in Hz

val_audio_files = ' ' #path of audio data
val_pitch_files = ' ' #path of pitch data in Hz

print('Total train audio and pitch files:',len(train_audio_files), len(train_pitch_files),'\n')
print('Total val audio and pitch files:',len(val_audio_files),len(val_pitch_files),'\n')

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
print(f'No. of semitones..{num_semitones}')

bin_borders = []

for i in range(num_semitones+1):
    bin_borders.append(freq_min*np.power(2,i/B))
bin_borders_log = [np.log2(i/freq_min) for i in bin_borders]
print(bin_borders_log[0],bin_borders_log[-1])

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
        self.bi = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))
        self.td1 = TimeDistributed(Dense(num_bins-1,activation='softmax'))
       
    def call(self,x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        
        P = x.shape[2] * x.shape[3]
        intermediate = Reshape((win_size, P))(x)
        # x = self.bi(x)
        x = self.td1(intermediate)
        return x,intermediate

    def build_graph(self,raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x))


model = melody_extraction()
model.build_graph([win_size,int(Nfft/2)+1,1])

#################################################

l_rate = 1.e-5
optimizer = keras.optimizers.Adam(learning_rate=l_rate) 
loss_fn = keras.losses.CategoricalCrossentropy()

# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

epochs = 100
#################################################

def get_bin_probs(yt, num_bins, sd):
    """
    Computes binned probabilities from a truncated normal distribution over log-scaled bin edges.

    Given predicted target values (`yt`) and standard deviations (`sd`), this function models each prediction
    as a truncated normal distribution and computes the probability mass within specified bins. The bins are 
    defined by global `bin_borders_log`, assumed to be a log-spaced array of bin edges.

    Parameters:
    yt (tf.Tensor): Tensor of predicted target values
    num_bins (int): Number of bins used for computing the probabilities.
    sd (tf.Tensor): Tensor of standard deviations corresponding to `yt`. Must be same shape as `yt`.

    Returns:
    tf.Tensor: Tensor of bin probabilities with shape (*yt.shape, num_bins). Each slice along the last 
               dimension represents the probability mass falling into each bin for a corresponding prediction.
    """

    bin_borders_log_tf = tf.convert_to_tensor(bin_borders_log, dtype=tf.float32)

    # Expand bin_borders_log to match yt shape
    bin_borders_expanded = tf.reshape(bin_borders_log_tf[:-1], [1, 1, -1])  # Shape (1,1,10)

    # Define TruncatedNormal distribution
    source_tar_dist = tfd.TruncatedNormal(
        loc= yt[..., tf.newaxis],  # Expand yt to match bin shape (2,4,1)
        scale= sd[..., tf.newaxis],  
        low= tf.cast(bin_borders_log_tf[0], dtype=tf.float32),
        high= tf.cast(bin_borders_log_tf[-1], dtype=tf.float32)
    )


    # Compute CDF at bin edges
    left_bound_cdf = source_tar_dist.cdf(bin_borders_expanded) 
    right_bound_cdf = source_tar_dist.cdf(tf.reshape(bin_borders_log_tf[1:], [1, 1, -1]))

    # Compute bin probabilities
    bin_probs_mod = right_bound_cdf - left_bound_cdf  # Shape: (2,4,10)

    return bin_probs_mod


def custom_loss(pred_dist,gfv):
    """
    Computes a custom loss between predicted and ground truth distributions with class imbalance weighting.

    This function performs the following steps:
    1. Converts ground truth frequency values (`gfv`) to a log2-scaled target (`yt`) with 0 Hz mapped to zero_rep_bin = -0.521
    2. Computes the expected mean from the predicted distribution (`pred_dist`).
    3. Calculates a standard deviation based on the difference between predicted mean and target.
    4. Constructs a ground truth distribution (`gt_dist`) using a truncated normal approximation via `get_bin_probs`.
    5. Applies dynamic class weights based on the presence or absence of non-zero ground truth values.
    6. Computes the weighted histogram loss between predicted and ground truth distributions using a global `loss_fn`.

    Parameters:
    pred_dist (tf.Tensor): The predicted distribution over bins.
    gfv (tf.Tensor): Ground truth frequency values

    Returns:
    tf.Tensor: The computed weighted loss value.
    """

    mean_yp = tf.reduce_sum(bin_borders_log[:-1] * pred_dist, axis=-1)

    # Step 1: Compute log2(freq) only for nonzero values
    yt = tf.where(gfv > 0, tf.math.log(gfv/freq_min) / tf.math.log(2.0), gfv)  # log2 transformation
    
    # Step 2: Replace zero values with the constant C
    yt = tf.where(gfv == 0, tf.fill(tf.shape(gfv), tf.cast(zero_rep_bin, dtype=tf.float32)), yt)

    std_dev = tf.stop_gradient(tf.cast(tf.sqrt((mean_yp - yt)**2),dtype=tf.float32))
    std_dev = tf.maximum(std_dev,0.0001)
    gt_dist = get_bin_probs(yt,num_bins,std_dev) 

    vd = tf.cast(tf.where(gfv!=0, tf.ones_like(gfv),gfv),dtype=tf.int32)
    count_ones = tf.math.count_nonzero(vd)
    count_zeros = np.size(vd) - count_ones
    
    if count_zeros==0:
        class_weights = [0,1]
    elif count_ones==0:
        class_weights = [1,0]
    else:
        class_counts = [count_zeros,count_ones]
        class_weights = [max(class_counts)/count_zeros, max(class_counts)/count_ones]
    
    w = tf.gather(class_weights,vd)
    loss = loss_fn(gt_dist, pred_dist, sample_weight=w)
    return loss


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


def train_step(x,gfv):
    with tf.GradientTape() as tape:
        pred_fx,_ = model.call(x)
        loss = custom_loss(pred_fx,gfv)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    efv = find_expected_val(pred_fx)
    return loss,efv

def test_step(x,gfv):
    with tf.GradientTape() as tape:
        pred_fx,_ = model.call(x)
        loss = custom_loss(pred_fx,gfv)
        efv = find_expected_val(pred_fx)
    return loss,efv

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

val_dataset = tf.data.Dataset.from_tensor_slices((val_audio_files,val_pitch_files)) 
val_dataset = val_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32), tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

#################################################
mean = np.load('../total_mean.npy')
std = np.load('../total_std.npy')

weights_path = '../model_weights/M2/'

os.makedirs(weights_path, exist_ok=True)

filepath = weights_path + 'weights'

#################################################
    
rpa_train_epoch = []
rca_train_epoch = []
oa_train_epoch = []

rpa_val_epoch = []
rca_val_epoch = []
oa_val_epoch = []

train_loss_epoch = []
val_loss_epoch = []

k = 1
for epoch in tqdm(range(epochs)):
    print(f'Epoch...{epoch+1}')  
    tot_time = np.array([])   
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audio_files,train_pitch_files)).shuffle(len(train_audio_files))
    train_dataset = train_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32), tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size) 
      
    rpa_train_batch = []  
    rca_train_batch = []  
    oa_train_batch = []  

    loss_train_batch = []
    loss_val_batch = []
    
    for batch in train_dataset:
        x,gfv = batch
        x = (x-mean)/std        
        x = x[:,:,:,tf.newaxis]
        loss_value,efv = train_step(x,gfv)
        rpa,rca,oa = calc_rpa(gfv,efv)
        rpa_train_batch.append(rpa)
        rca_train_batch.append(rca)
        oa_train_batch.append(oa)
        loss_train_batch.append(loss_value)

    print(f'Training Loss per epoch..{np.mean(np.array(loss_train_batch))}')  
    train_loss_epoch.append(np.mean(np.array(loss_train_batch)))
    print(f'Train data - RPA : {np.mean(np.array(rpa_train_batch))} RCA : {np.mean(np.array(rca_train_batch))} OA : {np.mean(np.array(oa_train_batch))}')
    rpa_train_epoch.append(np.mean(np.array(rpa_train_batch)))
    rca_train_epoch.append(np.mean(np.array(rca_train_batch))) 
    oa_train_epoch.append(np.mean(np.array(oa_train_batch))) 
 

    rpa_val_batch = []
    rca_val_batch = []
    oa_val_batch = []

    for batch in val_dataset:
        x,gfv = batch
        x = (x-mean)/std 
        x = x[:,:,:,tf.newaxis]
        val_loss,efv = test_step(x,gfv)
        rpa,rca,oa = calc_rpa(gfv,efv)
        rpa_val_batch.append(rpa)
        rca_val_batch.append(rca)
        oa_val_batch.append(oa)
        loss_val_batch.append(val_loss)

    print(f'Validation Loss per epoch..{np.mean(np.array(loss_val_batch))}')  
    val_loss_epoch.append(np.mean(np.array(loss_val_batch)))
    print(f'Validation data - RPA : {np.mean(np.array(rpa_val_batch))} RCA : {np.mean(np.array(rca_val_batch))} OA : {np.mean(np.array(oa_val_batch))}')
    rpa_val_epoch.append(np.mean(np.array(rpa_val_batch)))
    rca_val_epoch.append(np.mean(np.array(rca_val_batch)))
    oa_val_epoch.append(np.mean(np.array(oa_val_batch)))


    if (epoch+1)%20==0:
        l_rate = np.round(math.pow(0.90,k),3) * l_rate
        optimizer = keras.optimizers.Adam(learning_rate=l_rate)
        k+=1

model.save_weights(filepath,save_format='tf')
print('Model Saved')        

