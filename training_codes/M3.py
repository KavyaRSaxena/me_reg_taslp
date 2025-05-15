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
import time
import pdb
import keras
from tqdm import tqdm
import tensorflow_probability as tfp


tfd = tfp.distributions


os.environ["CUDA_VISIBLE_DEVICES"]="0" #0
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#################################################

train_audio_files = sorted(glob('/hdd_storage/data/kavyars2/Datasets/SPL/npy_data/train/audio/*.npy'))
train_pitch_files = sorted(glob('/hdd_storage/data/kavyars2/Datasets/SPL/npy_data/train/pitch/*.npy'))

val_audio_files = sorted(glob('/hdd_storage/data/kavyars2/Datasets/SPL/npy_data/val/audio/*.npy'))
val_pitch_files = sorted(glob('/hdd_storage/data/kavyars2/Datasets/SPL/npy_data/val/pitch/*.npy'))

print('Total train audio and pitch files:',len(train_audio_files), len(train_pitch_files),'\n')
print('Total val audio and pitch files:',len(val_audio_files), len(val_pitch_files),'\n')

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

num_bins = len(bin_borders_log)

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
model.build_graph([win_size,int(Nfft/2)+1,1])

#################################################

l_rate = 1.e-5
optimizer = keras.optimizers.Adam(learning_rate=l_rate) 
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()

epochs = 100
#################################################

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Custom weighted binary cross-entropy loss for handling class imbalance.

    This loss function applies different weights to positive and negative classes during training,
    which is useful when dealing with imbalanced datasets. It also includes clipping to prevent
    log(0) issues.

    Parameters:
    weight_pos (float): Weight for the voiced positive class (label = 1).
    weight_neg (float): Weight for the unvoiced negative class (label = 0).
    epsilon (float): Small constant to avoid log(0) by clipping predicted values.

    Methods:
    call(y_true, y_pred):
        Computes the weighted binary cross-entropy loss.

        Parameters:
        y_true (tf.Tensor): Ground truth binary labels (0 or 1).
        y_pred (tf.Tensor): Predicted probabilities. Shape must match `y_true`.

        Returns:
        tf.Tensor: Scalar tensor representing the mean weighted loss.
    """

    def __init__(self, weight_pos, weight_neg,epsilon):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred,[y_pred.shape[0],y_pred.shape[1]])
        y_true = tf.cast(y_true, tf.float32) 
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        loss = - (self.weight_pos * y_true * tf.math.log(y_pred) + 
                  self.weight_neg * (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)


def calc_weights():
    """
    Calculates class weights for binary classification of voiced and unvoiced frames.

    This function iterates over a list of training pitch files (`train_pitch_files`), where each file contains 
    a NumPy array of ground truth frequency values (`gfv`). Non-zero values are treated as voiced (1), 
    and zeros as unvoiced (0). It computes the total number of voiced and unvoiced frames, then derives 
    class weights inversely proportional to their frequencies.

    Returns:
    list[float]: A list of two weights [weight_unvoiced, weight_voiced], where the more frequent class 
                 receives a smaller weight to balance the loss contribution during training.

    Notes:
    - Assumes `train_pitch_files` is a global list of file paths to `.npy` files containing pitch values.
    - Useful for addressing class imbalance in voice activity detection or pitch modeling tasks.
    """

    class_unvoiced = 0
    class_voiced = 0
    for i in range(len(train_pitch_files)):
        gfv = np.load(train_pitch_files[i])
        gfv[gfv!=0] = 1
        class_voiced+= np.count_nonzero(gfv)
        class_unvoiced+= np.size(gfv) - np.count_nonzero(gfv)
    
    class_counts = [class_unvoiced, class_voiced]
    class_weights = [max(class_counts)/class_unvoiced, max(class_counts)/class_voiced]
    return class_weights


weights = calc_weights()

epsilon = 1e-5
bce_loss_fn = WeightedBinaryCrossEntropy(weights[1], weights[0], epsilon) #voiced (positive), unvoiced (negative)
cce_loss_fn = keras.losses.CategoricalCrossentropy()


def get_bin_probs(gfv_log,num_bins,std):
    """
    Computes bin probabilities from a Truncated Normal distribution over log-scaled frequency values.

    This function models the ground truth frequency values (`gfv_log`) as a Truncated Normal distribution 
    and computes the probability mass within predefined bins. The bins are defined by global `bin_borders_log` 
    which represent the log-space edges for the bins.

    Parameters:
    gfv_log (tf.Tensor): Ground truth frequency values in log-scale.
    num_bins (int): Number of bins used for computing the probabilities.
    std (tf.Tensor): Standard deviation values.

    Returns:
    np.ndarray: A NumPy array containing the bin probabilities. Shape: [batch, num_frames, num_bins-1].
    """

    bin_borders_log_tf = tf.convert_to_tensor(bin_borders_log, dtype=tf.float32)
    gfv_log = tf.where(gfv_log > 0, gfv_log, tf.constant(np.nan, dtype=tf.float32))

    bin_borders_exp = tf.reshape(bin_borders_log_tf, [1, 1, -1])  # Shape: [1, 1, num_bins]
    freq_log_exp = tf.expand_dims(gfv_log, axis=-1)  # Shape: [batch, num_frames, 1]
    std_exp = tf.expand_dims(std, axis=-1)  # Shape: [batch, num_frames, 1]

    # Create Truncated Normal distribution
    source_tar_dist = tfd.TruncatedNormal(
        loc=freq_log_exp,  # Shape: [batch, num_frames, 1]
        scale=std_exp,  # Shape: [batch, num_frames, 1]
        low=bin_borders_exp[:, :, 0],  # Shape: [1, 1]
        high=bin_borders_exp[:, :, -1]  # Shape: [1, 1]
    )
    
    # Compute CDF values for bin borders
    left_bound_cdf = source_tar_dist.cdf(bin_borders_exp[:, :, :-1])  # Shape: [batch, num_frames, num_bins-1]
    right_bound_cdf = source_tar_dist.cdf(bin_borders_exp[:, :, 1:])  # Shape: [batch, num_frames, num_bins-1]

    # Compute bin probabilities
    bin_probs = right_bound_cdf - left_bound_cdf  # Shape: [batch, num_frames, num_bins-1]

    # Replace NaNs with zeros (for ignored zero frequencies)
    bin_probs = tf.where(tf.math.is_nan(bin_probs), tf.zeros_like(bin_probs), bin_probs)

    return bin_probs.numpy()

    
def custom_loss(gfv,vd_pred,pd_probs):
    """
    Computes a custom loss combining binary cross-entropy and categorical cross-entropy with class imbalance.

    This function combines binary cross-entropy (BCE) and categorical cross-entropy (CCE) with dynamic weighting 
    to compute a final loss. The BCE focuses on voiced/unvoiced predictions (`vd_pred`), while CCE evaluates the 
    match between the predicted distribution (`pd_probs`) and the ground truth distribution, based on the 
    log-transformed frequency values (`gfv`). The loss also includes a dynamic weighting between BCE and CCE.

    Parameters:
    gfv (tf.Tensor): Ground truth frequency values.
    vd_pred (tf.Tensor): Predicted binary values for voiced/unvoiced classification.
    pd_probs (tf.Tensor): Predicted distribution probabilities for each bin.

    Returns:
    tf.Tensor: The computed total loss, which is a weighted combination of BCE and CCE.
    """

    vd_pred = tf.reshape(vd_pred,[vd_pred.shape[0],-1])
    vd_gfv = tf.cast(gfv>0,tf.float32)

    gfv_log = tf.where(gfv > 0, tf.math.log(gfv/freq_min) / tf.math.log(2.0), gfv)  # log2 transformation

    efv_log = tf.reduce_sum(bin_borders_log[:-1]*pd_probs, axis=-1)
    efv_log = efv_log * vd_gfv

    std_dev = tf.stop_gradient(tf.cast(tf.sqrt((efv_log - gfv_log)**2),dtype=tf.float32))
    std_dev = tf.maximum(std_dev,0.0001)

    gt_dist = get_bin_probs(gfv_log,num_bins,std_dev)


    bce_loss = bce_loss_fn(vd_gfv,vd_pred)
    cce_loss = cce_loss_fn(gt_dist,pd_probs,sample_weight=vd_gfv)

    scaling_factor = 0.6
    loss = bce_loss + (scaling_factor * cce_loss)  # Dynamic weighting
    return loss


def calculate_expected_value(pred_vd,pred_probs):
    """
    Computes the expected value of a frequency prediction based on predicted probabilities.

    This function computes the expected value for the predicted frequency values using a weighted sum of 
    log-scaled bin borders and predicted probabilities. The result is then scaled back using `freq_min` 
    and the inverse log transformation. A mask is applied to ensure that only the values for which the 
    voiced/unvoiced prediction (`pred_vd`) is above a certain threshold (0.5) are considered.

    Parameters:
    pred_vd (tf.Tensor): Predicted binary voiced/unvoiced values, used as a mask for valid predictions.
    pred_probs (tf.Tensor): Predicted probabilities for each bin.

    Returns:
    tf.Tensor: The computed expected frequency values, with shape: [batch, num_frames].
    """

    pred_vd = tf.reshape(pred_vd, [tf.shape(pred_vd)[0], -1]) 
    
    weighted_sum = tf.reduce_sum(bin_borders_log[:-1] * pred_probs, axis=-1) 
    # Compute expected value using vectorized operations
    exp_val = freq_min * tf.pow(2.0, weighted_sum)

    # Apply mask for pred_vd >= 0
    mask = tf.cast(pred_vd >= 0.5, tf.float32)
    exp_val = exp_val * mask
    return exp_val


def train_step(x,gfv):
    with tf.GradientTape() as tape:
        vd_pred,pd_probs_pred = model.call(x)
        loss = custom_loss(gfv,vd_pred,pd_probs_pred)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    efv = calculate_expected_value(vd_pred,pd_probs_pred)
    return loss, efv


def test_step(x,gfv):
    vd_pred,pd_probs_pred = model.call(x)
    loss = custom_loss(gfv,vd_pred,pd_probs_pred)
    efv = calculate_expected_value(vd_pred,pd_probs_pred)
    return loss, efv

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
val_dataset = val_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32),tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

#################################################

mean = np.load('../total_mean.npy')
std = np.load('../total_std.npy')

weights_path = '../model_weights/M3/' 

os.makedirs(weights_path,exist_ok=True)
filepath = weights_path + 'weights'

#################################################
    
rpa_update = []
rca_update = []
oa_update = []

rpa_val_update = []
rca_val_update = []
oa_val_update = []

train_loss = []
test_loss = []

k = 1
for epoch in range(epochs):
    print(f'Epoch...{epoch+1}')  
    tot_time = np.array([])   
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audio_files,train_pitch_files)).shuffle(len(train_audio_files))
    train_dataset = train_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32),tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size) 
      
    rpa_train_batch = []  
    rca_train_batch = []  
    oa_train_batch = [] 
    
    t_loss = []
    v_loss = []
    
    
    for step,batch in tqdm(enumerate(train_dataset)):
        x,gfv = batch
        x = (x-mean)/std        
        x = x[:,:,:,tf.newaxis]
        loss_value, efv = train_step(x,gfv)
        t_loss.append(loss_value)
        rpa,rca,oa = calc_rpa(gfv,efv)
        rpa_train_batch.append(rpa)
        rca_train_batch.append(rca)
        oa_train_batch.append(oa)
        
    train_time_end = time.time()
    print(f'Training Loss per epoch..{np.mean(np.array(t_loss))}')  
    train_loss.append(np.mean(np.array(t_loss)))
    print(f'Train data - RPA : {np.mean(np.array(rpa_train_batch))} RCA : {np.mean(np.array(rca_train_batch))} OA : {np.mean(np.array(oa_train_batch))}')
    rpa_update.append(np.mean(np.array(rpa_train_batch)))
    rca_update.append(np.mean(np.array(rca_train_batch))) 
    oa_update.append(np.mean(np.array(oa_train_batch))) 
        
    rpa_val_batch = []
    rca_val_batch = []
    oa_val_batch = []

    for step,batch in tqdm(enumerate(val_dataset)):
        x,gfv = batch
        x = (x-mean)/std 
        x = x[:,:,:,tf.newaxis]
        val_loss, efv = test_step(x,gfv)
        rpa,rca,oa = calc_rpa(gfv,efv)
        rpa_val_batch.append(rpa)
        rca_val_batch.append(rca)
        oa_val_batch.append(oa)
        v_loss.append(val_loss)
    

    print(f'Validation Loss per epoch..{np.mean(np.array(v_loss))}')  
    test_loss.append(np.mean(np.array(v_loss)))
    print(f'Test data - RPA : {np.mean(np.array(rpa_val_batch))} RCA : {np.mean(np.array(rca_val_batch))} OA : {np.mean(np.array(oa_val_batch))}')
    rpa_val_update.append(np.mean(np.array(rpa_val_batch)))
    rca_val_update.append(np.mean(np.array(rca_val_batch)))
    oa_val_update.append(np.mean(np.array(oa_val_batch)))
    
    
    if (epoch+1)%10==0:
        l_rate = np.round(math.pow(0.90,k),3) * l_rate
        optimizer = keras.optimizers.Adam(learning_rate=l_rate)
        k+=1
    
model.save_weights(filepath,save_format='tf')
print('Model Saved')    

        