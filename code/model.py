#This script contains the main content of the CRMSS model. 
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Activation, add, Convolution1D, Convolution2D, Dense, Dropout, Embedding
from keras.layers import GRU, Input
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention


def ConvBlock(input, filter_num, k_size):
    Conv_out = Convolution1D(filters=filter_num, kernel_size=k_size, padding='same')(input)
    Conv_out = BatchNormalization()(Conv_out)
    Conv_out = Activation('selu')(Conv_out)
    Conv_out = Dropout(0.5)(Conv_out)
    return Conv_out

def MultiScale(input): # residual block
    A = ConvBlock(input, 64, 1)
    B = ConvBlock(input, 64, 1)
    B = ConvBlock(B, 64, 3)  
    C = ConvBlock(input, 64, 1)
    C = ConvBlock(C, 64, 5)

    merge = Concatenate(axis=-1)([A, B, C])
  
    sc_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    sc_y = BatchNormalization()(sc_y)
    merge1 = add([sc_y, merge])
    result = Activation('selu')(merge1)
    return result

def get_model(embedding1):

    ### kmer embedding Glove
    embedding_input = Input(shape=(seq_len,), name='embedding_input1')
    embedding_embed = Embedding(input_length=seq_len, input_dim=embedding1.shape[0], weights=[embedding1],
                                output_dim=embedding1.shape[1], trainable=False)(embedding_input)
    sequence = Convolution1D(filters=64, kernel_size=7, padding='same')(embedding_embed)
    sequence = BatchNormalization(axis=-1)(sequence)

    profile_input = Input(shape=(101,5), name='profile_input')
    profile = Convolution1D(filters=64, kernel_size=4, padding='same')(profile_input)
    profile = BatchNormalization(axis=-1)(profile)

    fea_merge1 = Concatenate()([sequence, profile])
    fea_incepres = MultiScale(fea_merge1)

    fea_incepres = Convolution1D(filters=16, kernel_size=5, padding="valid", activation="selu")(fea_incepres)
    fea_re = keras.layers.Reshape(
        [fea_incepres.shape[1], fea_incepres.shape[2]])(fea_incepres)
    result_bigru1 = Bidirectional(GRU(64, return_sequences=True))(fea_re)
    result_attself = SeqSelfAttention(attention_activation='sigmoid')(result_bigru1)
    result_bilstm = Bidirectional(GRU(32))(result_attself)

    rbpfile_input = Input(shape=(168,), name='rbp_input')

    fea_merge2 = Concatenate()([result_bilstm, rbpfile_input])
    result_drop = Dropout(0.3)(fea_merge2)
    ss_output = Dense(2, activation='softmax', name='ss_output')(result_drop)

    return Model(inputs=[embedding_input, profile_input, rbpfile_input], outputs=[ss_output])
