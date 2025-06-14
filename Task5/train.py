#! /usr/bin/python3
import sys
import random
from contextlib import redirect_stdout

import keras

from transformer import TransformerBlock, TokenAndPositionEmbedding

from tensorflow.keras import regularizers, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, concatenate, Flatten, Bidirectional, LSTM, LeakyReLU
from tensorflow.keras.initializers import Constant

from dataset import *
from codemaps import *

def build_network(codes):
    # Sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()
    emb_dim = 100

    # Input and Embedding
    inptW = Input(shape=(max_len,))
    embW = Embedding(input_dim=n_words, output_dim=emb_dim, mask_zero=False)(inptW)

    # CNN layer (local feature extraction)
    conv = Conv1D(filters=30, kernel_size=3, activation='relu', padding='same')(embW)

    # BiLSTM layer 
    lstm = Bidirectional(LSTM(units=300))(conv)

    # Dense + LeakyReLU + Dropout
    dense = Dense(100)(lstm)
    act = LeakyReLU(alpha=0.1)(dense)
    drop = Dropout(0.2)(act)

    # Output layer
    out = Dense(n_labels, activation='softmax')(drop)

    # Compile model
    model = Model(inputs=inptW, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
   


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --


# directory with files to process
trainfile = sys.argv[1]
validationfile = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
suf_len = 5
codes = Codemaps(traindata, max_len)

# build network
model = build_network(codes)
with redirect_stdout(sys.stderr) :
   model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)

Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr) :
   model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv,Yv), verbose=1)
   
# save model and indexs
model.save(modelname)
codes.save(modelname)

