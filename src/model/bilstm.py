from tensorflow.keras import *
from tensorflow.keras.layers import *

def BiLSTM(input_shape, hidden_dim=128, num_classes=None):
    input_layer = Input(input_shape)
    x = Bidirectional(LSTM(units=hidden_dim,
                           return_sequences=True,
                           recurrent_dropout=0.25,
                           dropout=0.50))(input_layer)
    x = Bidirectional(LSTM(units=hidden_dim,
                           return_sequences=True,
                           recurrent_dropout=0.25,
                           dropout=0.50))(x)
    x = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    return Model(inputs=input_layer, outputs=x)
