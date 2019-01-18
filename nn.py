from sklearn.metrics import jaccard_similarity_score, f1_score, fbeta_score
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, SpatialDropout1D, GRU, GlobalMaxPool1D, SimpleRNN

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np

import warnings
warnings.filterwarnings('ignore')

################################################################################


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": 0.0}
tweet = "Tweet"


################################################################################


def split_xy(train_data, test_data):

    """
    Splits dataset into tokens & labels and computers the vocabulary size and 
    the padded sequence size.
    
    param data: (dataframe) Labelled Training or Testing set
    
    returns: Sequence of Tokens, Labels, Sequence Length, Vocabular Size
    """

    # Split the dataset into feature and labels
    train_X = train_data[tweet]
    train_Y = train_data[emotions]
    test_X  = test_data[tweet]
    test_Y  = test_data[emotions]

    # Define Tokens
    all_X     = pd.concat([train_X, test_X])
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(all_X)
    
    # Convert Tweets to Token Sequence
    train_token = tokenizer.texts_to_sequences(train_X)
    test_token  = tokenizer.texts_to_sequences(test_X)
    
    # Compute Sequence Size & Vocabulary Size
    maxlen     = max([len(x.split()) for x in all_X])
    vocab_size = len(tokenizer.word_index) + 1
    
    # Pad Token Sequences
    train_padded = pad_sequences(train_token, maxlen=maxlen)
    test_padded  = pad_sequences(test_token,   maxlen=maxlen)

    return train_padded, train_Y, test_padded, test_Y, maxlen, vocab_size


################################################################################


def lstm_model(vocab_size=2000, input_length=32):
    
    
    #############  HYPER-PARAMETER TUNING  #############
    
    # LSTM Parameters
    embed_dim         = 128
    lstm_units        = 64
    lstm2_units       = 64
    s_dropout         = 0.0  #0.1
    dropout           = 0.0  #0.1
    recurrent_dropout = 0.0  #0.1

    # Activation, Cost, Optimization, Metrics Parameters
    activation    = 'sigmoid'
    loss          = 'binary_crossentropy'
    optimizer     = 'adam'
    metrics       = ['accuracy']

    ####################################################


    model = Sequential()
    
    model.add( Embedding(vocab_size,
                         embed_dim,
                         input_length=input_length,
                         mask_zero=True))
    
    model.add(BatchNormalization())

    model.add( Bidirectional( GRU(lstm_units,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   return_sequences=True
                                  ) ))
    
    model.add( Bidirectional( GRU(lstm2_units,
                                  dropout=dropout,
                                  recurrent_dropout=recurrent_dropout) ))
    
    model.add(Dense(len(emotions), activation=activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    # Tracks all Callbacks
    callbacks = []
    
    # Saves Best Model Parameters
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    callbacks.append(checkpointer)


    # Returns LSTM Model & Best Model Parameters
    return model, {'callbacks': callbacks}


################################################################################


def train_and_predict(train_data: pd.DataFrame,
                      test_data:  pd.DataFrame) -> pd.DataFrame:


    #####  HYPER-PARAMETER TUNING  #####
    
    # Model Fitting Parameters
    epochs     = 1
    batch_size = 4
    threshold  = 0.331
    
    ####################################
    
    
    # Split Features/Labels  &  Compute Sequence Length + Size of Vocabulary
    train_X, train_Y, test_X, test_Y, train_maxlen, train_vocab_size = split_xy(train_data, test_data)
    

    # Bidirectional LSTM Model
    model, params = lstm_model(vocab_size=train_vocab_size,
                               input_length=train_maxlen)
    
    # Fit Model
    history = model.fit(train_X, train_Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        #validation_data=(test_X, test_Y),
                        shuffle=True,
                        **params)
    
    # Make Predictions for the Dev Set
    test_predictions = (1-threshold + model.predict(test_X))
    
    # Saves a Copy of the Original Probabilities
    test_prob           = test_data.copy()
    test_prob[emotions] = test_predictions

    # Classifies Each Run on the Dev Set
    test_predictions           = test_data.copy()
    test_predictions[emotions] = test_prob[emotions].astype(int)
    
    
    return test_prob, test_predictions


################################################################################

def print_metrics(df1, df2):

    sim      = jaccard_similarity_score(df1, df2)
    f1_micro = f1_score(df1, df2, average='micro')
    f1_macro = f1_score(df1, df2, average='macro')
        
    print("\taccuracy: {:.4f}".format(sim),      "\t",
            "f1-micro: {:.4f}".format(f1_micro), "\t",
            "f1-macro: {:.4f}".format(f1_macro))

    return sim


################################################################################


if __name__ == "__main__":

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv("2018-E-c-En-train.txt",         **read_csv_kwargs)
    test_data  = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)

    # Number of Times to Run the Prediction Algorithm
    num_predictions = 5
    num_models      = 0
    
    for num_models in range(num_predictions):
        
        # Makes Predictions on Each Run of the Dev Set
        print("\n\nModel {0}:".format(num_models))
        
        tprob, tpred = train_and_predict(train_data, test_data)
        tsim = print_metrics(test_data[emotions], tpred[emotions])
        
        if num_models==0:
            #dev_predictions  = dprob.copy()
            test_predictions = tprob.copy()
        else:
            #dev_predictions[emotions]  += dprob[emotions]
            test_predictions[emotions] += tprob[emotions]
        """
        print("Current Dev Ensemble Metrics:")
        temp1 = (dev_predictions[emotions]/(num_models+1)).astype(int)
        score = print_metrics(dev_data[emotions], temp1[emotions])
        
        print("Current Test Ensemble Metrics:")
        temp2 = (test_predictions[emotions]/(num_models+1)).astype(int)
        print_metrics(test_data[emotions], temp2[emotions])
        """

    # Final Prediction Based on Multiple Runs
    # Reduces Run-to-Run Variations & Improves Overall Prediction Accuracy
    test_predictions[emotions] /= num_predictions
    test_predictions.to_csv("tthreshold.txt", sep="\t", index=False)

    # saves predictions and prints out multi-label accuracy
    test_predictions[emotions] = (test_predictions[emotions]).astype(int)
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
