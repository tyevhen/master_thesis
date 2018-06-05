from data_helper import *
import numpy as np
from nltk.tokenize import word_tokenize
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, MaxPooling1D, Embedding, concatenate, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def create_tokenizer(docs):
    tok = Tokenizer()
    tok.fit_on_texts(docs)
    return tok

def max_doc_len(docs):
    counts = [len(word_tokenize(sent)) for sent in all_data]
    return np.max(counts)

def encode_text(tokenizer, docs, max_doc_len):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=max_doc_len, padding='post')
    return padded

def define_model(input_length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(input_length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=128, kernel_size=2, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=30)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(input_length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=30)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(input_length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=30)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    model.summary()
    return model

# Load data and labels
Xtrain = load_dump("train_pt_new")
Xdev = load_dump("dev_pt_new")

Ytrain = np.array(load_dump("Ytrain_new"))
Ydev = np.array(load_dump("Ydev_new"))


# print(Xtrain.shape)
# print(Xdev.shape)

all_data = Xtrain+Xdev

# Create tokenizer
tok = create_tokenizer(all_data)

# Calculate vocabulary size and maximum document length
vocab_size = len(tok.word_index) + 1
max_length = max_doc_len(all_data)

# Encode and pad data
trainX = encode_text(tok, Xtrain, max_length)
devX = encode_text(tok, Xdev, max_length)

# Define model
model = define_model(max_length, vocab_size)
tensorboard_viz = TensorBoard(log_dir='./logdir',
                              batch_size=32,
                              write_graph=True,
                              embeddings_layer_names=None)

callbacks = [tensorboard_viz]
model.fit([trainX, trainX, trainX], Ytrain, epochs=2, batch_size=128, shuffle=True, callbacks=callbacks)
model.save('model.h5')

# load the model
model = load_model('model.h5')

# evaluate model on training dataset
loss, acc = model.evaluate([trainX, trainX, trainX], Ytrain, verbose=0)
print('Train Accuracy: %f' % (acc * 100))

# evaluate model on test dataset dataset
# loss, acc = model.evaluate([testX, testX, testX], Ytest, verbose=0)
#
# print('Test Accuracy: %f' % (acc * 100))

y_probs = model.predict([devX, devX, devX])
y_preds = y_probs.argmax(axis=-1)
print(y_preds)

