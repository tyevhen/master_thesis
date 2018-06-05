import numpy as np
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import StratifiedKFold
from data_helper import prepare_data, load_dump
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def shallow_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

healthy_data = load_dump("hdump.obj")
ill_data = load_dump('idump.obj')
Xtrain, Xtest, Ytrain, Ytest = prepare_data(ill_data, healthy_data, 0.7)


# tfidf_vec = TfidfVectorizer(max_df=0.95, min_df=10, dtype=np.int8)
# tfidf = tfidf_vec.fit(Xtrain+Xtest)

# Xtrain_tf = tfidf.transform(Xtrain).toarray()
# Xtest_tf = tfidf.transform(Xtest).toarray()
# print('x_train_tf shape:', Xtrain_tf.shape)
# print('x_test_tf shape:', Xtest_tf.shape)

num_classes = 2

# print('Vectorizing sequence data...')
# cv = CountVectorizer(dtype=np.int8)
# Xtrain_encoded = cv.fit_transform(Xtrain).toarray()
# Xtest_encoded = cv.transform(Xtest).toarray()

# print('x_train_cv shape:', Xtrain_encoded.shape)
# print('x_test_cv shape:', Xtest_encoded.shape)

vec_features = Pipeline([
    ('tfidf', CountVectorizer()),
])

vec_features.fit(Xtrain+Xtest)
print(Xtrain[0])
Xtrain = vec_features.transform(Xtrain)
print(Xtrain[0])
Xtest = vec_features.transform(Xtest)
print('x_train shape:', Xtrain.shape)
print('x_test shape:', Xtest.shape)


print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(Ytrain, num_classes)
y_test = keras.utils.to_categorical(Ytest, num_classes)

print('Building model...')
epochs = 5
batch_size = 128
input_shape = Xtrain.shape[1]
model = shallow_model(input_shape)
tensorboard_viz = TensorBoard(log_dir='./logdir',
                              batch_size=128,
                              write_graph=True,
                              embeddings_layer_names=None)

callbacks = [tensorboard_viz]

history = model.fit(Xtrain, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=callbacks,
                    shuffle=True)

score = model.evaluate(Xtest, y_test,
                       batch_size=batch_size,
                       verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
