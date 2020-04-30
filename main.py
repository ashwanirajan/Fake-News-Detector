import keras

import numpy as np
import pandas as pd
from keras.layers import Conv1D, Add, Concatenate, Dropout

import os
from keras.layers import MaxPooling1D, Embedding


from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
MAX_SEQ_LENGTH = 700
MAXWORDS = 100000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

ctype = input("Select binary of multiclass\n")
#####Preprocessing
number_classes = 0

if ctype == 'binary':
    number_classes = 2
    
elif ctype == 'multiclass' :
    number_classes = 6


train = pd.read_csv('trainingdata.tsv', sep = "\t", header = None)
val = pd.read_csv('valdata.tsv', sep = "\t", header = None)
test = pd.read_csv('testdata.tsv', sep = "\t", header = None)

train.columns = [ 'index', 'ID', 'label' , 'statement', 'subject', 'speaker',  'speaker job title' ,
                      'state info', 'party', 'barely true counts', 'false counts',  'half true counts', 
                      'mostly true counts', 'pants on fire counts', 'context', 'justification' ]

val.columns = train.columns

test.columns = train.columns

train = train.drop(columns = 'index')
val = val.drop(columns = 'index')
test = test.drop(columns = 'index')
binary = []

for s in train['label']:
    if s == 'barely-true' or s == 'false' or s == 'pants-fire' :
        binary.append('false')
    else :
        binary.append('true')

train['binary'] = binary

binary = []

for s in val['label']:
    if s == 'barely-true' or s == 'false' or s == 'pants-fire' :
        binary.append('false')
    else :
        binary.append('true')

val['binary'] = binary

binary = []

for s in test['label']:
    if s == 'barely-true' or s == 'false' or s == 'pants-fire' :
        binary.append('false')
    else :
        binary.append('true')

test['binary'] = binary

statements_train = []
labels_train = []
bin_labels_train = []
statements_val = []
labels_val = []
bin_labels_val = []
statements_test = []
labels_test = []
bin_labels_test = []

for i in range(len(train)):
    text = train.statement[i]
    jstf = train.justification[i]
    text2 = str(text) + " " + str(jstf)
    statements_train.append(text2)
    if ctype == "binary" :
        labels_train.append(train.binary[i])
    elif ctype == "multiclass" :
        labels_train.append(train.label[i])
    
for i in range(len(val)):
    text = val.statement[i]
    jstf = val.justification[i]
    text2 = str(text) + " " + str(jstf)
    statements_val.append(text2)
    if ctype == "binary" :
        labels_val.append(val.binary[i])
    elif ctype == "multiclass" :
        labels_val.append(val.label[i])
       
for i in range(len(test)):
    text = test.statement[i]
    jstf = test.justification[i]
    text2 = str(text) + " " + str(jstf)
    statements_test.append(text2)
    if ctype == "binary" :
        labels_test.append(test.binary[i])
    elif ctype == "multiclass" :
        labels_test.append(test.label[i])
    
tokenize = Tokenizer(num_words=MAXWORDS, filters='! \'"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenize.fit_on_texts(statements_train)
seqs_train = tokenize.texts_to_sequences(statements_train)
word_ind_train = tokenize.word_index

#tokenize.fit_on_texts(statements_val)
seqs_val = tokenize.texts_to_sequences(statements_val)
word_ind_val = tokenize.word_index

#tokenize.fit_on_texts(statements_test)
seqs_test = tokenize.texts_to_sequences(statements_test)
word_ind_test = tokenize.word_index



from sklearn.preprocessing import LabelEncoder

labels_train = np.array(labels_train)

label_enc = LabelEncoder()
elabels_train = label_enc.fit_transform(labels_train)

labels_val = np.array(labels_val)
elabels_val = label_enc.fit_transform(labels_val)

labels_test = np.array(labels_test)
elabels_test = label_enc.fit_transform(labels_test)

####all data
data_train = pad_sequences(seqs_train, maxlen=MAX_SEQ_LENGTH)
labels_train = to_categorical(elabels_train,num_classes = number_classes)

data_val = pad_sequences(seqs_val, maxlen=MAX_SEQ_LENGTH)
labels_val = to_categorical(elabels_val,num_classes = number_classes)

data_test = pad_sequences(seqs_test, maxlen=MAX_SEQ_LENGTH)
labels_test = to_categorical(elabels_test,num_classes = number_classes)


from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM

from keras.layers.normalization import BatchNormalization

#Using Pre-trained word embeddings

emb_index = {}
f = open(os.path.join("glove.6B/", 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    emb_index[word] = coefs

f.close()

print('Total %s word vectors in Glove.' % len(emb_index))

emb_matrix = np.random.random((len(word_ind_train) + 1, EMBEDDING_DIM))
for word, i in word_ind_train.items():
    emb_vector = emb_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector

       
embedding_layer = Embedding(len(word_ind_train) + 1, EMBEDDING_DIM, weights=[emb_matrix], input_length=MAX_SEQ_LENGTH)


embedding_vecor_length = 32
modell = Sequential()
modell.add(embedding_layer)
modell.add(Dropout(0.2))
modell.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
modell.add(MaxPooling1D(pool_size=2))
modell.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
modell.add(MaxPooling1D(pool_size=2))
#model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
modell.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
modell.add(BatchNormalization())
#modell.add(Dense(256, activation='relu'))
modell.add(Dense(128, activation='relu'))
modell.add(Dense(64, activation='relu'))
modell.add(Dense(number_classes, activation='softmax'))
modell.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(modell.summary())
modell.fit(data_train, labels_train, epochs=2, batch_size=64)

modell.save('lstm.h5')

#val_preds = modell.predict(data_val)
#val_preds = np.round(val_preds)
#correct_predictions = float(sum(val_preds == labels_val)[0])
#print("Correct predictions:", correct_predictions)
#print("Total number of test examples:", len(labels_val))
#print("Accuracy of model1: ", correct_predictions/float(len(labels_val)))
#
## Creating the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#x_pred = modell.predict(data_val)
#x_pred = np.round(x_pred)
#x_pred = x_pred.argmax(1)
#y_test_s = labels_val.argmax(1)
#cm = confusion_matrix(y_test_s, x_pred)

test_preds = modell.predict(data_test)
test_preds = np.round(test_preds)
correct_predictions = float(sum(test_preds == labels_test)[0])
print("Correct predictions:", correct_predictions)
print("Total number of test examples:", len(labels_test))
print("Accuracy of model: ", correct_predictions/float(len(labels_test)))

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
x_pred = modell.predict(data_test)
x_pred = np.round(x_pred)
x_pred = x_pred.argmax(1)
y_test_s = labels_test.argmax(1)
cm = confusion_matrix(y_test_s, x_pred)
print(cm)
