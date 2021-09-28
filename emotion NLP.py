import pandas as pd
dt_train=pd.read_csv("D:\projek\machine learning\emotion NLP\emotion.csv")
dt_test=pd.read_csv("D:\projek\machine learning\emotion NLP\emotion_test.csv")

#train 
category_train = pd.get_dummies(dt_train.emotion)
ds_train = pd.concat([dt_train,category_train],axis=1)
ds_train = ds_train.drop(columns='emotion')
ds_train.info()
print(ds_train.head())

desc_train = ds_train['desc'].values
tag_train = ds_train[['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']].values
print(desc_train)


#test
category_test = pd.get_dummies(dt_test.emotion)
ds_test = pd.concat([dt_test,category_test],axis=1)
ds_test = ds_test.drop(columns='emotion')
ds_test.info()
print(ds_test.head())

desc_test = ds_test['desc'].values
tag_test = ds_test[['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']].values
print(desc_test)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1700, oov_token= 'oov')
tokenizer.fit_on_texts(desc_train)
tokenizer.fit_on_texts(desc_test)

seq_train= tokenizer.texts_to_sequences(desc_train)
seq_test= tokenizer.texts_to_sequences(desc_test)

pad_train = pad_sequences(seq_train)
pad_test = pad_sequences(seq_test)

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,GlobalMaxPooling1D
from tensorflow.keras.optimizers import SGD

model = Sequential([
                    Embedding(input_dim=1700, output_dim=16),
                    LSTM(64),
                    
                    Dropout(0.75),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(32, activation='relu'),
                    Dropout(0.25),
                    Dense(6, activation='softmax'),
])


model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=(['accuracy'])
)

print(model.summary())

from tensorflow.keras.callbacks import Callback

class mycallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>=0.9 and logs.get('val_accuracy')>=0.9):
      print("\nAkurasi dan val_akurasi telah mencapai >90%")
      self.model.stop_training = True
callback = mycallback()

history = model.fit(pad_train, tag_train, 
                    steps_per_epoch=100,
                    epochs=40, 
                    validation_data=(pad_test, tag_test),
                    callbacks=[callback])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuraacy')
plt.legend()
plt.show()







