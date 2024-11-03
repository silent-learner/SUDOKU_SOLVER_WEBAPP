import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import load_model

X = np.load('/content/images.npy')/255.0
y = np.load('/content/labels.npy')

X = np.reshape(X,(-1,128,128,1))

X_train, X_eval, y_train, y_eval = train_test_split(X,y,test_size=0.05,random_state=2024)
print("Training Set Shape = ",X_train.shape)
print("Validation Set Shape = ",X_eval.shape)

y_train = to_categorical(y_train,10)
y_eval = to_categorical(y_eval, 10)

model = Sequential()

model.add((Conv2D(60,(5,5),input_shape=(128,128,1) ,padding = 'same' ,activation='relu')))
model.add((Conv2D(60, (5,5),padding="same",activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

model.add((Conv2D(30, (3,3),padding="same", activation='relu')))
model.add((Conv2D(30, (3,3), padding="same", activation='relu')))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5.keras',
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32,callbacks=[checkpoint])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy vs Epoch.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss vs Epoch.png')
plt.show()

best_model = load_model('best_model.h5.keras')

loss, acc = best_model.evaluate(X_eval, y_eval, batch_size=32)
# loss, acc = model.evaluate(scaler.transform(X_test), y_test, batch_size=batch_sizee)
print(f"\nTest accuracy: {100*acc} %")
print(f"\nTest Loss: {loss}")

test_image = cv2.imread('/content/Image 24.jpg',cv2.IMREAD_GRAYSCALE)
plt.imshow(test_image)
test_image = np.reshape(test_image,(1,128,128,1))
test_image = test_image / 255.0
print(test_image.shape)
y_pred = best_model.predict(test_image)
y_hat = np.argmax(y_pred,axis=1)
print(np.max(y_pred,axis=1))
print(y_hat)

