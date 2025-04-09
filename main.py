import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
data_train_path='Skin cancer ISIC The International Skin Imaging Collaboration/Train'
img_width=180
img_height=180
data_train=tf.keras.utils.image_dataset_from_directory(data_train_path,image_size=(img_width,img_height),shuffle=True,batch_size=32,
                                                      validation_split=False)
data_cat=data_train.class_names
print(data_cat)
plt.figure(figsize=(10,10))
for image,labels in data_train.take(1):
  for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(image[i].numpy().astype('uint8'))
    plt.title(data_cat[labels[i]])
    plt.axis('off')
model=Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(128),
    layers.Dense(len(data_cat))
])
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(data_train,epochs=15,batch_size=32,verbose=1)
epochs_range=range(15)
plt.figure(figsize=(8,8))
plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['accuracy'],label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
epochs_range=range(15)
plt.figure(figsize=(8,8))
plt.subplot(1,2,2)
plt.scatter(epochs_range,history.history['accuracy'],label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
image_path='Skin cancer ISIC The International Skin Imaging Collaboration/Test/nevus/ISIC_0000008.jpg'
image=tf.keras.utils.load_img(image_path,target_size=(img_width,img_height))
img_arr=tf.keras.utils.img_to_array(image)
img_bat=tf.expand_dims(img_arr,0)
predict=model.predict(img_bat)
score=tf.nn.softmax(predict)
print('image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))
model.save('skincancer.keras')
