#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


df_train = pd.read_csv('fashion-mnist_train.csv')
df_test = pd.read_csv('fashion-mnist_test.csv')


# In[18]:


df_train.shape


# In[19]:


df_test.shape


# In[20]:


df_train.head()


# In[21]:


y = df_train['label']
X = df_train.drop('label',axis=1)


# In[22]:


class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
'''
0 => T-shirt/top 
1 => Trouser 
2 => Pullover 
3 => Dress 
4 => Coat 
5 => Sandal 
6 => Shirt 
7 => Sneaker 
8 => Bag 
9 => Ankle boot '''


# In[23]:


X_train = X /255
y_train = y 


# In[24]:


X_train.iloc[0]


# In[25]:


X_train.shape,y_train.shape


# In[50]:


X_train = np.array(X_train).reshape(-1, 28,28,1)
X_train.shape


# In[51]:


y_train.shape


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train,X_val,y_train,y_val =train_test_split(X_train,y_train,test_size=0.2,random_state=2020)


# In[52]:


X_train.shape,y_train.shape


# In[39]:


X_val.shape,y_val.shape


# In[40]:


from tensorflow.keras.models import Sequential
import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D


# In[49]:


cnn_model = keras.models.Sequential([
                         keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         keras.layers.Flatten(),
                         keras.layers.Dense(units=128, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax')
])


# In[57]:


cnn_model.summary()


# In[58]:


cnn_model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[111]:


cnn_model.fit(X_train, y_train, epochs=20, batch_size=512, verbose=1, validation_data=(X_val, y_val))


# In[112]:


cnn_model.predict(np.expand_dims(X_val[0],axis=0)).round(2)


# In[113]:


np.argmax(cnn_model.predict(np.expand_dims(X_val[0],axis=0)).round(2))


# In[68]:


X_val[0]


# In[114]:


y_pred = cnn_model.predict(X_val)
y_pred.round(2)


# In[110]:


X_val[2]


# In[ ]:





# In[115]:


cnn_model.evaluate(X_val,y_val)


# In[116]:


from sklearn.metrics import confusion_matrix
 
plt.figure(figsize=(16,9))
y_pred_labels = [ np.argmax(label) for label in y_pred ]
cm = confusion_matrix(y_val, y_pred_labels)
 
# show cm 
sns.heatmap(cm, annot=True, fmt='d',xticklabels=class_labels, yticklabels=class_labels)


# In[117]:


from sklearn.metrics import classification_report
cr= classification_report(y_val, y_pred_labels, target_names=class_labels)
print(cr)


# In[118]:



cnn_model.save('fashion_cnn_model.h5') # Save model


# In[119]:


# Load model
fashion_mnist_cnn_model = keras.models.load_model('fashion_cnn_model.h5')
 


# In[122]:


Y_pred_sample = fashion_mnist_cnn_model.predict(np.expand_dims(X_val[0], axis=0)).round(2)
Y_pred_sample
 


# In[123]:


np.argmax(Y_pred_sample[0])


# In[ ]:




