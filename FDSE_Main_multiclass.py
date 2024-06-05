import numpy as np
import os
import random

import scipy.io as sio

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers as kl

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import itertools
from sklearn.metrics import classification_report

from utils import*
import librosa
import librosa.display
from PIL import Image
import joblib

model_path="./checkpoint/model252.h5"
scaler_path ="./checkpoint/scaler252.pkl"
transformer_path="./checkpoint/transformer252.pkl"

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# tf.keras.utils.set_random_seed(seed_value)
# tf.random.set_seed()

# =============================================================================
# plot confusion matrix
# =============================================================================
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# =============================================================================
# source_dir
# =============================================================================
source_dir= './mat_new/'
save_dir='./figure/'
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
# =============================================================================
# load mat files
# =============================================================================
normal_features=sio.loadmat(os.path.join(source_dir,'normal_252.mat')) 
normal_features=normal_features['normal']

crackle_features=sio.loadmat(os.path.join(source_dir,'crackle_252.mat')) 
crackle_features=crackle_features['crackle']

wheeze_features=sio.loadmat(os.path.join(source_dir,'wheeze_252.mat')) 
wheeze_features=wheeze_features['wheeze']

both_features=sio.loadmat(os.path.join(source_dir,'both_252.mat')) 
both_features=both_features['both']    

X = np.concatenate((normal_features[:,:-1], crackle_features[:,:-1], wheeze_features[:,:-1],both_features[:,:-1]), axis=0)
y = np.concatenate((normal_features[:,-1],crackle_features[:,-1], wheeze_features[:,-1], both_features[:,-1]), axis=0)
print("X.shape:",X.shape)
print(min(X[0]),max(X[0]))
# =============================================================================
# normalization
# =============================================================================
min_max_scaler=MinMaxScaler()
X = min_max_scaler.fit_transform(X) 
joblib.dump(min_max_scaler, scaler_path)
print("===scaler Saved===")
# =============================================================================
# feature reduction (K-PCA)
# =============================================================================
transformer = KernelPCA(n_components=184, kernel='linear') #40% of 322 = 97
X = transformer.fit_transform(X)
joblib.dump(transformer, transformer_path)
print("===transformer Saved===")

print("X.shape:",X.shape)
tf.random.set_seed(42)
# Define function to build model with specified random seed
def build_model(feature_size, n_classes, dropout):
    """ Build a small model for multi-label classification """
    inp = kl.Input((feature_size,))
    x = kl.Dense(1024, activation='relu')(inp)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(640, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(512, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(256, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(128, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(64, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    x = kl.Dense(32, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(dropout)(x)
    out = kl.Dense(n_classes, activation='softmax')(x) # change softmax
    model = keras.Model(inputs=inp, outputs=out)
    return model

# =============================================================================
# devide data into test,train, and validation sets
# =============================================================================
y = to_categorical(y)
print('y value: ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
print('X train - y train:', X_train.shape, y_train.shape)
print('X test - y test:', X_test.shape, y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1) # 0.25
print('X train - val', X_train.shape, y_train.shape)


# =============================================================================
# build model
# =============================================================================
model = build_model(feature_size=X_train.shape[-1], n_classes=4, dropout= 0.2)
print('Built model!!!', model)
# # =============================================================================
# train model
# =============================================================================
callback = EarlyStopping(monitor='loss', patience=3)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
criterion = tf.keras.losses.categorical_crossentropy
model.compile(optimizer=opt, loss=criterion,metrics=['acc'])
trainedmodel = model.fit(X_train, y_train,batch_size = 128,epochs=100, validation_data = (X_val, y_val), callbacks=[callback])
model.save(model_path)
print("===Model Saved===")

fig = plt.figure()
plt.plot(model.history.history['val_loss'], 'r',model.history.history['loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Loss Score')
plt.grid(1)
plt.savefig(save_dir+'training_loss.jpg',dpi=300)


fig = plt.figure()
accuracy = trainedmodel.history['acc']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(1)
plt.savefig(save_dir+'Acc.jpg',dpi=300)
# =============================================================================
# evaluate model
# =============================================================================
print('x test:', X_test.shape)
Y_Score=model.predict(X_test)
print('Y_score:', Y_Score.shape)
y_pred = np.argmax(Y_Score, axis=1)
print('y test: ', y_test.shape)
cm=confusion_matrix(np.argmax(y_test, axis=1),y_pred)
print(cm)

fig = plt.figure()
plot_confusion_matrix(cm, classes=['crackle', 'wheeze', 'both', 'normal'])
plt.savefig(save_dir+'conf_matrix.jpg',dpi=300)

test_loss=model.evaluate(X_test,y_test,verbose=1)#evaluate model
print("test_loss:", test_loss)#print test loss and metrics information

# =============================================================================
# ROC curve where positive label is abnormal
# =============================================================================
classes_ = ['','crackle','wheeze','both']
for pos_label in range(1,4):
    fpr, tpr, thresholds = roc_curve(y_test[:, pos_label], Y_Score[:, pos_label], pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)\npos_label=%s' % (roc_auc, pos_label))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(save_dir+f'roc_class_{classes_[pos_label]}.jpg', dpi=300)


# =============================================================================
# ROC curve where positive label is normal
# =============================================================================
pos_label=0
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), Y_Score[:,pos_label], pos_label=pos_label)
roc_auc = auc(fpr, tpr)# calculate auc value
fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)\npos_label=Abnormal' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.grid(1)
plt.legend(loc="lower right")
plt.savefig(save_dir+'roc_abnormal.jpg',dpi=300)

avg_cm = confusion_matrix(np.argmax(y_test, axis=1),y_pred)
print('confusion matrix',avg_cm)

# fixed metrics(기존은 True/Predicted -> 논문은 Predicted/Total) 
# cm[label][predicted]
se = (avg_cm[1][1] + avg_cm[2][2] + avg_cm[3][3])/(avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3] 
                                                   + avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3]
                                                  + avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])

sp = avg_cm[0][0]/(avg_cm[0][0] + avg_cm[0][1] + avg_cm[0][2] + avg_cm[0][3])
sc = (se+sp)/2

s_crackle = avg_cm[1][1]/(avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3])
s_wheezle = avg_cm[2][2]/(avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3])
s_both = avg_cm[3][3]/(avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])

print('Specificity Sp:', sp)
print('Sensitivity Se:', se)
print('Score Sc:', sc)
print('crackle accuracy:', s_crackle)
print('wheezle accuracy:', s_wheezle)
print('both accuracy:', s_both)

# Classification report
#print(classification_report(y_test, y_pred))