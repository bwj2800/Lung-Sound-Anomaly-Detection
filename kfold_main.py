import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
import random
sys.path.append(os.path.abspath('./'))
from model.cnn_lstm import CNN_LSTM

accuracies = []
precisions = []
recalls = []
confusion_matrices = []
# One-hot encode the target variable y (but don't use it for splitting)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1))

# Use the original, unencoded y for splitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]  # Use original y for splitting

  # One-hot encode y for individual folds after splitting
    y_train = onehot_encoder.transform(y_train.reshape(-1, 1))
    y_val = onehot_encoder.transform(y_val.reshape(-1, 1))

    # Build and compile your model
    model = build_model(feature_size=X_train.shape[-1], n_classes=4, dropout=0.2)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    criterion = tf.keras.losses.categorical_crossentropy
    model.compile(optimizer=opt, loss=criterion, metrics=['accuracy', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

    # Train the model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    trained_model = model.fit(X_train, y_train, batch_size=128, epochs=100, 
                               validation_data=(X_val, y_val), callbacks=[callback])

    # Evaluate the model on the test set
    test_metrics = model.evaluate(X_val, y_val)
    accuracies.append(test_metrics[1])
    precisions.append(test_metrics[2])
    recalls.append(test_metrics[3])

    # Compute the confusion matrix
    y_val = np.argmax(y_val, axis = 1)
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    # clas_report = classification_report(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    confusion_matrices.append(cm)

    # Save the model
    # model.save(f'./model_seed/4gr_Res50_TCN/5f/model_{i}.h5')
    # Reset the model for the next fold
    tf.keras.backend.clear_session()

# Calculate the average metrics across all folds
avg_accuracy = sum(accuracies) / len(accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)

print('Average Accuracy:', avg_accuracy)
print('Average Precision:', avg_precision)
print('Average Recall:', avg_recall)

# Calculate the average confusion matrix across all folds
avg_cm = sum(confusion_matrices) / len(confusion_matrices)

se = (avg_cm[1][1] + avg_cm[2][2] + avg_cm[3][3])/(avg_cm[1][0] + avg_cm[1][1] + avg_cm[1][2] + avg_cm[1][3] 
                                                   + avg_cm[2][0] + avg_cm[2][1] + avg_cm[2][2] + avg_cm[2][3]
                                                  + avg_cm[3][0] + avg_cm[3][1] + avg_cm[3][2] + avg_cm[3][3])
sp = avg_cm[0][0]/(avg_cm[0][0] + avg_cm[0][1] + avg_cm[0][2] + avg_cm[0][3])
sc = (se+sp)/2
print('Specificity Sp:', sp)
print('Sensitivity Se:', se)
print('Score Sc:', sc)