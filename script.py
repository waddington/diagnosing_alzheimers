from time import time

import numpy as np
import glob
import csv
from PIL import Image
from keras import optimizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Model

import models

# Some configurations
logging_dir = 'logs'
logging_name = 'test_1'
models_dir = 'models'
model = models.shortened_model()  # Look in models.py for different models | Can also load a model from a file
batch_size = 1  # Because of the way the data generator and augmentation works, the actual batch size is this value * 7 (there are 6 transformed images added)

# Read data
# Images are 256x256
train_X = []
train_Y = []
with open("./mri-png/trainingImages.csv") as csv_file:
	reader = csv.reader(csv_file)
	for row in reader:
		img_id = row[0]
		img_dx = row[1]
		if img_dx == "NL":
			dx = [1,0,0]
		elif img_dx == "MCI":
			dx = [0,1,0]
		else:
			dx = [0,0,1]
		img = Image.open("./mri-png/"+img_id+".png").convert(mode='L')
		train_X.append([np.asarray(img)])
		train_Y.append(dx)
train_X = np.array(train_X)
train_Y = np.array(train_Y)

validate_X = []
validate_Y = []
with open("./mri-png/testImages.csv") as csv_file:
	reader = csv.reader(csv_file)
	for row in reader:
		img_id = row[0]
		img_dx = row[1]
		if img_dx == "NL":
			dx = [1,0,0]
		elif img_dx == "MCI":
			dx = [0,1,0]
		else:
			dx = [0,0,1]
		img = Image.open("./mri-png/"+img_id+".png").convert(mode='L')
		validate_X.append([np.asarray(img)])
		validate_Y.append(dx)
validate_X = np.array(validate_X)
validate_Y = np.array(validate_Y)

# Compile the model
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, clipnorm=1.)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

# model.summary()

# Set up tensorboard
tensorboard = TensorBoard(log_dir=logging_dir+"/"+logging_name,
						  histogram_freq=0,
						  write_graph=True,
						  write_images=True)

# Set up early stopping callback
earlystop = EarlyStopping(monitor='val_loss',
						  min_delta=0.0001,
						  patience=5,
						  verbose=1,
						  mode='auto')

# Train the model
model.fit(x=train_X,
		  y=train_Y,
		  validation_data=(validate_X, validate_Y),
		  epochs=100,
		  verbose=1,
		  callbacks=[tensorboard])

# Save the model
# model.save(models_dir+"/"+logging_name+".h5")