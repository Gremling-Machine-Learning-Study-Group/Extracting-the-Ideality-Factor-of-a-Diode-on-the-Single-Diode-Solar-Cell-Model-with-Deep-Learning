import numpy as np
from keras import models
#from keras.models import Sequential
from keras import layers
from keras import callbacks
#from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers
import tensorflow as tf

#Estructure based "boston_house_prices" example by François Chollet on "Deep Learning with Python"
#Define the estructure of the neural network (dont need to be so deep in this case):
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(26, activation='sigmoid', input_dim=26))
    #model.add(layers.Dense(26, activation='sigmoid'))
    #model.add(layers.Dense(264, activation='sigmoid'))
    #model.add(layers.Dense(132, activation='sigmoid'))
    #model.add(layers.Dense(66, activation='sigmoid'))
    #model.add(layers.Dense(660, activation='relu', input_shape=(66*2,)))
    model.add(layers.Dense(1))
    
    sgd = optimizers.SGD(lr=0.005);
    model.compile(#optimizer='rmsprop',
                  loss='mse', 
                  optimizer=sgd,
                  metrics=['mae'])
    
    return model

#Function that will generate a curve to be ploted:
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

#Print the current "epoch" during training:
def epoca_feedback(epoch, logs):
    print(epoch)

#Define the callbacks list (a keras functionality):
callbacks_list = [callbacks.LambdaCallback(on_epoch_end=epoca_feedback)]

#Import the generated data:
train_data = np.load('train_helal_samples_n.npy')
train_targets = np.load('train_helal_labels_n.npy')
test_data = np.load('test_helal_samples_n.npy')
test_targets = np.load('test_helal_labels_n.npy')

#Normalize the data:
#INPUT
mean = train_data.mean()
train_data -= mean
std = train_data.std()
train_data /= std
test_data -= mean
test_data /= std

#Number of epochs:
num_epochs = 1000

#Train the deep neural net:
model = build_model()
history = model.fit(train_data[:800],
                    train_targets[:800],
                    validation_data = (train_data[800:], train_targets[800:]),
                    epochs = num_epochs,
                    batch_size = 1,
                    verbose = 0,
                    callbacks=callbacks_list
                    )
    
#Save the logs of the training to be ploted:
all_mae_histories = []
mae_history = history.history['val_mean_absolute_error']
all_mae_histories.append(mae_history)
average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

smooth_mae_history = smooth_curve(average_mae_history)

#Print the curve
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#Test with new data (test data that we generated)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("Test MSE:", test_mse_score)
print("Test MAE", test_mae_score)

#Test the neural net with the data that I want to extract the n parameter:
dados_importar = np.genfromtxt("curva_0.txt", delimiter="", names="V, I")
v  = dados_importar['V']
j  = dados_importar['I']

dados_utilizar = np.zeros((1, 26))

for i in range(26):
    dados_utilizar[0, i] = j[i]

dados_utilizar -= mean
dados_utilizar /= std

#Export the trained deep neural net so we can use it after
model.save('trained_network_n_helal')

model = tf.keras.models.load_model('trained_network_n_helal')
prediction = model.predict(dados_utilizar)
print(prediction)
