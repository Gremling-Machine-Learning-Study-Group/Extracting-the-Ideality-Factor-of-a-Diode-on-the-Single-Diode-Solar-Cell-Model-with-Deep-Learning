import numpy as np
from keras import models
#from keras.models import Sequential
from keras import layers
#from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers
import tensorflow as tf

#from random import randint
#from sklearn.preprocessing import MinMaxScaler
#from kera.optimizers import Adam

'''
I see some basic bugs in this methodology. Your final layer of the network has
 a softmax layer. This would mean it would output 5 values, which sum to 1, 
 and behave as a probability distribution. What you actually want to predict
 is true numbers, or rather floating point values (under some fixed precision
 arithmetic).

If you have a range, then probably using a sigmoid and rescaling the final 
layer would to match the range (just multiply with the max value) would help
 you. By default sigmoid would ensure you get 5 numbers between 0 and 1.

The other thing should be to remove the cross entropy loss and use a loss like
 RMS, so that you predict your numbers well. You could also used 1D
 convolutions instead of using Fully connected layers.
 
Objetivos:
-normalizar dados com uma média nova entre 0~1
!!!normalizar dados com a média dos inputs

outra opção:
    excluir dimensão tempo da eqo
    1D convolutions
'''

#estrutura da rede baseada no exemplo de "boston_house_prices"
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(32, activation='sigmoid', input_dim=66))
    model.add(layers.Dense(32, activation='sigmoid'))
    #model.add(layers.Dense(264, activation='sigmoid'))
    #model.add(layers.Dense(132, activation='sigmoid'))
    #model.add(layers.Dense(66, activation='sigmoid'))
    #model.add(layers.Dense(660, activation='relu', input_shape=(66*2,)))
    model.add(layers.Dense(1))
    
    sgd = optimizers.SGD(lr=0.0005);
    model.compile(#optimizer='rmsprop',
                  loss='mse', 
                  optimizer=sgd,
                  metrics=['mae'])
    
    return model

#função para gerar uma curva, para ser, posteriormente, plotada
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

#importar dados criados:
train_data = np.load('train_samples_I0.npy')
train_targets = np.load('train_labels_I0.npy')
test_data = np.load('test_samples_I0.npy')
test_targets = np.load('test_labels_I0.npy')

#print(train_data.shape)
#print(train_targets.shape)
#(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

#preparar dados:
#INPUT
mean = train_data.mean()
train_data -= mean
std = train_data.std()
train_data /= std
test_data -= mean
test_data /= std

'''
#OUTPUT
train_targets -= mean
train_targets /= std
test_targets -= mean
test_targets /= std
'''

#k-fold validation:
k = 4
num_val_samples = len(train_data)//k
num_epochs = 1000

all_mae_histories = []

for i in range(k):
    
    print('processing fold #', i+1)
    
    #preparar validation_data da partição # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    #preparar os dados para o treinamento: de todas as outras partições
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    
    #dar build no keras model
    model = build_model()
    
    #alterar o shape dos inputs temporarios da rede para "A x B*C"
    #partial_train_data = partial_train_data.reshape(100, 66 * 2)
    #val_data = val_45data.reshape(50, 66 * 2)
    
    #treinar o modelo    (in silent mode ==> verbose = 0)
    
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        validation_data = (val_data, val_targets),
                        epochs = num_epochs,
                        batch_size = 1,
                        verbose = 0
                        )
    
    #salvar dados da efetividade do modelo
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
    #calcular média para analizar:            pq a média???
    average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    #gerar uma curva com os dados:
    smooth_mae_history = smooth_curve(average_mae_history)

    #printar a curva
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
    
#Testar com novos dados(gerados)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)

#teste final com dados reais:
dados_importar = np.genfromtxt("set_dados_01.txt", delimiter=" ", names="V, I")
v  = dados_importar['V']
j  = dados_importar['I']

dados_utilizar = np.zeros((1, 66))

for i in range(66):
    dados_utilizar[0, i] = j[i]

dados_utilizar -= mean
dados_utilizar /= std

#salvar/exportar modelo treinado:
model.save('trained_network_12_n')

model = tf.keras.models.load_model('trained_network_12_n')
prediction = model.predict(dados_utilizar)
print(prediction)

'''
#Usando 'k-fold validation':
k = 4
num_val_samples = len(train_data) // k
num_epochs = 45
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
'''



'''
#testar o modelo final:
train_data = train_data.reshape(150, 66 * 2)

model = build_model()
model.fit(train_data,
         train_targets,
         epochs=415,
         batch_size=16,
         verbose=0)

test_data = test_data.reshape(150, 66 * 2)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)
'''

'''
for i in range(1000):
    younger_ages = randint(3,4)
    train_sample.append(younger_ages)
    train_label.append(0)
    
    older_ages = randint(1,2)
    train_sample.append(older_ages)
    train_label.append(1)
    
train_sample = np.array(train_sample)
train_label = np.array(train_label)

scalar = MinMaxScaler(feature_range=(0,1))
scalar_train_sample = scalar.fit_transform(train_sample.reshape(-1,1))

model = Sequential([Dense(16, input_dim=1, activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(32, activation='softmax')])

model.summary()

model.compile(Adam(lr=0.001),
              loss='sparce_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_sample,
          train_label,
          batch_size=10,
          epochs=10)

test_sample = [] #tensor com 80 sets da dados simulados
test_label = [] #tensor com 20 sets de dados simulados

for i in range(500):
    younger_ages = randint(3,4)
    test_sample.append(younger_ages)
    test_label.append(0)
    
    older_ages = randint(1,2)
    test_sample.append(older_ages)
    test_label.append(1)
    
test_sample = np.array(test_sample)
test_label = np.array(test_label)

test_sample_output = 
model.predict_classes(test_sample,
                      batch_size=10)

from sklearn.metrics import confusion_matrix
predictedvalues=confusion_matrix()

'''


