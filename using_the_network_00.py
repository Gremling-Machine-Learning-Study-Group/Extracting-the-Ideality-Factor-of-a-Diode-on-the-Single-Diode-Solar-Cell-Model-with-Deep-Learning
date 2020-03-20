import numpy as np
import tensorflow as tf

def prepare(data):
    train_data = np.load('train_samples00.npy')
    mean = train_data.mean()
    train_data = train_data - mean
    std = train_data.std()    
    data -= mean
    data /= std
    data = data.reshape(1, 66 * 2)
    return data

#importar dados:
dados_importar = np.genfromtxt("set_dados_01.txt", delimiter=" ", names="V, I")
v  = dados_importar['V']
j  = dados_importar['I']

#inserir dados em um tensor do fomato esperado pela network:
dados_utilizar = np.zeros((1, 66, 2))

for i in range(66):
    dados_utilizar[0, i, 0] = v[i]
    dados_utilizar[0, i, 1] = j[i]

#print(dados_utilizar.shape)

#obter media e std dos dados de treino para usar na função prepare
train_data = np.load('train_samples00.npy')
mean = train_data.mean()
train_data = train_data - mean
std = train_data.std()

'''
#carregar networks, fazer previsões, e printar:
print("network 00:")
model = tf.keras.models.load_model('trained_network_00')
prediction = model.predict(prepare(dados_utilizar))
print(prediction)
prediction *= std
prediction += mean
print(prediction)

print("network 01:")
model = tf.keras.models.load_model('trained_network_01')
prediction = model.predict(prepare(dados_utilizar))
print(prediction)
prediction *= std
prediction += mean
print(prediction)

print("network 02:")
model = tf.keras.models.load_model('trained_network_02')
prediction = model.predict(prepare(dados_utilizar))
print(prediction)
prediction *= std
prediction += mean
print(prediction)
'''
print("network 03:")
model = tf.keras.models.load_model('trained_network_03')
prediction = model.predict(prepare(dados_utilizar))
print(prediction)
prediction *= std
prediction += mean
print(prediction)

print("network 04:")
model = tf.keras.models.load_model('trained_network_04')
prediction = model.predict(prepare(dados_utilizar))
print(prediction)
prediction *= std
prediction += mean
print(prediction)



'''
#tratar os dados:
def prepare(train_data):
    mean = train_data.mean()
    train_data = train_data - mean
    std = train_data.std()
    train_data = train_data/std
    return train_data

#carregar network:
model = tf.keras.models.load_model('trained_network')

#fazer previsão com novos dados:
prediction = model.predict(prepare(novos_dados))

print(prediction)
'''