import numpy as np
from random import uniform
from random import random
from random import randint
import matplotlib.pyplot as plt
from scipy.special import lambertw

#curva antiga
'''
#(I_sc-I_0*(np.exp(q*(V+j*r_s)/(n*k_b*t))-1.)-(V+j*r_s)/R_sh)
def curva(I_0, R_sh, n, I_sc, r_s, V):
    j_tentativa=-5.00 #valor inivial chutado
    j_estimado=42 #gambiarra
    #while para chutar valores de j
    while(round(j_estimado,3)!=round(j_tentativa,3)):
        j_tentativa = round(j_tentativa,3)
        #print(j_tentativa)
        j_estimado = (I_sc-I_0*(np.exp(q*(V+j_tentativa*r_s)/(n*k_b*t))-1.)-(V+j_tentativa*r_s)/R_sh)
        if(round(j_estimado,3)<=round(j_tentativa,3)):
            #print("ACHEI!!!")
            #print(j_estimado)
            #print(j_tentativa)
            break
        else:
            j_tentativa+=0.001
    return j_estimado
'''

def curva(i0, rsh, n, iph, rs, v):
    vth = k_b*t/q
    a = n*vth
    gsh = 1./rsh
    argW = rs*i0 / (a * (rs * gsh + 1.)) * np.exp((rs * (iph + i0) + v) /(a * (rs * gsh + 1.)))
    lambertwterm = lambertw(argW).real
    return  (iph + i0 - v * gsh) /(rs * gsh + 1.) - (a / rs) * lambertwterm

def small_rdm(a, a_exp, b, b_exp):
    exp = randint(a_exp, b_exp)
    base = uniform(a,b)
    return base * 10**exp

#definir tensores preenchidos com zeros:
train_samples = np.zeros((1000,66)) #inputs -> index x V x I
train_labels_1 = np.zeros((1000,1)) #outputs index x parametros esperados
train_labels_2 = np.zeros((1000,4))

test_samples = np.zeros((1000,66))
test_labels_1 = np.zeros((1000,1))
test_labels_2 = np.zeros((1000,4))

#definir parametros invariantes:
q=1.60217662e-19   #???
k_b=1.38064852e-23 #m2kgs-2K-1???
t=298.6            #K???

#criar dados artificiais:
numero_de_dados = 1000

#gerar train labels
for i in range(numero_de_dados): 
    
    train_labels_1[i,0]=small_rdm(1.0,-14,10,-2) #I_0
    train_labels_2[i,0]=uniform(1.0,1000)     #R_sh
    train_labels_2[i,1]=uniform(1.0,2.0)       #n             
    train_labels_2[i,2]=uniform(0.1,0.4)       #I_sc
    train_labels_2[i,3]=uniform(0.001,0.1)     #r_s
    '''
    #limites artigo
    train_labels_2[i,0]=small_rdm(1.0,-9,10,-7) #I_0
    train_labels_1[i,0]=uniform(100,350)       #R_sh
    train_labels_2[i,1]=uniform(1.0,2.0)       #n             
    train_labels_2[i,2]=uniform(0.27,0.30)     #I_sc
    train_labels_2[i,3]=uniform(0.01,0.06)     #r_s
    '''
    
    print(int((100*i)/numero_de_dados),"% - ",i+1)
    
    '''
    prange = (	[1.e-14,1.e-1],     #I0  ???
			[1.0,1000.],              #Rsh
			[1.0, 1.5],               #n  ???
			[0.1,0.4],                #Isc
			[0.001,0.1])              #Rs 
    '''
b=0
#gerar samples
for i in range(numero_de_dados):
    
    for j in range(66):
        
        #primeira coluna:
        #train_samples[i, j, 0] += 0.01*j 
        V = 0.01*j
        
        #segunda coluna:
        train_samples[i, j] = curva(
                train_labels_1[i,0],         #I_0
                train_labels_2[i,0],         #R_sh
                train_labels_2[i,1],         #n
                train_labels_2[i,2],         #I_sc
                train_labels_2[i,3],         #r_s
                V)                           #V
    
    #print(samples[i,j])
    a=int((100*i)/numero_de_dados)
    if a!=b:
        print(b,"%")
        plt.plot(train_samples[i], 'r')
        plt.show()
    b=a
    
print("Train Finalizado!")

np.save("train_samples_I0", train_samples, allow_pickle=True, fix_imports=True)
np.save("train_labels_I0", train_labels_1, allow_pickle=True, fix_imports=True)

'''
print(train_samples)
print(train_labels_1)
print(train_labels_2)
'''

'''    
train_sample = np.array(train_sample)
train_label = np.array(train_label)
'''

'''
#gerar test_labels
for i in range(150):
    
    test_labels[i,0]=uniform(1.0e-14,1.0e-1) #I_0
    test_labels[i,1]=uniform(1.0, 1000.)    #R_sh
    test_labels[i,2]=uniform(1.0, 2.0)      #n
    test_labels[i,3]=uniform(0.1, 0.4)      #I_sc
    test_labels[i,4]=uniform(0.001, 0.1)    #r_s
    
#gerar test_samples
for i in range(150):
    
    for j in range(66):
        
        #primeira coluna:
        test_samples[i, j, 0] += 0.01*j 
        V = 0.01*j
        
        #segunda coluna:
        test_samples[i, j, 1] = curva(test_labels[i,0], #I_0
               test_labels[i,1],                   #R_sh
               test_labels[i,2],                   #n
               test_labels[i,3],                   #I_sc
               test_labels[i,4],                   #r_s
               V)                             #V
    
    #print(samples[i,j])
    print(int((100*i)/150.),"%")
    
print("Train Finalizado!")

np.save("test_samples01", test_samples, allow_pickle=True, fix_imports=True)
np.save("test_labels01", test_labels, allow_pickle=True, fix_imports=True)

print(test_samples)
print(test_labels)
'''

'''
train_data = np.load('train_samples00.npy')
train_targets = np.load('train_labels00.npy')
test_data = np.load('test_samples00.npy')
test_targets = np.load('test_labels00.npy')

#INPUT
mean = train_data.mean()
train_data -= mean
std = train_data.std()
train_data /= std
test_data -= mean
test_data /= std
#OUTPUT
print(train_targets)
train_targets -= mean
train_targets /= std
print(train_targets)
train_targets *= std
train_targets += mean
print(train_targets)
'''

'''
#importar dados:
dados_importar = np.genfromtxt("set_dados_01.txt", delimiter=" ", names="V, I")
v  = dados_importar['V']
j  = dados_importar['I']

#colocar em um tensor do fomato o esperado:
dados_utilizar = np.zeros((1, 66, 2))

for i in range(66):
    dados_utilizar[0, i, 0] = v[i]
    dados_utilizar[0, i, 1] = j[i]
    
print(dados_utilizar.shape)

dados_utilizar = dados_utilizar.reshape(1, 66 * 2)
'''