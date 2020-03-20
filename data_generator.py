import numpy as np
from random import uniform
from random import randint
import matplotlib.pyplot as plt
from scipy.special import lambertw

#By Brute Force
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
#By lambert W:
def curva(i0, gsh, n, iph, rs, v):
    vth = k_b*t/q
    a = n*vth
    #gsh = 1./rsh
    argW = rs*i0 / (a * (rs * gsh + 1.)) * np.exp((rs * (iph + i0) + v) /(a * (rs * gsh + 1.)))
    lambertwterm = lambertw(argW).real
    return  (iph + i0 - v * gsh) /(rs * gsh + 1.) - (a / rs) * lambertwterm

def small_rdm(a, a_exp, b, b_exp):
    exp = randint(a_exp, b_exp)
    base = uniform(a,b)
    return base * 10**exp

numero_de_dados = 1000
numero_de_pontos = 26

#Create tensors with zeros
train_samples = np.zeros((numero_de_dados, numero_de_pontos)) #inputs -> index x V x I
train_labels_1 = np.zeros((numero_de_dados,1)) #outputs index x parametros esperados
train_labels_2 = np.zeros((numero_de_dados,4))

test_samples = np.zeros((numero_de_dados, numero_de_pontos))
test_labels_1 = np.zeros((numero_de_dados,1))
test_labels_2 = np.zeros((numero_de_dados,4))

#Fixed parameters of the model
q=1.60217646e-19   #Electron charge
k_b=1.3806503e-23  #Boltzmann constant?
t=306.15      #Temperature (obtained with the data)



#Create artificial data to train the network:

#Generate the train LABELS:
for i in range(numero_de_dados): 
    
    '''
    #limites teoricos:
    train_labels_2[i,0]=small_rdm(1.0,-14,10,-2) #I_0
    train_labels_2[i,1]=uniform(1.0,1000)        #R_sh
    train_labels_2[i,2]=uniform(1.0,2.0)         #n             
    train_labels_2[i,3]=uniform(0.1,0.4)         #I_sc
    train_labels_1[i,0]=uniform(0.001,0.1)       #r_s
    '''
    
    '''
    #limites artigo_adhimar:
    train_labels_2[i,0]=small_rdm(1.0,-9,10,-7) #I_0
    train_labels_1[i,0]=uniform(100,350)        #R_sh
    train_labels_2[i,1]=uniform(1.0,2.0)        #n             
    train_labels_2[i,2]=uniform(0.27,0.30)      #I_sc
    train_labels_2[i,3]=uniform(0.01,0.06)      #r_s
    '''
    
    #limites artigo_helal:
    train_labels_2[i,0]=uniform(0.0000001,0.000001) #I_0~I_sd
    train_labels_2[i,1]=uniform(0.01,0.03)          #G_sh...R_sh = G_sh^-1
    train_labels_1[i,0]=uniform(1.0,2.0)            #n (Ideality factor of the diode)       
    train_labels_2[i,2]=uniform(0.7,0.8)            #I_sc=I_ph
    train_labels_2[i,3]=uniform(0.03,0.04)          #r_s
    
    #print(int((100*i)/numero_de_dados),"% - ",i+1)

#Import my model of data to generate the same numbers of points (t):
dados_importar = np.genfromtxt("curva_0.txt", delimiter="", names="V, I")
v  = dados_importar['V']
J  = dados_importar['I']    

b=0
#gerar samples
for i in range(numero_de_dados):
    
    for j in range(numero_de_pontos):
        
        #primeira coluna:
        V = v[j]
        #segunda coluna:
        train_samples[i, j] = curva(
                train_labels_2[i,0],         #I_0
                train_labels_2[i,1],         #R_sh ou G_sh
                train_labels_1[i,0],         #n
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
    
#print("Train Finalizado!")

#Save the data to import in the training
np.save("train_helal_samples_n", train_samples, allow_pickle=True, fix_imports=True)
np.save("train_helal_labels_n", train_labels_1, allow_pickle=True, fix_imports=True)
