from matplotlib import pyplot as plt
import pandas as pd
import numpy as n
import math
import random
import numpy as np
from math import e
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import xlrd

entradas = np.array([])
saidas= np.array([])
#entradas 
book = xlrd.open_workbook('C:/Users/victo/Documents/entradas.xlsx')
sheet = book.sheet_by_name('sonar')
entradas  = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]

#saídas
book = xlrd.open_workbook('C:/Users/victo/Documents/saida.xlsx')
sheet = book.sheet_by_name('Planilha1')
saidas  = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]

x=np.array(entradas)
t=np.array(saidas)
x.transpose()
t.transpose()
print(len(x))
print(len(x[0]))

v=[]
w=[]
listaciclos=[]
listaerro=[]
neuroniosentreda=60
neuroniosescondido=int(input('Neuronios na camada escondida:'))
neuroniossaida=1
z=np.zeros((1,neuroniosescondido))
deltaw=np.zeros((neuroniosescondido,1))
zin=np.zeros((1,neuroniosescondido))
deltinhav=np.zeros((1,neuroniosescondido))
deltav=np.zeros((neuroniosentreda,neuroniosescondido))
deltabv=np.zeros((neuroniosescondido,1))
#inicialização dos pesos
alfa=float(input('Taxa de aprendizagem (0< alfa <=1):'))
errototaladmssivel=float(input('Erro total admissivel:'))
nciclos=int(input('Numero de ciclos maximo:'))
#pesos da camada escondida
v=np.random.uniform(-0.5, 0.5, (neuroniosentreda,neuroniosescondido))
bv=np.random.uniform(-0.5, 0.5, (neuroniosescondido,1))
#pesos da camada de saida
w=np.random.uniform(-0.5, 0.5, (neuroniosescondido,neuroniossaida))
bw=random.uniform(-0.5,0.5)
###########TREINAMENTO#############################
ciclo=0
acc2=0
errototal=10
g=0
while (ciclo<nciclos) and (errototal>errototaladmssivel):
    ciclo=ciclo+1
    errototal=0
    for i in range(208):
        for j in range(neuroniosescondido):
            for g in range(60):
                acc2= acc2 + (float(x[i][g])*float(v[g][j])) 
            zin[0][j]= acc2 +bv[j][0]
            z[0][j]=(2/(1+e**(-zin[0][j])))-1 #sigmoide

        acc=np.matmul(z,w)
        yin=acc+bw
        y=(2/(1+e**(-yin)))-1
        print('target [:',t[i],']  saida:',float(y))

    deltinhaw=(t[i]-y)*0.5*(1+y)*(1-y)
    for n in range(neuroniosescondido):
        deltaw[n][0]=alfa*deltinhaw*z[0][n]
        deltabw=alfa*deltinhaw

    for m in range(neuroniosescondido):
        deltinhav[0][m]=deltinhaw*w[m][0]*0.5*(1+z[0][m])*(1-z[0][m])

    for n in range(neuroniosentreda):
        for j in range(neuroniosescondido):
            deltav[n][j]=alfa*deltinhav[0][j]*float(x[i][n])
    for p in range(neuroniosescondido):
        deltabv[p][0]=alfa*deltinhav[0][p]
    ##########atualização de pesos#############
    w=w+deltaw
    bw=bw+deltabw
    for g in range(neuroniosentreda):
        for j in range(neuroniosescondido):
            v[g][j]=v[g][j]+deltav[g][j]
    for h in range(neuroniosescondido):
        bv[h][0]=bv[h][0]+deltabv[h][0]
        
    errototal=errototal+0.5*((t[i]-y)**2)
    listaciclos.append(ciclo)
    listaerro.append(float(errototal))
    print('errototal:',float(errototal))
#parte grafica
#grafico erro
print('CICLOS COMPLETOS:',ciclo)
for i in range(208):
    for j in range(neuroniosescondido):
        acc2=0
        for g in range(60):
            acc2= acc2 + (float(x[i][g])*float(v[g][j])) 
        zin[0][j]= acc2 +bv[j][0]
        z[0][j]=(2/(1+e**(-zin[0][j])))-1 #sigmoide
    acc=np.matmul(z,w)
    yin=acc+bw      
    y=(2/(1+e**(-yin)))-1
     print('target[:',t[i],']  saida:',float(y))

     print('w:',w)
     print('bw:',bw)
#for r in range(nciclos):
 #plt.scatter(listaciclos[r],listaerro[r])
 #plt.pause(0.001)
plt.plot(listaciclos,listaerro)
plt.show()
