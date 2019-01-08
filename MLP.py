import numpy as np
import csv as csv
import random
import matplotlib.pyplot as plt
import math

''' 
    Planejamento de Desenvolvimento
    1 - Importar a base de dados. -----DONE
    2 - Definir a quantidade de neuronio na camada intermediaria e de saida. -----DONE
    3 - Setar os pesos das sinapses de ligação com cada neuronio aleatoriamente. -----DONE
    4 - Contruir funções de calculo do valores sinapticos. -----DONE
    5 - Calcular função do erro do neuronio. 
    6 - Calcular erro das sinapses. 
    7 - Realizar a função de propagação do erro.
    8 - Contruir função de treinamento.
'''

'''
    TUTORIAL DO ALGORITMO
    Passo 1: Atribuir valores aleatorios para os pesos e limites
    Passo 2: Calcular os valores dos neuronios na camada oculta
    Passo 3: Calcular o erro do neuronio na camada de saida
'''

def open_file(path):
    """Essa função organiza o dataset para treino
       considerando que as classes que serão o target
       estejam na ultima coluna."""
 
    with open(path) as dataset: #Usamos with pois garante fechar o documento.
        data = np.array(list(csv.reader(dataset)))#Armazenamos todo o dataset em uma array.
        labels = np.array(list(set(data[1:,-1])))#Essa operação é util para eliminar valores repetidos.
        header  = data[0] #Esse é o cabeçario da tabela.
        x_data = np.zeros((len(data)-1,len(data[0])-1))#x_data são os dados para treino.
        y_data = np.empty(len(data)-1)#y_data são as classe alvo.
 
        for x in range(1,len(data)):#O for começa de 1 pois na primeira linha esta o cabeçario.
            x_data[x-1] = data[x][:-1]#Armazeno em x_data apenas as features.
 
            for y in range(len(labels)):#dou um for na variavel labels
                if labels[y] in data[x]:#avalio qual classe esta contida na linha
                    y_data[x-1] = y#Substituo a string por um float no caso 0 ou 1.
    return header,x_data,y_data


class Mlp(object):
    """
       alpha: defaut=0.01 # A taxa de aprendizado.
       n_features: O número de features no seu dataset.
       n_iter: O número de iterações realizadas pelo perceptron."""
 
    def __init__(self,alpha=0.01,n_features = 4,n_iter=500, intermedioario = 4, saida = 2):
        self.alhpa = alpha #taxa de aprendizagem
        self.n_iter = n_iter # Treino
        self.features = n_features

        self.sinapsesItermediaria = np.random.random((n_features,intermedioario)) #Pesos na camada Intermediaria
        self.sinapsesSaida = np.random.random((intermedioario,saida)) #Pesos na camada Saida

        self.neuronioIntermediarios = np.random.random(intermedioario) #Neuronios Intermediarios
        self.neuronioSaida = np.random.random(saida) #Neuronios Saida

        self.neuronioIntermediariosSigmoid = np.zeros(intermedioario)#Saida dos neuronios intermediarios
        self.neuronioSaidaSigmoid = np.linspace(-1,-1,saida)#Sainda dos neuronios de saida. Default -1

        self.tempErroCalculadoNeuroniosIntermediario = np.zeros(intermedioario)#Erro do Neuronio intermediario
        self.tempErroCalculadoNeuroniosSaida = np.zeros(saida)#Erro do Neuronio saida

        self.tempAjusteNeuroniosIntermediarios = np.zeros(intermedioario) #Valor a atualizar neuronio intermediario
        self.tempAjusteNeuroniosSainda = np.zeros(saida)#Valor a atualizar neuronio saida

        self.tempAjustePesosIntermediarios = np.zeros((n_features,intermedioario))#Valor a atualizar sinapse intermediario
        self.tempAjustePesosSainda = np.zeros((intermedioario,saida))#Valor a atualizar sinapse saida

        self.erroGerado = 0

        
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def somaEntradasIntermediario(self, y,x_data):
        soma = 0        
        for x in range(0,x_data.shape[0]):
            soma += x_data[x] * self.sinapsesItermediaria[x,y]                       
        return self.sigmoid(soma - (1 * self.neuronioIntermediarios[y]))
    
    def somaIntermediarioSainda(self,y,data ):
        soma = 0
        for x in range(0,data.shape[0]):
            soma += data[x] * self.sinapsesSaida[x,y] 
        return self.sigmoid(soma - (1 * self.neuronioSaida[y]))

    def validationResult(self,data):
        for x in range(0,self.neuronioSaida.shape[0]):
            if self.neuronioSaidaSigmoid[x] != x and self.neuronioSaidaSigmoid[x] != -1 :# tem que validar se e nulo 
                self.calErroGerado(self.neuronioSaidaSigmoid[x],x)
                self.calErroNeuronioSaida(x)    
                self.calPesosSinapseNeuronioUpdateSainda(x)
                self.calErroNeuroniosIntermediarios(x)
                for y in range(0,self.neuronioIntermediarios.shape[0]):      
                     self.calPesosSinapseNeuronioIntermediario(y,data) 
                self.updatePesos(x)     
                                     
    def calErroGerado(self, valorEsperado, valorObtido):
        self.erroGerado = valorEsperado - valorObtido
 
    def calErroNeuronioSaida(self,x):
       self.tempErroCalculadoNeuroniosSaida[x] = self.neuronioSaidaSigmoid[x]*(1-self.neuronioSaidaSigmoid[x])*self.erroGerado
        
    def calPesosSinapseNeuronioUpdateSainda(self,neuronioSaida):
        for x in range(0,self.neuronioIntermediariosSigmoid.shape[0]):
            self.tempAjustePesosSainda[x,neuronioSaida] = (self.alhpa * self.neuronioIntermediariosSigmoid[x] * self.tempErroCalculadoNeuroniosSaida[neuronioSaida]) + self.sinapsesSaida[x,neuronioSaida]
        self.tempAjusteNeuroniosSainda[neuronioSaida] = (self.alhpa * (-1) * self.tempErroCalculadoNeuroniosSaida[neuronioSaida]) + self.neuronioSaida[neuronioSaida]
        
    def calErroNeuroniosIntermediarios(self,neuronioSainda):
        for x in range(0,self.neuronioIntermediariosSigmoid.shape[0]):
            self.tempErroCalculadoNeuroniosIntermediario[x] = self.neuronioIntermediariosSigmoid[x] * (1 - self.neuronioIntermediariosSigmoid[x]) * (self.tempErroCalculadoNeuroniosSaida[neuronioSainda] * self.sinapsesSaida[x,neuronioSainda])

    def calPesosSinapseNeuronioIntermediario(self, neuronioIntermediario,data):
        for x in range(0,data.shape[0]):
            self.tempAjustePesosIntermediarios[x,neuronioIntermediario] = (self.alhpa * data[x] * self.tempErroCalculadoNeuroniosIntermediario[neuronioIntermediario]) + self.sinapsesItermediaria[x,neuronioIntermediario]
        self.tempAjusteNeuroniosIntermediarios[neuronioIntermediario] = (self.alhpa * (-1) * self.tempErroCalculadoNeuroniosIntermediario[neuronioIntermediario]) + self.neuronioIntermediarios[neuronioIntermediario]
              
    def updatePesos(self, neuronioSaida):  

        for x in range(0,self.neuronioIntermediariosSigmoid.shape[0]):
            self.sinapsesSaida[x,neuronioSaida] = self.tempAjustePesosSainda[x,neuronioSaida]
        self.neuronioSaida[neuronioSaida] = self.tempAjusteNeuroniosSainda[neuronioSaida]
        
        for y in range(0,self.neuronioIntermediarios.shape[0]):
            for x in range(0,self.features):
                self.sinapsesItermediaria[x,y] = self.tempAjustePesosIntermediarios[x,y]
            self.neuronioIntermediarios[y] = self.tempAjusteNeuroniosIntermediarios[y]       
    
    def fit(self,x_data):
        a = 0
        x = 0
        y = 0
        w = 0
        while a <= self.n_iter:  

            while x < x_data.shape[0] : #percorre toda base de dados           
                    
                while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses intermediarias
                    self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                    y = y + 1                    

                while w < self.neuronioSaida.shape[0]:# inicia o processo de soma das sinapses de saida 
                    self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                    w = w + 1

                self.validationResult(x_data[x,])
                x = x + 1
                # break
            a = a + 1
            # break
        print("Fim da execucao")

    def teste(self,x_data,y_data):
        a = 0
        x = 0
        y = 0
        w = 0
        while a <= self.n_iter:  

            while x < x_data.shape[0] : #percorre toda base de dados           
                    
                while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses intermediarias
                    self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                    y = y + 1                    

                while w < self.neuronioSaida.shape[0]:# inicia o processo de soma das sinapses de saida 
                    self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                    w = w + 1

                self.validationTeste(y_data[x])
                x = x + 1
                
            a = a + 1
            

    def validationTeste(self,x):
        distancia0 = self.dist_euclidiana(x,self.neuronioSaidaSigmoid[0])
        distanica1 = self.dist_euclidiana(x,self.neuronioSaidaSigmoid[1])
        if(distancia0 > distanica1):
            print("Versicolor") 
        else:
            print("Setosa")
    
    def dist_euclidiana(self,v1, v2):
        soma =  math.pow(v1 - v2, 2)
        return math.sqrt(soma)
 


#header: Contem o cabeçario do dataset sendo que a ultima coluna é a classe.
#x_data: Contem as features para treino ou seja as primeiras quatro colunas do conjuto.
#y_data: Contem as classes de cada linha de x_data, sendo 0 para setosa e 1 para versicolor
header,x_data,y_data = open_file("iris.csv")
headerT,x_dataT,y_dataT = open_file("teste.csv")

perceptron = Mlp() #Instanciado MLP
perceptron.fit(x_data) #iniciando treinamento
perceptron.teste(x_dataT,y_dataT)