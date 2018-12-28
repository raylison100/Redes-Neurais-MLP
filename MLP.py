import numpy as np
import csv as csv
import random
import matplotlib.pyplot as plt

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
 
    def __init__(self,alpha=0.01,n_features = 4,n_iter=10, intermedioario = 4, saida = 2):
        self.alhpa = alpha #taxa de aprendizagem
        self.n_iter = n_iter # Treino

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
            if self.neuronioSaidaSigmoid[x] != data and self.neuronioSaidaSigmoid[x] != -1 :# tem que validar se e nulo 
                self.calErroGerado(self.neuronioSaidaSigmoid[x],x)
                self.calErroNeuronioSaida(x)
         

    def calErroGerado(self, valorEsperado, valorObtido):
        self.erroGerado = valorEsperado - valorObtido

    def calErroNeuronioSaida(self,x):
       self.tempErroCalculadoNeuroniosSaida[x] = (self.neuronioSaidaSigmoid[x]*(1-self.neuronioSaidaSigmoid[x])*self.erroGerado)
        
    def calPesoSinapseUpdateSainda(self,x):
        index = x
        while x <= index:
            self.tempAjustePesosSainda[x,index] = self.alhpa * self.neuronioIntermediariosSigmoid[x] * self.tempErroCalculadoNeuroniosSaida[x]
        
    # def calPesoNeuronioUpdateSainda(self):
    # def calupdatePesos(self):            
    
    def fit(self,x_data,y_data):
        a = 0
        x = 0
        y = 0
        w = 0
        while a <= self.n_iter:  

            while x < x_data.shape[0] : #percorre toda base de dados           
                    
                while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses
                    self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                    y = y + 1                    

                while w < self.neuronioSaida.shape[0]:
                    self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                    w = w + 1

                self.validationResult(y_data[0])
                x = x + 1
                break
            a = a + 1
            break
            
        


#header: Contem o cabeçario do dataset sendo que a ultima coluna é a classe.
#x_data: Contem as features para treino ou seja as primeiras quatro colunas do conjuto.
#y_data: Contem as classes de cada linha de x_data, sendo 0 para setosa e 1 para versicolor
header,x_data,y_data = open_file("iris.csv")



perceptron = Mlp() #Instanciado MLP
perceptron.fit(x_data,y_data) #iniciando treinamento

 