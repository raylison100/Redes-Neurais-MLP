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
        x_data = np.zeros((len(data),len(data[0])-1))#x_data são os dados para treino.
        y_data = np.zeros(len(data))#y_data são as classe alvo.      
        for x in range(0,len(data)):#O for começa de 1 pois na primeira linha esta o cabeçario.
            x_data[x] = data[x][:-1]#Armazeno em x_data apenas as features. 
            for y in range(len(labels)):#dou um for na variavel labels
                if labels[y] in data[x]:#avalio qual classe esta contida na linha
                    y_data[x] = y#Substituo a string por um float no caso 0 ou 1.  
    return labels,x_data,y_data


class Mlp(object):
    """ 
       alpha: defaut=0.01 # A taxa de aprendizado.
       n_features: O número de features no seu dataset. Lembrar de Alterar de acordo com o tamanho do vertor de caracteristicas
       n_iter: O número de iterações realizadas pelo perceptron."""
 
    def __init__(self,alpha=0.01,n_features = 4,n_iter=1, intermedioario = 2, saida = 2):
        self.alhpa = alpha #taxa de aprendizagem
        self.n_iter = n_iter # Treino
        self.features = n_features

        self.sinapsesItermediaria = np.random.random((n_features,intermedioario)) #Pesos das snapses na camada Intermediaria
        self.sinapsesSaida = np.random.random((intermedioario,saida)) #Pesos das snapses na camada Saida

        self.neuronioIntermediarios = np.random.random(intermedioario) #Peso dos neuronios Intermediarios
        self.neuronioSaida = np.random.random(saida) #Peso dos neuronios Saida

        self.neuronioIntermediariosSigmoid = np.zeros(intermedioario)#Saida dos neuronios intermediarios
        self.neuronioSaidaSigmoid = np.linspace(-1,-1,saida)#Sainda dos neuronios de saida. Default -1

        self.tempErroCalculadoNeuroniosIntermediario = np.zeros(intermedioario)#Erro dos Neuronios intermediario
        self.tempErroCalculadoNeuroniosSaida = np.zeros(saida)#Erro dos Neuronios de saida

        self.tempAjusteNeuroniosIntermediarios = np.zeros(intermedioario) #Valor a atualizar dos neuronio intermediario
        self.tempAjusteNeuroniosSainda = np.zeros(saida)#Valor a atualizar dos neuronio saida

        self.tempAjustePesosIntermediarios = np.zeros((n_features,intermedioario))#Valor a atualizar das sinapses intermediario
        self.tempAjustePesosSainda = np.zeros((intermedioario,saida))#Valor a atualizar das sinapses de saida

        self.erroGerado = 0

        
    def sigmoid(self, x):
        resultado =(1 / (1 + np.exp(-x)))
        return resultado

    def somaEntradasIntermediario(self, y,x_data):
        soma = 0        
        for x in range(0,x_data.shape[0]):
            soma += x_data[x] * self.sinapsesItermediaria[x,y]  
        resultado  =  self.sigmoid(soma - (1 * self.neuronioIntermediarios[y]))                    
        return resultado
    
    def somaIntermediarioSainda(self,y,data ):
        soma = 0
        for x in range(0,data.shape[0]):
            soma += data[x] * self.sinapsesSaida[x,y] 
        return self.sigmoid(soma - (1 * self.neuronioSaida[y]))

    def validationResult(self,data):
        if self.neuronioSaidaSigmoid[data] != data and self.neuronioSaidaSigmoid[data] != -1 :# tem que validar se e nulo 
            self.calErroGerado(self.neuronioSaidaSigmoid[data],data)
            self.calErroNeuronioSaida(data)    
            self.calPesosSinapseNeuronioUpdateSainda(data)
            self.calErroNeuroniosIntermediarios(data)
            for y in range(0,self.neuronioIntermediarios.shape[0]):      
                self.calPesosSinapseNeuronioIntermediario(y,data) 
            self.updatePesos(data)        
                 
                                     
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
    
    def fit(self,x_data,y_data):
        a = 0
        x = 0
        y = 0
        w = 0
        self.Save("-----------Iniciando treinamento---------------\n")
        while a <= self.n_iter:  

            while x < x_data.shape[0] : #percorre toda base de dados           
                    
                while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses intermediarias
                    self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                    y = y + 1                    

                while w < self.neuronioSaida.shape[0]:# inicia o processo de soma das sinapses de saida 
                    self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                    w = w + 1

                self.validationResult(y_data[x])
                x = x + 1
                # break  
            a = a + 1
            # break            
        self.Save("-----------Fim do Treinamento---------------\n\n\n")

    def teste(self,x_data,y_data,labels):
        x = 0
        y = 0
        w = 0
        while x < x_data.shape[0] : #percorre toda base de dados           
                    
            while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses intermediarias
                self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                y = y + 1                    

            while w < self.neuronioSaida.shape[0]:# inicia o processo de soma das sinapses de saida 
                self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                w = w + 1

            self.validationTeste(y_data[x],labels)
            self.Save("---------------------Entrada---------------------\n")
            self.SaveNp(x_data[x,])
            self.Save("\n")
            x = x + 1
            self.printRede()
            break
            

    def validationTeste(self,classe,label):
        distanciaM = 0
        for x in range (0,self.neuronioSaidaSigmoid.shape[0]):
            distancia =  self.dist_euclidiana(classe,self.neuronioSaidaSigmoid[x])
            if(distancia >= distanciaM):
                distanciaM = distancia
                index = x
        print("CLasse ->  ",label[index])
        
                         
    def dist_euclidiana(self,v1, v2):
        soma =  math.pow(v1 - v2, 2)
        return math.sqrt(soma)

    def printRede(self):
        self.Save("------------------------------------------\n")
        self.SaveNp(self.sinapsesItermediaria) #Pesos das snapses na camada Intermediaria
        self.Save("\n")
        self.SaveNp(self.sinapsesSaida) #Pesos das snapses na camada Saida
        self.Save("\n")
        self.SaveNp(self.neuronioIntermediarios) #Pesos dos neuronios Intermediarios
        self.Save("\n")
        self.SaveNp(self.neuronioSaida)#Pesos dos neuronios de Saida
        self.Save("\n")
        self.SaveNp(self.neuronioIntermediariosSigmoid) #Saida dos neuronios intermediarios
        self.Save("\n")
        self.SaveNp(self.neuronioSaidaSigmoid) #Sainda dos neuronios de saida. Default -1
        self.Save("\n")
        self.Save("------------------------------------------\n")
    
    def Save(self,log):
        arquivo = open('log.txt','a')
        arquivo.write(log)
        arquivo.close()

    def SaveNp(self,x):
        with open('log.txt', 'a') as f:
            f.write(" ".join(map(str, x)))


#header: Contem o cabeçario do dataset sendo que a ultima coluna é a classe.
#label,x_data: Contem as features para treino ou seja as primeiras quatro colunas do conjuto.
#label,y_data: Contem as classes de cada linha de x_data, sendo 0 para setosa e 1 para versicolor

# labels,x_data,y_data = open_file("histoTreinamento.csv")
#labels,x_dataT,y_dataT = open_file("histoTeste.csv")

#labels,x_data,y_data = open_file("iris.csv")
labels,x_dataT,y_dataT = open_file("teste.csv")

# label,x_data,y_data = open_file("histogramas.csv")

perceptron = Mlp() #Instanciado MLPexit
# perceptron.fit(x_data,y_data) #iniciando treinamento
perceptron.printRede()
perceptron.teste(x_dataT,y_dataT,labels)