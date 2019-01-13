import numpy as np
import csv as csv
import random
import matplotlib.pyplot as plt
import math
from  tempfile  import  TemporaryFile 

''' 
    Planejamento de Desenvolvimento
    1 - Importar a base de dados. -----DONE
    2 - Definir a quantidade de neuronio na camada intermediaria e de saida. -----DONE
    3 - Setar os pesos das sinapses de ligação com cada neuronio aleatoriamente. -----DONE
    4 - Contruir funções de calculo do valores sinapticos. -----DONE
    5 - Calcular função do erro do neuronio. -----DONE
    6 - Calcular erro das sinapses. -----DONE
    7 - Realizar a função de propagação do erro.-----DONE
    8 - Contruir função de treinamento.-----DONE
    9 - COntruir relatorio -----DONE
'''

'''
    TUTORIAL DO ALGORITMO
    Passo 1: Atribuir valores aleatorios para os pesos e limites
    Passo 2: Calcular os valores dos neuronios na camada oculta
    Passo 3: Calcular o erro do neuronio na camada de saida
'''

def classes(path):
    with open(path) as dataset:
        data = np.array(list(csv.reader(dataset)))#Armazenamos todo o dataset em uma array.
        labels = np.array(list(set(data[1:,-1])))#Essa operação é util para eliminar valores repetidos.
    return labels

def open_file(path,label):
    with open(path) as dataset: #Usamos with pois garante fechar o documento.
        data = np.array(list(csv.reader(dataset)))#Armazenamos todo o dataset em uma array.
        x_data = np.zeros((len(data),len(data[0])-1))#x_data são os dados para treino.
        y_data = np.zeros(len(data))#y_data são as classe alvo.      
        for x in range(0,len(data)):#O for começa de 1 pois na primeira linha esta o cabeçario.
            x_data[x] = data[x][:-1]#Armazeno em x_data apenas as features. 
            for y in range(len(labels)):#dou um for na variavel labels
                if labels[y] in data[x]:#avalio qual classe esta contida na linha
                    y_data[x] = y#Substituo a string por um float no caso 0 ou 1.  
    return x_data,y_data

class Mlp(object):
    """ 
       alpha: defaut=0.01 # A taxa de aprendizado.
       n_features: O número de features no seu dataset. Lembrar de Alterar de acordo com o tamanho do vertor de caracteristicas
       n_iter: O número de iterações realizadas pelo perceptron."""
 
    def __init__(self,alpha,n_features,n_iter, intermedioario, saida,confisao):
        self.alhpa = alpha #taxa de aprendizagem
        self.n_iter = n_iter # Treino
        self.features = n_features
        self.confusao = np.zeros([confisao,confisao], dtype = int)
        self.acuracia = 0
        self.somaTotal = 0
        self.erro =0
        self.precisao = 0
        self.recall = 0
        self.fmeasure = 0
        self.vp = 0
        self.fp = 0
        self.fn = 0

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

    def validationResult(self,classe,entrada):
        if self.neuronioSaidaSigmoid[classe] != classe and self.neuronioSaidaSigmoid[classe] != -1 :# tem que validar se e nulo 
            self.calErroGerado(self.neuronioSaidaSigmoid[classe],classe)
            self.calErroNeuronioSaida(classe)    
            self.calPesosSinapseNeuronioUpdateSainda(classe)
            self.calErroNeuroniosIntermediarios(classe)
            for y in range(0,self.neuronioIntermediarios.shape[0]):      
                self.calPesosSinapseNeuronioIntermediario(y,entrada) 
            self.updatePesos(classe)                      
                                     
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
        self.Save("---------Inicio do Treinamento------------\n")
        self.printRede()
        a = 0
        x = 0
        y = 0
        w = 0
        while a < self.n_iter:
            while x < x_data.shape[0] : #percorre toda base de dados           
                while y < self.neuronioIntermediarios.shape[0]:# inicia o processo de soma das sinapses intermediarias
                    self.neuronioIntermediariosSigmoid[y] = self.somaEntradasIntermediario(y,x_data[x,])
                    y = y + 1                    
                while w < self.neuronioSaida.shape[0]:# inicia o processo de soma das sinapses de saida 
                    self.neuronioSaidaSigmoid[w] = self.somaIntermediarioSainda(w,self.neuronioIntermediariosSigmoid)   
                    w = w + 1  
                self.validationResult(int(y_data[x]),x_data[x,])
                x = x + 1 
            a = a + 1      
        self.Save("-----------Fim do Treinamento-------------\n")
        self.printRede()#MLP POS TREINO

    def teste(self,x_data,y_data,labels):
        self.Save("############# Iniciando teste ############\n\n")
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
            # self.Save("Entrada ----> ")
            # self.Save(str(x_data[x,]))
            # self.Save("\n")
            x = x + 1
        self.CarregaRelatorio()
                      
    def validationTeste(self,classe,label):
        distanciaM = 0
        index = -1
        for x in range (0,self.neuronioSaidaSigmoid.shape[0]):
            distancia =  self.dist_euclidiana(classe,self.neuronioSaidaSigmoid[x])
            if(distancia > distanciaM or x == 0):
                distanciaM = distancia
                index = x
        self.matrizConfusao(int(classe),index)
        print("CLasse ->  ",label[index])
                     
    def dist_euclidiana(self,v1, v2):
        soma =  math.pow(v1 - v2, 2)
        return math.sqrt(soma)

    def CarregaRelatorio(self):
        self.somaTotalFunction()
        self.verdadeirosPositivos()
        self.falsosNegativos()
        self.falsosPositivos()
        self.acuraciaFunction()
        self.erroFunction()
        self.precisaoFunction()
        self.recallFunction()
        self.fMeasureFunction()
        # LOG em arquivo
        self.Save("\n##########################################")
        self.Save("\n---------------Relatorio------------------\n\n")
        self.Save("Matriz de confusao:\n") 
        self.Save(str(self.confusao))  
        self.Save("\nTotal de elementos: ")  
        self.Save(str(self.somaTotal))
        self.Save("\nVerdadeiros positivos: ")
        self.Save(str(self.vp))
        self.Save("\nFalsos positivos: ")
        self.Save(str(self.fp))
        self.Save("\nFalsos negativos: ")
        self.Save(str(self.fn))
        self.Save("\nAcuracia: ")
        self.Save(str(self.acuracia)) 
        self.Save("\nErro: ")
        self.Save(str(self.erro))
        self.Save("\nPrecisao: ")
        self.Save(str(self.precisao)) 
        self.Save("\nRecall: ")
        self.Save(str(self.recall)) 
        self.Save("\nF-Measure:")
        self.Save(str(self.fmeasure))  
        self.Save("\n\n##########################################\n")
        self.Save("-------------FIM DA EXECUSAO--------------\n")      
        self.Save("##########################################\n\n")                    

    def matrizConfusao(self,esperado,obtido):
        self.confusao[esperado,obtido] += 1 
        
    def somaTotalFunction(self):
        soma = 0
        for x in range(0,self.confusao.shape[0]):
            for y in range(0,self.confusao.shape[1]):
                soma += self.confusao[x,y]
        self.somaTotal = soma

    def acuraciaFunction(self):        
        self.acuracia = self.vp/self.somaTotal
    
    def erroFunction(self):        
        self.erro = (self.fp + self.fn)/self.somaTotal
    
    def precisaoFunction(self):
        self.precisao = self.vp/(self.vp + self.fp)
    
    def recallFunction(self):
        self.recall = self.vp/(self.vp + self.fn)
    
    def fMeasureFunction(self):
        self.fmeasure = 2 * ((self.precisao*self.recall)/(self.precisao+self.recall))

    def verdadeirosPositivos(self):
        somaDiagonal = 0
        for x in range(0,self.confusao.shape[0]):
            somaDiagonal += self.confusao[x,x]
        self.vp = somaDiagonal

    def falsosPositivos(self):
        fp = 0
        for x in range(0,self.confusao.shape[0]):
            for y in range(0,self.confusao.shape[1]):
                if(x != y and y < x ):
                    fp += self.confusao[x,y]
        self.fp = fp   

    def falsosNegativos(self):
        fn = 0
        for x in range(0,self.confusao.shape[0]):
            for y in range(0,self.confusao.shape[1]):
                if(x != y and y > x ):
                    fn += self.confusao[x,y]
        self.fn = fn

    def printRede(self):
        self.Save("------------------------------------------")
        self.Save("\n#Pesos das snapses na camada Intermediaria: \n")
        self.Save(str(self.sinapsesItermediaria))
        self.Save("\n\n#Pesos das snapses na camada Saida\n")
        self.Save(str(self.sinapsesSaida))
        self.Save("\n\n#Pesos dos neuronios Intermediarios\n")
        self.Save(str(self.neuronioIntermediarios)) 
        self.Save("\n\n#Pesos dos neuronios de Saida\n")
        self.Save(str(self.neuronioSaida))
        # self.Save("\n\n#Saida dos neuronios intermediarios\n")
        # self.Save(str(self.neuronioIntermediariosSigmoid)) 
        # self.Save("\n\n#Sainda dos neuronios de saida. Default -1\n")
        # self.Save(str(self.neuronioSaidaSigmoid)) 
        self.Save("\n")
        self.Save("------------------------------------------\n\n")
    
    def Save(self,log):
        arquivo = open('log.txt','a')
        arquivo.write(log)
        arquivo.close()

""" dataset """ #configuracao da base

labels = classes("Bases/iris.csv")
x_data,y_data = open_file("Bases/iris.csv",labels)
x_dataT,y_dataT = open_file("Bases/teste.csv",labels)

# labels = classes("Bases/histoTreinamento.csv")
# x_data,y_data = open_file("Bases/histoTreinamento.csv",labels)
# x_dataT,y_dataT = open_file("Bases/histoTeste.csv",labels)

# labels = classes("Bases/histogramas10.csv")
# x_data,y_data = open_file("Bases/histogramas10.csv",labels)
# x_dataT,y_dataT = open_file("Bases/histogramas10T.csv",labels)

""""""
""" MPL """ #configuracao do MPL

perceptron = Mlp(alpha=0.01,n_features = x_data.shape[1],n_iter=10000, intermedioario = 4, saida = labels.shape[0],confisao = labels.shape[0] ) #Instanciado MLP
perceptron.fit(x_data,y_data) #Iniciando treinamento
perceptron.teste(x_dataT,y_dataT,labels)#Testando