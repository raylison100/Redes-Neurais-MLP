import numpy as np
import csv as csv
import random
import matplotlib.pyplot as plt

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
        
        print(y_data)
    return header,x_data,y_data


#header: Contem o cabeçario do dataset sendo que a ultima coluna é a classe.
#x_data: Contem as features para treino ou seja as primeiras quatro colunas do conjuto.
#y_data: Contem as classes de cada linha de x_data, sendo 0 para setosa e 1 para versicolor
header,x_data,y_data = open_file("iris.csv")
 
plt.scatter(x_data[:,1],x_data[:,0],c=y_data)
plt.title("Setosa x versicolor" )
plt.xlabel('Sepal.Width')
plt.ylabel('Sepal.Length')

#Um perceptron para 3 inputs
class Perceptron(object):
    """Essa classe consiste em um perceptron: um modelo
       linear consedido por  Frank Rosenblatt.
       Essa classe é apenas para classificação.
       alpha: defaut=0.01 # A taxa em que o erro sera propagado.
       n_features: O número de features no seu dataset.
       n_iter: O número de iterações realizadas pelo perceptron."""
 
    def __init__(self,alpha=0.02,n_features =3,n_iter=1000):
        self.w = np.random.uniform(1,n_features)
        self.alhpa = alpha
        self.n_iter = n_iter
 
    def _0_1_loss(self,x): #Função de perda que avalia o output
        if x.all() >= 0.0:
            return 1
        else:
            return 0
 
    def fit(self,x_data,y_data):
        """Esse método é usado para treinar o percetron.
           x_data: Uma numpy.array contendo as features para treino.
           y_data: Uma numpy.array contendo as classes(target)"""
 
        x_data = np.insert(x_data[:,],len(x_data[0]),1,axis=1)#acrecentamos o bias ao dataset
        for x in range(self.n_iter):                          #no caso mais uma coluna contendo apenas 1
            print("iteração número:{}".format(x))
            cum_erro = 0 #Aqui armazenamos o erro acumulado para parar a otimização
            for y in range(len(x_data)):
                output = np.dot(self.w,(x_data[y]))#O output é o produto dos pesos pela linha atual
                if self._0_1_loss(output) != y_data[y]: #avaliamos para ver se é correspondente.
                    cum_erro += 1 #Caso não seja acrecentamos a contagem de erro
                    erro = y_data[y] - output #medimos o erro da iteração de forma direta.(sem loss)
                    self.w += self.alhpa*erro*x_data[y] #Aqui os pesos são atualizados
            if cum_erro == 0: #Aqui avaliamos o erro acumulado caso sejá 0 para o treinamento.
                # print("Otimização terminada em {} iterações".format(x))
                break                
 
    def predict(self,vector):
        """O método predict pode levar uma numpy.array de uma ou duas
           dimensões."""
        if np.ndim(vector) == 1:#Avaliamos a quantidade de dimensões.
            vector = np.insert(vector,len(vector),1)#inserimos o bias
            prediction = self._0_1_loss(self.w.dot(vector))#fazemos a predição.
            return prediction
        else:#Caso contrario é feito o mesmo processo porem com uma array de duas dimensões.
            vector = np.insert(vector[:,],len(vector[0]),1,axis=1)
            prediction = [self._0_1_loss(self.w.dot(x)) for x in vector]
            return prediction

def showAccuracy(y_true,predictions):
    correct = 0
    for x in range(len(y_true)):
        if y_true[x] == predictions[x]:#Compara os respectivos valores
            correct += 1
    return float(correct)/len(y_true)



perceptron = Perceptron(n_features=4)
perceptron.fit(x_data,y_data)

print(perceptron.predict(x_data[35]))
print(perceptron.predict(x_data)) # isso ira retornar uma array contendo todas as labels

predi = perceptron.predict(x_data)

print( showAccuracy(y_data,predi))
