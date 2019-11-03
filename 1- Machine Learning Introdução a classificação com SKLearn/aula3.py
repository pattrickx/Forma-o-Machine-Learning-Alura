######### aula 3 parte 2
import pandas as pd
uri="https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
print(dados.head())
mapa={
    "unfinished" : "nao_finalizado",
    "expected_hours":"horas_esperadas",
    "price":"preco"

}
dados = dados.rename(columns = mapa)
print(dados.head())
####### inverter valores para facilitar a utilisação 
troca ={
    1:0,
    0:1
}
dados["finalizado"]=dados.nao_finalizado.map(troca)  ## cria nova coluna com os dados invertidos 
print(dados.head())

import seaborn as sns 
import matplotlib.pyplot as plt ## biblioteca grafica 
sns.scatterplot(x= "horas_esperadas", y= "preco", data= dados) #gera grafico de disperção 
plt.show() ## mostra grafico
sns.scatterplot(x= "horas_esperadas", y= "preco", hue="finalizado", data= dados) #difere os grupos com cor usando o hue 
plt.show()
sns.relplot(x= "horas_esperadas", y= "preco", col="finalizado",hue="finalizado" , data= dados) # cria grafico de barras difere os grupos com cor usando o col
plt.show()

x= dados[["horas_esperadas", "preco"]]
y= dados["finalizado"]

from sklearn.svm import LinearSVC ##biblioteca para treno e clasificação
from sklearn.metrics import accuracy_score  ## biblioteca para conseguirmetricas dos dados
from sklearn.model_selection import train_test_split # biblioteca para separação de dados de teste e treino

SEED= 20 #define ordem para aleatoriedade do random_state
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y, random_state=SEED, test_size=0.25, stratify =y)
print("treino feito com: ", len(treino_x)," e teste feito com: ", len(teste_x))
############ treinando rede
model=LinearSVC()
model.fit(treino_x,treino_y) ## treina rede
########### aplicando teste
previsoes= model.predict(teste_x) ## preve teste
########### verificando acuracia/ taxa de acerto 
taxa_de_acertos= accuracy_score(teste_y, previsoes)*100
print("taxa de acertos: %.2f%%" %(taxa_de_acertos))
############################################## aula 3  4
sns.scatterplot(x= "horas_esperadas", y= "preco", data= teste_x) #gera grafico de disperção 
plt.show() ## mostra grafico
import numpy as np
x_min= teste_x.horas_esperadas.min()
x_max= teste_x.horas_esperadas.max()
y_min= teste_x.preco.min()
y_max= teste_x.preco.max()
pixel=100
eixo_x=np.arange(x_min,x_max,(x_max-x_min)/pixel)
eixo_y=np.arange(y_min,y_max,(y_max-y_min)/pixel)

xx, yy=np.meshgrid(eixo_x,eixo_y)
pontos =np.c_[xx.ravel(),yy.ravel()]
print(pontos)
z= model.predict(pontos)
z= z.reshape(xx.shape)
print(z)
plt.contourf(xx,yy,z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco,c=teste_y ,s=1)
plt.show()

