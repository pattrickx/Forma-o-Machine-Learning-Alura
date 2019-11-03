####### aula 5 parte 2
from sklearn.svm import LinearSVC ##biblioteca para treno e clasificação
from sklearn.metrics import accuracy_score  ## biblioteca para conseguirmetricas dos dados
from sklearn.model_selection import train_test_split # biblioteca para separação de dados de teste e treino
import numpy as np
import pandas as pd
from datetime import datetime ## biblioteca que pega momento atual
uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
dados = pd.read_csv(uri)
print(dados.head())
mapa ={
    "mileage_per_year":"milhas_por_ano",
    "model_year": "ano_do_modelo",
    "price": "preco",
    "sold": "vendido"
}
dados = dados.rename(columns=mapa)
print(dados.head())
mapa= {
    "yes" : 1,
    "no" : 0 
}
dados["vendido"]=dados.vendido.map(mapa)
print(dados.head())


ano_atual=datetime.today().year
dados["idade_do_modelo"]=ano_atual-dados.ano_do_modelo
print(dados.head())
dados["km_por_ano"]= dados.milhas_por_ano*1.60934
print(dados.head())
dados= dados.drop(columns= ["Unnamed: 0","milhas_por_ano","ano_do_modelo"], axis=1) ## remove colunas, do eixo 1
print(dados.head())
################ separando dados
x= dados[["preco","idade_do_modelo","km_por_ano"]]
y= dados["vendido"]
############# previsão com 

SEED= 20 #define ordem para aleatoriedade do random_state
np.random.seed(SEED)
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y, test_size=0.25, stratify =y)
print("treino feito com: ", len(treino_x)," e teste feito com: ", len(teste_x))
############ treinando rede
model=LinearSVC()
model.fit(treino_x,treino_y) ## treina rede
########### aplicando teste
previsoes= model.predict(teste_x) ## preve teste
########### verificando acuracia/ taxa de acerto 
taxa_de_acertos= accuracy_score(teste_y, previsoes)*100
print("taxa de acertos: %.2f%%" %(taxa_de_acertos))
############################################## aula 5 parte 4

############### clsificador burro estratificado
from sklearn.dummy import DummyClassifier
dummy_stratified= DummyClassifier()
dummy_stratified.fit(treino_x, treino_y)
acuracia = dummy_stratified.score(teste_x,teste_y)*100
print("taxa de acertos dummy_stratified: %.2f%%" %(acuracia))
############### clsificador burro maior frequencia
dummy_mostfrequent= DummyClassifier()
dummy_mostfrequent.fit(treino_x, treino_y)
acuracia = dummy_mostfrequent.score(teste_x,teste_y)*100
print("taxa de acertos dummy_mostfrequent: %.2f%%" %(acuracia))

############################ previsão por svc
from sklearn.svm import SVC ##biblioteca para treno e clasificação
from sklearn.preprocessing import StandardScaler ## ajuste de escala

SEED= 5 #define ordem para aleatoriedade do random_state
np.random.seed(SEED)
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x,y, test_size=0.25, stratify =y)
print("treino feito com: ", len(treino_x)," e teste feito com: ", len(teste_x))
############ treinando rede

scaler = StandardScaler() 
scaler.fit(treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

model=SVC()
model.fit(treino_x,treino_y) ## treina rede
########### aplicando teste
previsoes= model.predict(teste_x) ## preve teste
########### verificando acuracia/ taxa de acerto 
taxa_de_acertos= accuracy_score(teste_y, previsoes)*100
print("taxa de acertos: %.2f%%" %(taxa_de_acertos))

################################################# aula 5 parte 5
from sklearn.tree import DecisionTreeClassifier
SEED= 5 #define ordem para aleatoriedade do random_state
np.random.seed(SEED)
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y, test_size=0.25, stratify =y)
print("treino feito com: ", len(treino_x)," e teste feito com: ", len(teste_x))
############ treinando rede

# scaler = StandardScaler() 
# scaler.fit(treino_x)
# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

model=DecisionTreeClassifier(max_depth=10)# max_depth=3 limita profundidade da arvore
model.fit(treino_x,treino_y) ## treina rede
########### aplicando teste
previsoes= model.predict(teste_x) ## preve teste
########### verificando acuracia/ taxa de acerto 
taxa_de_acertos= accuracy_score(teste_y, previsoes)*100
print("taxa de acertos: %.2f%%" %(taxa_de_acertos))

from sklearn.tree import export_graphviz
import graphviz 
feature=x.columns
dot_data = export_graphviz(model, out_file=None, filled=True, rounded= True, feature_names=feature, class_names=["não","sim"])
#print(dot_date)

import pydotplus # lib para criar grafico de pontos
grafico = pydotplus.graph_from_dot_data(dot_data)  

from IPython.display import Image  # lib para criar a imagem 
Image(grafico.create_png())
grafico.write_png("tree.png")
###### n esquecer de instalar graphviz no sistema operacional, criar variavel de sistema e instalar a biblioteca do python
