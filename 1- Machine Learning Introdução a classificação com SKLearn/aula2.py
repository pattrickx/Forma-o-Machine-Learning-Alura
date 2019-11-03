###################### aula 2 parte 2 
'''
import pandas as pd
uri="https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
print(pd.read_csv(uri))
dados=pd.read_csv(uri) # pegando dados do link
print(dados[["home","how_it_works","contact"]])
x=dados[["home","how_it_works","contact"]] # pasando parte dos dados para uma variavel
y=dados["bought"]
print(y.head()) ## .head() mostra os primeiros 5 elementos
###### alterando dados
mapa= {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"

}
dados = dados.rename(columns = mapa)

x=dados[["principal","como_funciona","contato"]]
y=dados["comprou"]
print(x.head())
print(y.head())
print(dados.shape) #.shape mostra o formato dos dados
##### definir dados pra treino e teste
treino_x =x[:75] ##pega dados ate o 75
treino_y =y[:75]
teste_x =x[75:] ##pega dados apartir do 76
teste_y =y[75:]

from sklearn.svm import LinearSVC ##biblioteca para treno e clasificação
model=LinearSVC()
model.fit(treino_x,treino_y) ## treina rede
previsoes= model.predict(teste_x) ## preve teste

from sklearn.metrics import accuracy_score  ## biblioteca para conseguirmetricas dos dados
taxa_de_acertos= accuracy_score(teste_y, previsoes)
print("taxa de acertos: %.2f%%" %(taxa_de_acertos*100))

######################################################3 aula 2 parte 5

from sklearn.model_selection import train_test_split # biblioteca para separação de dados de teste e treino
SEED= 20 #define ordem para aleatoriedade do random_state
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y, random_state=SEED, test_size=0.25, stratify =y)
print(treino_x.shape)
print(teste_x.shape)
'''

########################### ajustando codigo
import pandas as pd ## biblioteca usada para ler o svc no uri
from sklearn.svm import LinearSVC ##biblioteca para treno e clasificação
from sklearn.metrics import accuracy_score  ## biblioteca para conseguirmetricas dos dados
from sklearn.model_selection import train_test_split # biblioteca para separação de dados de teste e treino
############ pegando dados do link
uri="https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados=pd.read_csv(uri) 
print(dados.head())
############### alterando dados
mapa= {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"

}
dados = dados.rename(columns = mapa)
##### separando dados de resultados 
x=dados[["principal","como_funciona","contato"]]
y=dados["comprou"]
print(x.head()) 
print(y.head())
print(dados.shape) #.shape mostra o formato dos dados
############## separando dados de treino e teste
SEED= 20 #define ordem para aleatoriedade do random_state
#stratify =y torna proporção de resultados para treno e teste semelantes baseado nos resultados
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y, random_state=SEED, test_size=0.25, stratify =y)
print(treino_x.shape)
print(teste_x.shape)
print(treino_y.value_counts())
print(teste_y.value_counts())
############ treinando rede
model=LinearSVC()
model.fit(treino_x,treino_y) ## treina rede
########### aplicando teste
previsoes= model.predict(teste_x) ## preve teste
########### verificando acuracia/ taxa de acerto 
taxa_de_acertos= accuracy_score(teste_y, previsoes)*100
print("taxa de acertos: %.2f%%" %(taxa_de_acertos))

