import pandas as pd
uri="https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"
dados= pd.read_csv(uri)
print(dados.head())
mapa = {
    "movieId": "filme_id",
    "title" : "titulo",
    "genres" :"generos"
}
dados= dados.rename(columns=mapa)
print(dados.head())
generos = dados.generos.str.get_dummies() ## pegando todos os generos dos filmes e tornando em colunas
dados_filmes = pd.concat([dados,generos], axis=1)
print(dados_filmes.head())
####### escalando dados 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
generos_escalados=scaler.fit_transform(generos) 
print(generos_escalados)