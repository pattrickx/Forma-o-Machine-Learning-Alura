import pandas as pd
uri="https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"
dados= pd.read_csv(uri)
#print(dados.head())
mapa = {
    "movieId": "filme_id",
    "title" : "titulo",
    "genres" :"generos"
}
dados= dados.rename(columns=mapa)
#print(dados.head())
generos = dados.generos.str.get_dummies() ## pegando todos os generos dos filmes e tornando em colunas
dados_filmes = pd.concat([dados,generos], axis=1)
#print(dados_filmes.head())
####### escalando dados 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
generos_escalados=scaler.fit_transform(generos) 
#print(generos_escalados)
#########################################   AULA 2
from sklearn.cluster import KMeans ### lib de clusterisação
modelo =KMeans(n_clusters=3) ### define numero de clusters
modelo.fit(generos_escalados) ## treina com os dados e gera grupos
# print('Grupos{}'.format(modelo.labels_)) ## mostra gupos
# print(generos.columns)
# print(modelo.cluster_centers_)
#################### aula 3
import matplotlib.pyplot as plt

grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)

# print(grupos)
# grupos.plot.bar()
grupos.transpose().plot.bar(subplots=True,sharex=False, rot=0) ## transpose() faz a matriz transposta 
plt.show()
grupo = 0
filtro=(modelo.labels_==grupo)
#print(dados_filmes[filtro].sample(10)) #mostra parte dos dados de um dos grupos para verificação
############### ajuste de dimençoes
'''
from sklearn.manifold import TSNE

tsne=TSNE()
visualizacao = tsne.fit_transform(generos_escalados)
print(visualizacao)

import seaborn as sns
sns.set(rc={'figure.figsize':(13,13)})
sns.scatterplot(x=visualizacao[:,0],y=visualizacao[:,1] , hue= modelo.labels_, palette=sns.color_palette('Set1',3))
plt.show()
'''
# plt.scatter(visualizacao[:,0], visualizacao[:,1],c=modelo.labels_ ,s=1) #metodo alternativo
# plt.show()

############### aula 4
######## testando modelo com 20 grupos
modelo =KMeans(n_clusters=20) ### define numero de clusters
modelo.fit(generos_escalados) ## treina com os dados e gera grupos

grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)

grupos.transpose().plot.bar(subplots=True,sharex=False, rot=0) ## transpose() faz a matriz transposta 
plt.show()
grupo = 0
filtro=(modelo.labels_==grupo)
print(dados_filmes[filtro].sample(10)) #mostra parte dos dados de um dos grupos para verificação
def kmeans(numeros_de_clusters,generos):
    modelo=KMeans(n_clusters=numeros_de_clusters)
    modelo.fit(generos)
    return [numeros_de_clusters,modelo.inertia_]
print(kmeans(20,generos_escalados))
resultado = [kmeans(numero_grupos,generos_escalados) for numero_grupos in range(1,41)]
# metodo do cutuvelo 
########### procurar ponto de curva
resultado= pd.DataFrame(resultado, columns=['grupos', 'inertia'])
resultado.inertia.plot(xticks=resultado.grupos)
plt.show()
############################### teste com quantidade de grupos "ideais"
modelo =KMeans(n_clusters=17) ### define numero de clusters
modelo.fit(generos_escalados) ## treina com os dados e gera grupos

grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)

grupos.transpose().plot.bar(subplots=True,sharex=False, rot=0) ## transpose() faz a matriz transposta 
plt.show()
grupo = 0
filtro=(modelo.labels_==grupo)
print(dados_filmes[filtro].sample(10)) #mostra parte dos dados de um dos grupos para verificação