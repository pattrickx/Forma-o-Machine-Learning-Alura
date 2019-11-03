from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#####DADOS
porco1=[0, 1, 0]
porco2=[0, 1, 1]
porco3=[1, 1, 0]

cachorro1=[0, 1, 0]
cachorro2=[0, 1, 1]
cachorro3=[1, 1, 0]

treino_x =[porco1,porco2,porco3,cachorro1,cachorro2,cachorro3]#array de dados para treino
treino_y= [1,1,1,0,0,0]
#######TREINOS
model =LinearSVC()
model.fit(treino_x,treino_y)

####### TESTES
m1=[1,1,1]
m2=[1,1,0]
m3=[0,1,1]
testes_x=[m1,m2,m3] #array de dados para teste
testes_y=[0,1,1]
previsoes=model.predict(testes_x)
# corretos=(previsoes==testes_classes).sum()
# total = len(testes)
# taxa_de_acertos=corretos/total # 3 linha de cima da no mesmo que a de baixo 
taxa_de_acertos= accuracy_score(testes_y, previsoes)
print("taxa de acerto: %.2f"%(taxa_de_acertos*100))
