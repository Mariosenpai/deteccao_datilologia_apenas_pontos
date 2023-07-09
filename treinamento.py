import math

import numpy as np

from funcoes import pegaCaminhoArquivos, model, reshape
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

acoes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O','P' , 'Q','R', 'S', 'T', 'U', 'V','W' , 'X', 'Y', 'Z' ]

pasta_principal = 'letras'

caminho_dic = pegaCaminhoArquivos(pasta_principal)

data_pontos = []
data_treino = []
labels_treino = []

label_map = {label: num for num, label in enumerate(acoes)}

for acao in acoes:
    a = caminho_dic[acao]
    for arquivo in a:
        with open(arquivo, 'rb') as arquivo_aberto:
            conteudo = pickle.load(arquivo_aberto)
            data_treino.append(conteudo)
            # data_pontos += data_treino
            labels_treino.append(label_map[acao])

print(f"shape original = {np.array(data_pontos).shape}")

x = reshape(data_treino)
y = to_categorical(labels_treino).astype(int)

x_shape = x.shape

qnt_video = len(data_treino)
qnt_frame = len(data_treino[0])
pontos_por_frames = len(data_treino[0][0])

print(f"reshape = {x_shape}")
print(f"Shape label = {y.shape}")
print(f"Quantidade de videos = {qnt_video}")
print(f"Quantidade de frames por video = {qnt_frame}")
print(f"Quantidade de pontos por frame = {pontos_por_frames}")


#print(f'treino = {len(x_treino)}\nValidacao = {len(x_teste)}')

model = RandomForestClassifier()
epocas = 500
batch_size = 10
steps = math.ceil(len(x/batch_size))

nsamples, nx, ny = x.shape
d2_train_dataset = x.reshape((nsamples,nx*ny))

while True:
    base_teste = 0.125
    x_treino, x_teste, y_treino, y_teste = train_test_split(d2_train_dataset, y,test_size=base_teste)
    print(f'treino = {len(x_treino)}\nValidacao = {len(x_teste)}')

    model.fit(x_treino, y_treino)

    y_predict = model.predict(x_teste)

    score = accuracy_score(y_predict, y_teste)
    print('{}% de acertos na validação!'.format(score * 100))
    if score > 0.95: break
    


f = open('model/model_randomForest_A-Z.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

#model.save("model/dd_1.h5")
print("Modelo salvo!!")
