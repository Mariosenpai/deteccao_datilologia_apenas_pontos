import numpy as np

from funcoes import pegaCaminhoArquivos, model, reshape
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
import pickle

acoes = ['A', 'B', 'C']

pasta_principal = 'letras'

caminho_dic = pegaCaminhoArquivos(pasta_principal)

data_pontos = []
data_treino = []
labels_treino = []

label_map = {label:num for num, label in enumerate(acoes)}

for acao in acoes:
    a = caminho_dic[acao]
    for arquivo in a:
        with open(arquivo, 'rb') as arquivo_aberto:
            conteudo = pickle.load(arquivo_aberto)
            data_treino.append(conteudo)
            #data_pontos += data_treino
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


model = model(x_shape[1:], acoes)
#300
h = model.fit(x, y, epochs=500, batch_size=10)

model.save("model/dd.h5")
print("Modelo salvo!!")
