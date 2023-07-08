import numpy as np

from funcoes import pegaCaminhoArquivos, model
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


x_np = np.array(data_treino)
x = []

print(f"shape original = {x_np.shape}")

x_np_shape = x_np[0].shape
for i, video in enumerate(x_np):
    video = video.reshape(x_np_shape[0]*x_np_shape[1], x_np_shape[2])
    x.append(video)

x = np.array(x)
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

print("------------------------------------------------------")
print(y)
print(x[0])

model = model(x_shape[1:], acoes)
#1800
model.fit(x, y, epochs=2000, batch_size=10)

