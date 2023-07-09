from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import keras
from tensorflow.keras.callbacks import TensorBoard
from scipy.stats import zscore
import cv2
import os
import numpy as np
import math


def enquadra_mao(hands, img):
    add_espaco = 20
    tam_img = 300

    if hands:
        # evita erro de tive apenas uma parte da mao na tela
        try:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgBranca = np.ones((tam_img, tam_img, 3), np.uint8) * 255
            imgBox = img[y - add_espaco: y + h + add_espaco, x - add_espaco: x + w + add_espaco]
            # cv2.imshow("imageMao", imgBox)

            imgBoxShape = imgBox.shape

            proporcao_tela = h / w

            if proporcao_tela > 1:
                k = tam_img / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(imgBox, (w_cal, tam_img))
                img_resize_shape = img_resize.shape
                w_gap = math.ceil((tam_img - w_cal) / 2)
                imgBranca[:, w_gap:w_cal + w_gap] = img_resize
            else:
                k = tam_img / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(imgBox, (tam_img, h_cal))
                img_resize_shape = img_resize.shape
                h_gap = math.ceil((tam_img - h_cal) / 2)
                imgBranca[h_gap:h_cal + h_gap, :] = img_resize

            cv2.imshow("img_tam_fixo", imgBranca)

            return x, y, w, h
        except:
            pass


def redimenciona_pontos(pontos_frame, x, y):
    for ponto in pontos_frame:
        p_x = ponto[0]
        p_y = ponto[1]

        p_x = p_x - x
        p_y = p_y - y

        ponto[0] = p_x
        ponto[1] = p_y
    return pontos_frame


def normalizar_pontos(pontos_frame):
    pontos_x = []
    pontos_y = []
    pontos_z = []
    for pontos in pontos_frame:
        pontos_x.append(pontos[0])
        pontos_y.append(pontos[1])
        pontos_z.append(pontos[2])

    # normalizer vetores
    x_normalizado = ((pontos_x - np.min(pontos_x)) / (np.max(pontos_x) - np.min(pontos_x)) * 2) - 1
    y_normalizado = ((pontos_y - np.min(pontos_y)) / (np.max(pontos_y) - np.min(pontos_y)) * 2) - 1
    z_normalizado = ((pontos_z - np.min(pontos_z)) / (np.max(pontos_z) - np.min(pontos_z)) * 2) - 1

    for i, pontos in enumerate(pontos_frame):
        pontos[0] = x_normalizado[i]
        pontos[1] = y_normalizado[i]
        pontos[2] = z_normalizado[i]

    return pontos_frame


def reshape(data):
    print(f"Shape original {np.array(data).shape}")
    x = []
    x_np = np.array(data)
    x_np_shape = x_np[0].shape
    for i, video in enumerate(x_np):
        video = video.reshape(x_np_shape[0] * x_np_shape[1], x_np_shape[2])
        x.append(video)
    print(f"Reshape {np.array(x).shape}")
    return np.asarray(x)


def model(shape, acoes):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=shape))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(acoes), activation='softmax'))

    model.summary()
    # momentum=0.9
    optmizer = keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optmizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def pegaCaminhoArquivos(pasta_principal):
    caminho_dic = {}
    lista_arquivos = []

    # Itera sobre os diretórios dentro da pasta principal
    for diretorio in os.listdir(pasta_principal):
        # Verifica se o item é um diretório
        if os.path.isdir(os.path.join(pasta_principal, diretorio)):
            # Obtém o caminho completo para o diretório
            caminho_diretorio = os.path.join(pasta_principal, diretorio)
            # Itera sobre os arquivos dentro do diretório
            for arquivo in os.listdir(caminho_diretorio):
                # Verifica se o item é um arquivo
                if os.path.isfile(os.path.join(caminho_diretorio, arquivo)):
                    # Imprime o caminho completo do arquivo
                    caminho_arquivo = os.path.join(caminho_diretorio, arquivo)
                    lista_arquivos.append(caminho_arquivo)
        caminho_dic.update({diretorio: lista_arquivos})
        lista_arquivos = []

    return caminho_dic
