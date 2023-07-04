from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
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
            #cv2.imshow("imageMao", imgBox)

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

def normalizar_pontos(pontos_frame):

    maior_x = 0
    maior_y = 0
    maior_z = 0
    for ponto in pontos_frame:
        if maior_x > ponto[0]:
            maior_x = ponto[0]
        if maior_y > ponto[1]:
            maior_y = ponto[1]
        if maior_z > ponto[2]:
            maior_z = ponto[2]

    for ponto in pontos_frame:
        ponto[0] = ponto[0] / maior_x
        ponto[1] = ponto[1] / maior_y
        ponto[2] = ponto[2] / maior_z

    return pontos_frame


def model(shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.summary()

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

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
