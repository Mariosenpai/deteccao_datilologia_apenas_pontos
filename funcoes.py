from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os




def enquadra_mao(hands):
    add_espaco = 20
    if hands:
        #evita erro de tive apenas uma parte da mao na tela
        try:
            hand = hands[0]
            x,y,w,h = hand['bbox']
            imgCrop = img[y - add_espaco: y+h +add_espaco, x-add_espaco: x+w + add_espaco]
            cv2.imshow("imageMao", imgCrop)
        except:
            pass 


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
