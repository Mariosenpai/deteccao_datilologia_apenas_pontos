import cv2
import time
import pickle
from cvzone.ClassificationModule import Classifier
from funcoes import enquadra_mao, normalizar_pontos, redimenciona_pontos, reshape
import numpy as np
from tensorflow.keras.models import load_model

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#classificador = Classifier("model/dd.h5", "model/labels.txt")
classificador = load_model('model/dd.h5')

classificador.summary()



labels = {0:'A', 1:'B', 2:'C'}

# quantidade de frame que ira salva
qnt_frame = 30
cont_frame = 0
gravando = False
pontos_mao = []
qnt_salvos = 0
add_espaco = 20
letra = ""

while True:

    suc, img = cap.read()
    imgSaida = img.copy()
    hands, img = detector.findHands(img)
    enquadra_mao(hands, img)

    if hands:
        x, y, h, w = enquadra_mao(hands, img)
        hand = hands[0]
        pontos_frame = hand['lmList']

        pontos_mao.append(redimenciona_pontos(pontos_frame, x, y))
        cont_frame += 1

        if cont_frame >= qnt_frame:
            cont_frame = 0
            pontos_mao_normalizados = []
            for pontos in pontos_mao:
                pontos_frame_normalizado = normalizar_pontos(pontos_frame)
                pontos_mao_normalizados.append(pontos_frame_normalizado)
            pontos_mao = []
            
            pontos_reshape = reshape([pontos_mao_normalizados])
            
            index = np.argmax(classificador.predict(pontos_reshape), axis=1)
            print(index[0])
            letra = labels[index[0]]
        cv2.rectangle(imgSaida, (x-add_espaco, y - add_espaco - 50),
                    (x - add_espaco+90, y - add_espaco - 50 + 50), (255, 0 , 255), cv2.FILLED)
        cv2.putText(imgSaida, letra, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgSaida, (x - add_espaco, y - add_espaco),
                    (x + h + add_espaco , y + w + add_espaco ), (255, 0 , 255), 4)
        
    cv2.imshow("Image", imgSaida)
    cv2.waitKey(1)
