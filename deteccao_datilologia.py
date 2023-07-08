import cv2
import time
import pickle
from cvzone.ClassificationModule import Classifier
from funcoes import enquadra_mao, normalizar_pontos, redimenciona_pontos, reshape

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classificador = Classifier("model/dd.h5", "model/labels.txt")

labels = ['A', 'B', 'C']

# quantidade de frame que ira salva
qnt_frame = 30
cont_frame = 0
gravando = False
pontos_mao = []
qnt_salvos = 0

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

            pontos_reshape = reshape(pontos_frame_normalizado)
            predict, index = classificador.getPrediction(pontos_mao_normalizados)
            print(predict, index)

            cv2.putText(imgSaida, labels[index], (x, y - 20), cv2.FRONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Image", imgSaida)
    cv2.waitKey(1)
