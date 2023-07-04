import cv2
import time
import pickle
from funcoes import enquadra_mao, normalizar_pontos

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# quantidade de frame que ira salva
qnt_frame = 30

cont_frame = 0

gravando = False

pontos_mao = []

qnt_salvos = 0

while True:
    suc, img = cap.read()
    hands, img = detector.findHands(img)
    enquadra_mao(hands, img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # inicia a gravação
    if key == ord('s'):
        gravando = not gravando
        print(f"Gravando = {gravando}")

    if gravando:

        if cont_frame >= qnt_frame:
            gravando = False
            with open(f'letras/A/{time.time()}.pickle', 'wb') as arquivo:
                pickle.dump(pontos_mao, arquivo)
                print(pontos_mao)
            arquivo.close()

            qnt_salvos += 1
            cont_frame = 0
            pontos_mao = []

            print(f"Gravando = {gravando}")
            print(f"Quantidade salvos = {qnt_salvos}")

        if hands:
            x, y, h, w = enquadra_mao(hands, img)
            hand = hands[0]
            pontos_frame = hand['lmList']

            for ponto in pontos_frame:

                p_x = ponto[0]
                p_y = ponto[1]

                p_x = p_x - x
                p_y = p_y - y

                ponto[0] = p_x
                ponto[1] = p_y

            pontos_frame = normalizar_pontos(pontos_frame)

            pontos_mao.append(pontos_frame)
            cont_frame += 1
