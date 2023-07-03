import cv2
import time

from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands =1)

#quantidade de frame que ira salva
qnt_frame = 30

cont_frame = 0

gravando = False

pontos_mao = []

qnt_salvos = 0

while True:
    suc,img = cap.read()
    hands, img = detector.findHands(img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    #inicia a gravação
    if key == ord('s'):        
        gravando = not gravando
        print(f"Gravando = {gravando}")
       
    if gravando:
        
        if cont_frame >= qnt_frame:
            gravando = False
            arquivo = open(f'letras/C/{time.time()}.txt', 'a')
            arquivo.write(str(pontos_mao))
            arquivo.close()
            
            qnt_salvos += 1
            cont_frame = 0
            pontos_mao = []
            
            print(f"Gravando = {gravando}")
            print(f"Quantidade salvos = {qnt_salvos}")
         
        if hands:
            hand = hands[0]
            pontos_mao += hand['lmList']
            cont_frame += 1
                