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

model_dict = pickle.load(open('model/model_randomForest_A-Z.p', 'rb'))
classificador = model_dict['model']



labels = {-1:'',0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',7:'H', 8:'I', 9:'J',10:'K',11: 'L', 12:'M', 13:'N', 14:'O',
          15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V',22:'W', 23:'X', 24:'Y', 25:'Z'}

# quantidade de frame que ira salva
qnt_frame = 30
cont_frame = 0
gravando = False
cont_no_mao = 0
pontos_mao = []
qnt_salvos = 0
add_espaco = 20
letra = ""


def getIndex(index):
    cont = 0
    for i ,ind in enumerate(index[0]):
        cont+=1
        if ind == 1:
            return i
    if cont == len(index[0]):
        return -1

while True:

    suc, img = cap.read()
    imgSaida = img.copy()
    hands, img = detector.findHands(img)
    enquadra_mao(hands, img)

    if hands:
        try:
            x, y, h, w = enquadra_mao(hands, img)
            hand = hands[0]
            pontos_frame = hand['lmList']
        except:
            pass
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
            
            nsamples, nx, ny = pontos_reshape.shape
            d2_train_dataset = pontos_reshape.reshape((nsamples,nx*ny))
            
            #print(f"Reshape do reshape {d2_train_dataset.shape}")
            
            index = classificador.predict(np.array(d2_train_dataset))
            #print(index)
            letra = labels[getIndex(index)]
        cv2.rectangle(imgSaida, (x-add_espaco, y - add_espaco - 50),
                    (x - add_espaco+90, y - add_espaco - 50 + 50), (255, 0 , 255), cv2.FILLED)
        cv2.putText(imgSaida, letra, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgSaida, (x - add_espaco, y - add_espaco),
                    (x + h + add_espaco , y + w + add_espaco ), (255, 0 , 255), 4)
    
    
        
    cv2.imshow("Image", imgSaida)
    cv2.waitKey(1)
