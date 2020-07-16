# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 02:09:51 2020

@author: Luiz Cardoso
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:08:33 2020

@author: Luiz Cardoso
"""
#Neste código será adicionado uma parte que vai atuar na detecção de olhos também junto com a de faces
#Será feita de forma bem semelhante. Inicialmente ja se tem o haarcascade pronto para detecção de olhos
#Mais para frente será criado o próprio haarcascade para detecção de qualquer objeto

import cv2

image_path = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/Fotos/imagem3.jpg"

cascade_path_face = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_frontalface_default.xml"
cascade_path_eye = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_eye.xml"

myClf_face = cv2.CascadeClassifier(cascade_path_face)
myClf_eye = cv2.CascadeClassifier(cascade_path_eye)

img1 = cv2.imread(image_path)

grayFace = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


faceDetect = myClf_face.detectMultiScale(grayFace)

for (X1, Y1, W1, H1) in faceDetect:
    img1 = cv2.rectangle(img1, (X1, Y1), (X1 + W1, Y1 + H1), (0,255,255), 2)
    #Como o olho está dentro do quadrado da face, vamos trabalhar com o olho dentro desse FOR que desenha o quadrado da face
    #Não precisa do que ta fora do quadrado da face
    eyeLocal = img1[Y1:Y1+H1, X1:X1+W1] #definindo o quadrado da face
    grayEye = cv2.cvtColor(eyeLocal, cv2.COLOR_BGR2GRAY) #deixando cinza para jogar no modelo
    eyeDetect = myClf_eye.detectMultiScale(grayEye, scaleFactor = 1.1, minNeighbors = 2, minSize = (20,20)) #modelo apenas com a parte da face
    
    for (X2, Y2, W2, H2) in eyeDetect:
        #Passa o local do olho que foi a imagem que foi usado de ponto de partida para detectar os olhos nela
        cv2.rectangle(eyeLocal, (X2, Y2), (X2 + W2, Y2 + H2), (255, 0, 255), 2) #desenhando os retângulos
    
    

cv2.imshow("Face e Olhos Detectaos", img1)
cv2.waitKey()