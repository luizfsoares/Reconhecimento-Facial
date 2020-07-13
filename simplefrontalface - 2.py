# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:08:33 2020

@author: Luiz Cardoso
"""
#Basicamente aqui será trabalhado o mesmo modelo que foi feito anteriormente porém alterando alguns dos parâmetros de detectMultiScale
#para melhorar a performace, visto que o modelo mais simples é bem falho e precisa melhorar.

import cv2

image_path = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/Fotos/imagem6.jpg"

cascade_path = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_frontalface_default.xml"

myClf = cv2.CascadeClassifier(cascade_path)

img1 = cv2.imread(image_path)

grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


#Trabalhando outros parâmetros para melhorar o modelo
#scaleFactor - diminui a escala. Especifica o quanto do tamanho da imagem será reduzida
#minNeighboors - retira uma parte dos pontos vizinhos que são candidatos a ser um ponto de face
#minSize - indica o tamanho minimo do retangulo da face detectada. Faces de tamanho menor que esse são ignoradas

faceDetect = myClf.detectMultiScale(grayImg1, scaleFactor = 1.07, minNeighbors =  4, minSize = (30,30))

for (X, Y, W, H) in faceDetect:
    cv2.rectangle(img1, (X, Y), (X + W, Y + H), (0,255,255), 2)
    
cv2.imshow("Face Detectada", img1)
cv2.waitKey()

