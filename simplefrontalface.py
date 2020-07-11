# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 04:15:07 2020

@author: Luiz Cardoso
"""

#import do OpenCv
import cv2

#endereço da imagem utilizada
image_path = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/Fotos/imagem3.jpg"

#endereço do arquivo haarcascade utilizado (ja treinado para faces frontais)/ modelo utilizado
cascade_path = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_frontalface_default.xml"

#Carrega o algoritmo cascade e cria um classificador
myClf = cv2.CascadeClassifier(cascade_path)

#carrega a imagem
img1 = cv2.imread(image_path)

#É bom sempre converter pra escala de cinza, pois aumenta a eficiencia do modelo utilizado
#Usa a função cvtColor passando a imagem normal (colorida) e o tipo de transformação que quer, no caso gray scale
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#chama a funcao do haarcascade para detectar da imagem CINZA
faceDetect = myClf.detectMultiScale(grayImg1)

print(faceDetect)

#loop para percorrer todos os pixels na imagem encontrada
#Dentro do loop vai ser desenhado um retângulo com os pontos encontrados em faceDetect
for (X, Y, W, H) in faceDetect:
    cv2.rectangle(img1, (X, Y), (X + W, Y + H), (0, 0, 255), 2)

cv2.imshow("Foto Utilizada ", img1)
#cv2.imshow("Faces Reconhecidas ", faceDetect)
cv2.waitKey()