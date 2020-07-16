# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 01:17:51 2020

@author: Luiz Cardoso
"""

import cv2

#cascade para identificar face
cascade_path_face = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_frontalface_default.xml"
cascade_path_eye = "C:/Users/preda/.spyder-py3/Reconhecimento Facial/HaarCascade prontos/haarcascade_eye.xml"

#função de captura de video passando o parâmetro 0 que representa a câmera integrada, ou seja, a webcam do notebook
myCamera = cv2.VideoCapture(0)

myClf_face = cv2.CascadeClassifier(cascade_path_face)
myClf_eye = cv2.CascadeClassifier(cascade_path_eye)


while True:
    
    #recebe a imagem da camera (PRECISA DE DUAS VARIAVEIS para receber a função)
    camera, frame = myCamera.read()
    
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markedImg = myClf_face.detectMultiScale(grayImg, scaleFactor = 1.1, minNeighbors = 10, minSize = (100,100))
    
    #loop de marcação da face
    for (X1, Y1, W1, H1) in markedImg:
        
        frame = cv2.rectangle(frame, (X1, Y1), (X1+W1, Y1+H1), (255, 0, 255), 2)
        #joga em uma variavel, pois dentro do quadrado marcado da face será feita o reconhecimento dos olhos
        eyeLocal = frame[Y1:Y1+H1, X1:X1+W1]
        grayEye = cv2.cvtColor(eyeLocal, cv2.COLOR_BGR2GRAY)
        
        #executando a função de detecção do modelo com alguns parametros para mlhorar um pouco a eficiencia
        eyeDetect = myClf_eye.detectMultiScale(grayEye, scaleFactor = 1.1, minNeighbors = 3, minSize = (60,60))
        
        #loop de marcação dos olhos
        for (X2, Y2, W2, H2) in eyeDetect:
            
            #Desenha o retangulo usando o x,y,w,h do eyelocal, ou seja, em relação ao que ja está DENTRO do quadrado da face e nao da imagem toda
            cv2.rectangle(eyeLocal, (X2, Y2), (X2+W2, Y2+H2), (0, 255, 255), 2)
        
        
    
    #mostra a imagem da camera
    cv2.imshow("Imagem da webcam", frame)
    
    #condição de parada para desligar a webcam
    if cv2.waitKey(1) == ord('x'):
        break
    
#limpa os dados na memoria
myCamera.release()
#fecha a janela
cv2.destroyAllWindows()