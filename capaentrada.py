import cv2 as cv
import os
import imutils

modelo = 'FotosElon'
ruta1 = '/Users/davidlopez/PycharmProjects/Curso_deteccion_objetos/reconocimiento_contornos/reconocimientofacial1'
rutacompleta = ruta1 + '/' + modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

ruidos = cv.CascadeClassifier('/Users/davidlopez/PycharmProjects/Curso_deteccion_objetos/reconocimiento_contornos/entrenamientosOpenCVruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
#0 es una camara externa
#1 es la camara del pc
#camara= cv.VideoCapture(1)
camara= cv.VideoCapture('ElonMusk.mp4')
id = 0

while True:
    respuesta, captura = camara.read()
    if respuesta==False:break
    captura = imutils.resize(captura, width = 640)
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()
    # cambiando el valor de 1.3 y 5  varias las opciones para encontrar una cara
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    for(x,y,e1,e2) in cara:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0), 2)
        rostrocapturado = idcaptura[y:y+e1, x:x+e2]
        rostrocapturado= cv.resize(rostrocapturado, (160, 160), interpolation= cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado)
        id=id+1

    cv.imshow('resultado rostro', captura)

    if id==351:
        break

camara.release()
cv.destroyAllWindows()