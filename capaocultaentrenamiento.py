import cv2 as cv
import os
import numpy as np
from time import time


dataRuta = '/Users/davidlopez/PycharmProjects/Curso_deteccion_objetos/reconocimiento_contornos/reconocimientofacial1/Data'
listaData = os.listdir(dataRuta)
#print(listaData)
ids=[]
rostrosData = []
id = 0
tiempoInicial = time()

for fila in listaData:
    rutacompleta = dataRuta+ '/' + fila
    print('Iniciando lectura...')
    for archivo in os.listdir(rutacompleta):

        print('Imagenes: ', fila + '/' + archivo)
        ids.append(id)
        # el valor 0 lo convierte a escala de grises
        rostrosData.append(cv.imread(rutacompleta+'/'+archivo, 0))

    id = id+1
    tiempoFinalLectura = time()
    tiempoTotalLectura = tiempoFinalLectura - tiempoInicial
    print('Tiempo total lectura: ', tiempoTotalLectura)

entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
print('Iniciando el entreamiento...espere')
entrenamientoEigenFaceRecognizer.train(rostrosData, np.array(ids))
TiempofinalEntrenamiento = time()
TiempoTotalEntrenamiento = TiempofinalEntrenamiento - tiempoTotalLectura
print('Tiempo entrenamiento total: ', TiempoTotalEntrenamiento)
entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento concluido')