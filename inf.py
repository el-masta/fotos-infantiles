#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Creación de plantillas para foto infantil'''
__license__ = "GPL"
__version__ = "1.0.0"
__email__   = "sirmcoil73@gmail.com"
__author__  = "Mario Rodríguez"

import math
import cv2 as cv
import numpy as np


#Convertir imagen a RGB
def toRGB(img):
	'''Convierte imagen de BGR a RGB'''
	return cv.cvtColor(img, cv.COLOR_BGR2RGB)

#Detección de ojos para encontrar ángulo
def encontrar_angulo(img):
	'''Encuentra ojos en la imagen'''
	cascade = cv.CascadeClassifier('haarcascade_eye.xml')
	gris=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	ojos = cascade.detectMultiScale(gris, 2.1, 13)
	punto1=(ojos[0][0]+ojos[0][2]//2,ojos[0][1]+ojos[0][3]//2)
	punto2=(ojos[1][0]+ojos[1][2]//2,ojos[1][1]+ojos[1][3]//2)
	angulo=math.degrees(math.atan((punto2[1]-punto1[1])/(punto2[0]-punto1[0])))
	return angulo

#Rota la imagen de acuerdo al ángulo
def rotar(imagen,angulo):
	'''Rota la imagen con respecto a su centro'''
	alto,ancho,nada = imagen.shape
	matriz=cv.getRotationMatrix2D((ancho/2,alto/2),angulo,1)
	img=cv.warpAffine(imagen,matriz,(ancho,alto))
	return img
	
#Encuentra el rostro y recorta la imagen a tamaño infantil a 240dpi 
def cortar(img):
	'''Encuentra el rostro'''
	cascade = cv.CascadeClassifier('lbpcascade_frontalface.xml')
	gris=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = cascade.detectMultiScale(gris, 1.1, 5)
	x,y,w,h=faces[0][0],faces[0][1],faces[0][2],faces[0][3]
	centro=(x+w//2,y+h//2)
	esc=360/h
	esqsi=(centro[0]-int(300/esc),centro[1]-int(400/esc))
	esqid=(esqsi[0]+int(600/esc),esqsi[1]+int(720/esc))
	recorte=img[esqsi[1]:esqid[1],esqsi[0]:esqid[0]]
	redim=cv.resize(recorte,(600,720),interpolation=cv.INTER_AREA)
	return redim

#Hace un pequeño ajuste de nieveles de luz en la imagen
def niveles(img):
	'''Ajuste de niveles'''
	lab=cv.cvtColor(img, cv.COLOR_BGR2LAB)
	l,a,b=cv.split(lab)
	clahe=cv.createCLAHE(clipLimit=0.5,tileGridSize=(4,4))
	l=clahe.apply(l)
	lab=cv.merge((l,a,b))
	return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

def acomodarFotos(foto):
   # Crea un fondo gris claro
   fondo = toRGB(np.zeros((960,1440,3), np.uint8))
   fondo[:]=(240,240,240)
   # Crea una copia de la foto a tamaño infantil
   foto = cv.resize(foto, (236,283), interpolation = cv.INTER_CUBIC)
   # Acomoda la rejilla de 3 x 5 fotos
   for i in range(3):
      distY=(i*(28+foto.shape[0]))+28
      for j in range(5):
         distX=(j*(43+foto.shape[1]))+43
         fondo[distY:distY+foto.shape[0], distX:distX+foto.shape[1]] = foto
   return fondo

#cargar imagen
original=cv.imread('original.jpg')
#encuentra angulo de rotación
angulo=encontrar_angulo(original)
#rotar imagen
enderezada=rotar(original, angulo)
#Encuentra el rostro y recorta
recortada=cortar(enderezada)
#ajuste de niveles
ajustada=niveles(recortada) 
# Crea fondo y acomoda fotos
plantilla=acomodarFotos(ajustada)
cv.imwrite('plantilla.jpg',plantilla)