# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:54:49 2020

@author: StefanutVlad
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from PIL import Image

from scipy.signal import convolve2d

#path = r"Z:\an4curent\sem1\pni\proiect\ZebraWithAttachedShadow.jpg"
#path = r"Z:\an4curent\sem1\pni\laboratoare\lab2\Lab02_GST\ImgsTstLab02\floare_sa_gr_fc.jpg"
#path = r"Z:\an4curent\sem1\pni\laboratoare\lab2\Lab02_GST\ImgsTstLab02\imgSalvata2.jpg"
#path = r"Z:\an4curent\sem1\pni\laboratoare\lab2\Lab02_GST\ImgsTstLab02\smiley.jpg"
path = r"Z:\an4curent\sem1\pni\laboratoare\lab2\Lab02_GST\ImgsTstLab02\5.2.09.jpg"
img = cv2.imread(path)
img2 = Image.open(path).convert("L")



#functie afisare
def functieAfisare(img,textImg,textHist):
    
    #dimensiune panou
    fig = plt.figure(figsize=(15,7))
    
    #afisare imagine
    im = fig.add_subplot(121)
    im.imshow(img,cmap="gray")

    im.set_xlabel('Lungime[px]')
    im.set_ylabel('Inaltime[px]')
    plt.title(textImg)

    #afisare histograma
    im = fig.add_subplot(122)
    im.hist(img.ravel(),256,[0,256])


    im.set_xlabel('Nivel de gri')
    im.set_ylabel('Frecventa')
    plt.title(textHist)
    plt.show()
    
    return 1

     
#functie generare zgomot
def functieZgomot(img):
    
    #probabilitatea de aparitie
    prob=0.03
    
    #matricea imaginii cu valori de 0 si format uint8
    output = np.zeros(img.shape,np.uint8)
    
    cond = 1 - prob 
    
    #parcurgem matricea
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random() # rdn=[0;1]
            if rdn < prob:
                output[i][j] = 0   #pepper
            elif rdn > cond:
                output[i][j] = 255 #salt
            else:
                output[i][j] = img[i][j]
    return output



#functie filtrare mediana
def filtrareMediana(img, dimensFiltru):
     
    #nr de pixeli care formeaza un contur neagru al ferestrei/directie
    blackPx=dimensFiltru//2 #math.ceil()
    
    imgFiltrata=[]
    imgFiltrata= np.zeros((len(img),len(img[0])))
    
    #parcurgem imaginea originala
    for i in range(len(img)):
       for j in range(len(img[0])):    
           val=[]
           #verificam daca pixelul se afla in contur
           #parcurgem fereastra 
           for z in range(dimensFiltru):
               if i + z - blackPx >= 0 or i + z - blackPx <= len(img) - 1:
                  
                   #Daca fereastra se potriveste
                   for k in range(dimensFiltru):
                         #adaugam pixelii in lista
                         val.append(img[i-1 + z - blackPx][j-1 + k - blackPx])
              
               #Gasirea valorii mediane:
                   
               #ordonam pixelii crescator
               val.sort()
              
                #inlocuirea pixelului de interes cu valoarea din mijlocul listei
               imgFiltrata[i][j]= val[len(val) // 2]
    return imgFiltrata




def Kuwahara(img, dimensFereastra):
    #citim imaginea in valori float64 pt calcule mai precise
    imgCitita = img.astype(np.float64)
    
    #verificam dimensiunea ferestrei
    
    if dimensFereastra%2 ==0:
        raise Exception ("Dimensiune fereastra para. Dimensiune ceruta: impara")
        
    #Construirea regiunilor
    
    #linia de inceput a regiunii
    tempRow = np.hstack((np.ones((1,(dimensFereastra-1)//2+1)),np.zeros((1,(dimensFereastra-1)//2)))) # 1 1 1 0 0
    tempRow2 = np.hstack((np.ones((1,(dimensFereastra-1)//2)),np.zeros((1,(dimensFereastra-1)//2+1)))) # 1 1 0 0 0
    
    #padding
    tempPad = np.zeros((1,dimensFereastra)) # 0 0 0 0 0
    
    #Copiere linie
    tempKernel = np.tile(tempRow, ((dimensFereastra-1)//2,1)) #tmpavgkerrow de 3 ori
    tempKernel= np.vstack((tempKernel,tempRow2[tempRow2[:,0]<2]))
    
    #Adaugam padding
    tempKernel = np.vstack((tempKernel, np.tile(tempPad, ((dimensFereastra-1)//2,1)))) #tile => 2x pad .. => verticala: 3xtmpavgkerrow+2xpad
    
    #media fiecarui element din regiune
    tempKernel = tempKernel/np.sum(tempKernel) #np.sum(tempKernel) = nr elemente regiune
    
    # tempKernel - regiunea N-W
    
    # Construim fereastra cu cele 4 regiuni
    avgKernel = np.empty((4,dimensFereastra,dimensFereastra)) # array gol pt cele 4 regiuni
    
    #Regiuni
    avgKernel[0] = tempKernel			# regiunea N-W (a) 
    avgKernel[1] = np.fliplr(tempKernel)	# regiunea N-E (b)
    avgKernel[2] = np.flipud(tempKernel)	# regiunea S-W (c)
    avgKernel[3] = np.fliplr(avgKernel[2])	# Regiunea S-E (d)
    
    # Initializare patratul imaginii pixel cu pixel
    
    
    # Initializare array-uri pentru media regiunilor si deviatii
    medieRegiuni = np.zeros([4, imgCitita.shape[0],imgCitita.shape[1]])
    deviatiiStandard = medieRegiuni.copy()
    
    # Calcularea mediilor si deviatiilor pentru regiuni
    for k in range(4):
       medieRegiuni[k] = convolve2d(imgCitita, avgKernel[k],mode='same') 	    # media regiuniilor ; same=acelasi tip de date ca argument1
       squaredImg = (imgCitita-medieRegiuni[k])**2
       deviatiiStandard[k] = convolve2d(squaredImg, avgKernel[k], mode='same')  # media patratelor regiuniilor
    
    index = np.argmin(deviatiiStandard,0) # gasim indexul ferestrei cu deviatia minima
    
    # Construirea imaginii filtrate
    filtered = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            #formarea imaginii cu noile nivele de gri pt a inlocui valoarea pixelului din img originala
            filtered[row,col] = medieRegiuni[index[row,col], row,col]           
    return filtered.astype(np.uint8)



functieAfisare(img,'Imaginea originala', 'Histograma imaginii originale') 

#aplicare zgomot
imgZgomot=functieZgomot(np.array(img2))
functieAfisare(imgZgomot,'Imaginea afectata de zgomot', 'Histograma imaginii afectate de zgomot')

#filtrare mediana
imagineFiltrata=filtrareMediana(imgZgomot,3)
functieAfisare(imagineFiltrata,'Imagine filtrata Median','Histograma imaginii filtrate Median')

#filtrare kuwahara
imagineKuwahara=Kuwahara(imgZgomot,5)
functieAfisare(imagineKuwahara,'Imagine filtrata Kuwahara','Histograma imaginii filtrate Kuwahara')