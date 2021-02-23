# ImageProcessing---Kuwahara_Median-filtering


Median filtering and Kuwahara filter for image noise reduction

• Developer: Stefănuț Vlad Antonio

• Abstract
  .
	The project consists of analizing 2 filters: Median filter and Kuwahara filter in order to see the advantages and disadvantages of these 2 methods. Spatial filters are used in 
  ordar to reduce salt & pepper noise and they can be utilized in telecomunications and data acquisition systems.
	IDE Spyder will be used for the development of this project, using programming language Python. Spyder includes support for interactive instruments for data inspection and incorporates code quality instruments. It can be used trough many platforms via Anaconda.
	The project resumes to introducing a image, generating a random salt&pepper noise which will be applied to the initial image and filter the image using the Median filtering algorithm and the Kuwahara filtering algorithm in order to get rid of the noise applied to the initial image. The luminance histogram will also be displayed for each image.
	Entry data:
	Grayscale image;
	Different masks(3x3, 5x5, 7x7, etc.);
	Mask’s generating parameter;
	Output data:
	Initial image and its histrogram;
	The filtered image and its histogram.

• Salt & pepper noise

	- It causes pixel dispersion in an image from the image in white (255 - salt) and black (0 – pepper) values.
	- It deteriorates the image during its saving or transmission process.
	- The noise can be caused by instant varations of the image signal

• Median filter

	- Nonlinear filter;
	- Used by edge detection algorithms
	- Useful for noise reduction but inefficient for dense noises.
	- 
• Kuwahara filter

	- Smoothing nonlinear filter used in image processing for noise reduction; 
	- Improved Median filter because the image details are preserved better;
	
Median filtering – algorithm
  
	Choosing of an uneven filtering window (3x3, 5x5, etc);
	Scanning the image with the chosen window.
	Ascending/descending sorting of the window values.
	Swapping the noise affected pixel with the value of the pixel from the middle of the sorted array.
v(m,n)=median{y(m-k,n-l) },cu(k,l)∈W 

![image](https://user-images.githubusercontent.com/53474954/108867991-6fa29100-75fe-11eb-9e09-cf9c848e8eb9.png)


 
Kuwahara filtering – algorithm
  
	Choosing of an uneven filtering window (3x3, 5x5, etc);
	Scanning the image with the chosen window.
	Dividing the window in 4 equal regions.
	Calculus of median luminosity intensity and variation for each region.
	Calculus of median value of each region with the lowest variation ( output value for the central pixel). 
	Swapping the pixel value from the initial image

![image](https://user-images.githubusercontent.com/53474954/108868211-a678a700-75fe-11eb-81ac-19eb1ef760fb.png)
 		
![image](https://user-images.githubusercontent.com/53474954/108868238-aed0e200-75fe-11eb-83d1-caddfc546413.png)

Spyder program functions
	
- Generate and display the histogram of an image;
- Generate the random salt&pepper noise;
- Median filtering;
- Kuwahara filtering;
- Display function





Generate and display the histogram of an image function:

	#Display function  
	def functieAfisare(img,textImg,textHist):  
	      
	    #Pannel dimension  
	    fig = plt.figure(figsize=(15,7))  
	 
	    #Display image  
	    im = fig.add_subplot(121)  
	    im.imshow(img,cmap="gray")  
	  
	    im.set_xlabel('Lungime[px]')  
	    im.set_ylabel('Inaltime[px]')  
	    plt.title(textImg)  
	  
	    #Display hisogram  
	    im = fig.add_subplot(122)  
	    im.hist(img.ravel(),256,[0,256])  
	  
	  
	    im.set_xlabel('Nivel de gri')  
	    im.set_ylabel('Frecventa')  
	    plt.title(textHist)  
	    plt.show()  
	      
	    return 1

Generate the random salt&pepper noise	

	#Salt&pepper noise generation function
	def functieZgomot(img):  
	   
	    #probability  
	    prob=0.03  
	      
	    #image matrix value   
	    output = np.zeros(img.shape,np.uint8)  
	      
	    cond = 1 - prob   
	      
	    #matrix crossing  
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

Median filtering function

	#Median filtering function  
	def filtrareMediana(img, dimensFiltru):  
	    #number of pixels of the window’s border 
	    blackPx=dimensFiltru//2  
	      
	    imgFiltrata=[]  
	    imgFiltrata= np.zeros((len(img),len(img[0])))  
	      
	    #initiial image crossing  
	    for i in range(len(img)):  
	       for j in range(len(img[0])):      
	           val=[]  
	           #check if pixel is inside the border  
	           #window crossing   
	           for z in range(dimensFiltru):  
	               if i + z - blackPx >= 0 or i + z - blackPx <= len(img) - 1:  
	                    
	                   #if window fit  
	                   for k in range(dimensFiltru):  
	                         #then add the pixels to the list 
	                         val.append(img[i-1 + z - blackPx][j-1 + k - blackPx])  
	                
	               #Finding the median value:  
	                     
	               #The pixels are ascending sorted
	               val.sort()  
	                
	                #Swap the pixel with the middle value from the list  
	               imgFiltrata[i][j]= val[len(val) // 2]  
	    return imgFiltrata
      
Kuwahara filtering function

	def Kuwahara(img, dimensFereastra):  
	    #read the image values as float64 for more precise calculus  
	    imgCitita = img.astype(np.float64)  
	      
	    #check window size
	      
	    if dimensFereastra%2 ==0:  
	        raise Exception ("Dimensiune fereastra para. Dimensiune ceruta: impara")  
	          
	    #Building the regions  
	      
	    #First line of the first region  
	    tempRow = np.hstack((np.ones((1,(dimensFereastra-1)//2+1)),np.zeros((1,(dimensFereastra-1)//2)))) # 1 1 1 0 0  
	    tempRow2 = np.hstack((np.ones((1,(dimensFereastra-1)//2)),np.zeros((1,(dimensFereastra-1)//2+1)))) # 1 1 0 0 0  
	      
	    #padding  
	    tempPad = np.zeros((1,dimensFereastra)) # 0 0 0 0 0  
	      
	    #Copy the first line  
	    tempKernel = np.tile(tempRow, ((dimensFereastra-1)//2,1)) #tmpavgkerrow 3 times  
	    tempKernel= np.vstack((tempKernel,tempRow2[tempRow2[:,0]<2]))  
	      
	    #Add padding  
	    tempKernel = np.vstack((tempKernel, np.tile(tempPad, ((dimensFereastra-1)//2,1)))) #tile => 2x pad .. => vertical: 3x tmpavgkerrow + 2x pad  
	      
	    #Average of each element from the region  
	    tempKernel = tempKernel/np.sum(tempKernel) #np.sum(tempKernel) = number of elements from each region
	      
	    # tempKernel - N-W region
	      
	    # Building the window with 4 regions  
	    avgKernel = np.empty((4,dimensFereastra,dimensFereastra)) # empty array for the 4 regions  
	      
	    #Regions
	    avgKernel[0] = tempKernel           # N-W (a) region  
	    avgKernel[1] = np.fliplr(tempKernel)    # N-E (b) region  
	    avgKernel[2] = np.flipud(tempKernel)    # S-W (c) region 
	    avgKernel[3] = np.fliplr(avgKernel[2])  # S-E (d) region  
	      
	    # Initialized squared image pixel by pixel
	      
	      
	    # Initialized arrays for average of the regions and deviations  
	    medieRegiuni = np.zeros([4, imgCitita.shape[0],imgCitita.shape[1]])  
	    deviatiiStandard = medieRegiuni.copy()  
	      
	    # Regions average and deviations 
	    for k in range(4):  
	       medieRegiuni[k] = convolve2d(imgCitita, avgKernel[k],mode='same')        # regions average ; same= same data type as argument1  
	       squaredImg = (imgCitita-medieRegiuni[k])**2  
	       deviatiiStandard[k] = convolve2d(squaredImg, avgKernel[k], mode='same')  # average of squared regions
	      
	    index = np.argmin(deviatiiStandard,0) # Index of the window with lowest deviation 
	      
	    # Building the filtered image
	    filtered = np.zeros(img.shape)  
	    for row in range(img.shape[0]):  
	        for col in range(img.shape[1]):  
	            # Image building with the new graylevel values  
	            filtered[row,col] = medieRegiuni[index[row,col], row,col]             
	    return filtered.astype(np.uint8)
      
Display function

	functieAfisare(img,'Imaginea originala', 'Histograma imaginii originale')   
	  
	#Applying noise  
	imgZgomot=functieZgomot(np.array(img2))  
	functieAfisare(imgZgomot,'Imaginea afectata de zgomot', 'Histograma imaginii afectate de zgomot')  
	  
	#Median filtering  
	imagineFiltrata=filtrareMediana(imgZgomot,3)  
	functieAfisare(imagineFiltrata,'Imagine filtrata Median','Histograma imaginii filtrate Median')  
	  
	#Kuwahara filtering  
	imagineKuwahara=Kuwahara(imgZgomot,5)  
	functieAfisare(imagineKuwahara,'Imagine filtrata Kuwahara','Histograma imaginii filtrate Kuwahara')  


• Results
 
 IMAGE1:
 
 
 ![image](https://user-images.githubusercontent.com/53474954/108868820-3f0f2700-75ff-11eb-9f83-a07bcb409c43.png)
 ![image](https://user-images.githubusercontent.com/53474954/108868837-43d3db00-75ff-11eb-8829-70cc34c71833.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108870593-ed679c00-7600-11eb-8bab-3427d69c320f.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108870606-ef315f80-7600-11eb-8226-84f17cd76c2c.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108870613-f0fb2300-7600-11eb-9355-f47049dca5ef.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108870627-f22c5000-7600-11eb-8235-808bd70b9ede.png)


 IMAGE2:


 ![image](https://user-images.githubusercontent.com/53474954/108869120-8d242a80-75ff-11eb-800f-1b999c2acd2f.png)
 ![image](https://user-images.githubusercontent.com/53474954/108869138-92817500-75ff-11eb-9469-264fba78113c.png) 
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869156-97462900-75ff-11eb-9da1-d390b4791773.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869167-9ad9b000-75ff-11eb-9f82-8aa630e20d1e.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869181-9e6d3700-75ff-11eb-93ce-ac40320ba317.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869195-a2995480-75ff-11eb-8695-eb9ccc0adfa4.png)
 
 
IMAGE3:


 ![image](https://user-images.githubusercontent.com/53474954/108869232-aaf18f80-75ff-11eb-93c6-c10a4cec583f.png)
 ![image](https://user-images.githubusercontent.com/53474954/108869252-afb64380-75ff-11eb-9204-5b36a9601c28.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869267-b349ca80-75ff-11eb-9378-b6de06689d97.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869280-b6dd5180-75ff-11eb-89e5-5dea56004d67.png) 
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869294-b9d84200-75ff-11eb-8f0c-2411edceb049.png) 
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869306-bd6bc900-75ff-11eb-8b9b-c3435fb4ba0a.png)
 

IMAGE4:
 
 
 ![image](https://user-images.githubusercontent.com/53474954/108869339-c3fa4080-75ff-11eb-91ad-d64a02f687b6.png)
 ![image](https://user-images.githubusercontent.com/53474954/108869346-c78dc780-75ff-11eb-8bf9-24120c3c8d06.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869358-ca88b800-75ff-11eb-9fff-d796e36fb0aa.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869377-d07e9900-75ff-11eb-9d55-7cc02ffaf3ec.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869395-d5434d00-75ff-11eb-9db5-8f924ec7e6dd.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869419-daa09780-75ff-11eb-8917-4974e914007c.png)
 

 IMAGE5:

 ![image](https://user-images.githubusercontent.com/53474954/108869441-e12f0f00-75ff-11eb-99dd-f4dd62817ae4.png)
 ![image](https://user-images.githubusercontent.com/53474954/108869453-e55b2c80-75ff-11eb-9101-a2f820e7e2da.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869482-ebe9a400-75ff-11eb-8c0d-9f33d0b6bf0b.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869498-ef7d2b00-75ff-11eb-8868-5cfc73903f1d.png)
 
3x3

 ![image](https://user-images.githubusercontent.com/53474954/108869510-f310b200-75ff-11eb-9a97-288fe84899e6.png)
 
5x5

 ![image](https://user-images.githubusercontent.com/53474954/108869525-f60ba280-75ff-11eb-9fd4-9fbf305b5bd0.png)


Bibliography

	[Thankachan2014] R. Thankachan, P. S. Varsha, Improved Kuwahara Filter for Bipolar Impulse Noise Removal and Edge Preservation in Color Images and Videos. International Journal of Engineering Research & Technology (IJERT), Vol.3, November 2014
	[Bartyzel2016] K. Bartyzel, Adaptive Kuwahara Filter.Signal, in: Image and Video Processing, Editura Springer, Vol.10, April 2016
	[Vlaicu97] Aurel Vlaicu, Prelucrarea numerică a imaginilor, Editura Albastră,  Cluj-Napoca, 1997. 
	www.automation.ucv.ro/imago/Sinteza2009.pdf
	http://sorana.academicdirect.ro/pages/doc/PI/Curs_05.pdf
	https://www.miv.ro/ro/documentatie/pi/PIlab07.pdf

Complete project

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
	    blackPx=dimensFiltru/2  
	      
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
	    avgKernel[0] = tempKernel           # regiunea N-W (a)   
	    avgKernel[1] = np.fliplr(tempKernel)    # regiunea N-E (b)  
	    avgKernel[2] = np.flipud(tempKernel)    # regiunea S-W (c)  
	    avgKernel[3] = np.fliplr(avgKernel[2])  # Regiunea S-E (d)  
	      
	    # Initializare patratul imaginii pixel cu pixel  
	      
	      
	    # Initializare array-uri pentru media regiunilor si deviatii  
	    medieRegiuni = np.zeros([4, imgCitita.shape[0],imgCitita.shape[1]])  
	    deviatiiStandard = medieRegiuni.copy()  
	      
	    # Calcularea mediilor si deviatiilor pentru regiuni  
	    for k in range(4):  
	       medieRegiuni[k] = convolve2d(imgCitita, avgKernel[k],mode='same')        # media regiuniilor ; same=acelasi tip de date ca argument1  
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


