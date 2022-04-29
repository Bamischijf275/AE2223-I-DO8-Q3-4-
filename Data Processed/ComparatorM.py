# read image through command line
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#from scipy import ndimage
import numpy as np
#import argparse
#import imutils
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def existingFiberIds(arr, greaterthanany):  #####Checking what Fiber IDs are actually existing, returns array of the ids
    mp = {k: 0 for k in range(int(greaterthanany)+5)}

    existingfibers=[]
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                #print(arr[i][j], i,j)
                mp[arr[i][j]] += 1

    for i in range(int(greaterthanany)+1):
        if mp[i]!=0:
            existingfibers=np.append(existingfibers, i, axis=None)

    return existingfibers

# Function to find count of all
# majority elements in a Matrix
def majorityInMatrix(arr,greaterthanany):  ###### Returns the number that happens the most times in a matrix
    # we take length equal to max
    # value in array
    #greaterthanany=findlargestIDnumber(arr)
    mp = {k: 0 for k in range(int(greaterthanany)+1)}
    #print(mp)
    # Store frequency of elements
    # in matrix
    #print(arr, "arr")
    #print(mp)
    maxi=0
    maxi2=0
    #print(greaterthanany,"bruh")
    #print(arr.shape, greaterthanany)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                #print(arr[i][j],"arr[i][j]")
                #print(mp[arr[i][j]])
                mp[int(arr[i][j])] += 1

                if mp[int(arr[i][j])]>maxi:
                    maxi=mp[int(arr[i][j])]
                    maxi2=arr[i][j]
    #print(maxi,mp)

    return maxi2

def GTpixels(matrix,matrix2,number): #Finds the corresponding id number in the other matrix (the ids do not directly correspond)

    imax = 0
    jmax = 0
    z=0
    imin = len(matrix)
    jmin = len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                if i > imax: imax = i
                if i < imin: imin = i
                if j> jmax: jmax = j
                if j < jmin: jmin = j



    jmax=jmax+1
    imax=imax+1
    #print(zi,zj,imax, imin, jmax, jmin,number)
    #print("pixel boundary:",imin,imax,jmin,jmax)
    newmatrix=np.zeros((imax-imin,jmax-jmin),dtype="int")
    newmatrix2 = np.zeros((imax - imin, jmax - jmin), dtype="int")
    #print(newmatrix)
    for i in range(imin,imax):
        for j in range(jmin,jmax):
            if matrix[i][j]==number:
                newmatrix[i-imin][j-jmin]=matrix[i][j]
            #if matrix[i][j] ==number:
                newmatrix2[i-imin][j-jmin]=matrix2[i][j]
    #matrix2 = np.pad(matrix2, ((2, 2), (2, 2)))
    #newmatrix3=matrix2[imin:imax,jmin:jmax]
    #print(newmatrix3,"gayy",number)
    #if number == 15:
    #print(newmatrix,"\n", newmatrix2)
    #print("cut out pixels of a specific fiber of GT:\n",newmatrix)
    return newmatrix2#, newmatrix3

def CountPixels(matrix,number):    #nuber of pixels with a specidic id
    pixelnumber=0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                pixelnumber=pixelnumber+1
    return pixelnumber

def findlargestIDnumber(matrix): ###largest id number in the matrix
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]>numero: numero=matrix[i][j]
    return numero
def findlowestIDnumber(matrix): ### lowest id number in the matrix
    numero= findlargestIDnumber(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]!=0:
                if matrix[i][j]<numero: numero=matrix[i][j]
    return numero
def findlowestandhighest(matrix):
    numero2 = 1000000
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                if matrix[i][j] < numero2: numero2 = matrix[i][j]
                if matrix[i][j] > numero: numero = matrix[i][j]
    return numero2, numero


matrix = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data Processed\GroundTruth\Tape_B_1_1.jpg.tif.csv', delimiter=",")
matrix2= np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data Processed\Watershed\Tape_B_1_1.csv', delimiter=',')
###### Delete only if watershed
matrix2 = np.delete(matrix2, (0), axis=0)
matrix2 = np.delete(matrix2,(0), axis=1)
#matrix = np.delete(matrix, (0), axis=0)
#matrix = np.delete(matrix,(0), axis=1)
print(matrix.shape)

#matrix2 = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data\Tape_B\CSV_Masks\Tape_B_1_1.jpg.tif.csv', delimiter=",")
matrix2 = np.pad(matrix2, ((0, 2), (0, 2)))
matrix = matrix.astype(int)


#matrix2= matrix.astype(int)

def ComparatorOf2(matrix,matrix2): ###compares 2 matrices, returnes the number of identified and misidentified fibers
    numero=findlowestandhighest(matrix)
    numero2=findlargestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=int(numero[0])
    numero=numero[1]


    print(numero)
    numero=numero+1
    #start = time.time()
    #print("hellow2")

    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    #end = time.time()
    #print(end - start)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)

    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)

        #gtpixelb=gtpixel[1]
        #print(gtpixel,"gtpixelb")
        a=majorityInMatrix(gtpixel, numero2)
        notdetected=0
        #print(a)
        #print(number, a)
        if a!=0:

            pixelb=CountPixels(matrix2,a) ##########number of pixels of ID2 in the compared matrix
            pixela=CountPixels(gtpixel,a) ##########number of pixels of ID2 in the original fiber
            pixelc=CountPixels(matrix, number) ##########number of pixels of ID1 in the original fiber
            #print(pixela, pixelb, number, "gtpixelb:",gtpixelb)
            #print(GTpixels(matrix2,matrix2,414))
            #print(gtpixelb)
            ratio=pixela/pixelb
            ratio2=pixela/pixelc
            print(pixela, pixelb,pixelc, ratio, ratio2, number)
            if ratio>0.7:
                identified=identified+1
            #print(number, "identified")
            else:
                misidentified=misidentified+1
            #print(number, "misidentified")
        #if a==0:
            #notdetected=notdetected+1
        else: misidentified=misidentified+1


    #number=number+1
    #print("The ratio of pixels\n", pixela/pixelb)

    return identified, misidentified, numero, numberoffibers2, numberoffibers22, notdetected

start = time.time()
comparatorW=ComparatorOf2(matrix,matrix2)
print("For Watershed: identified", comparatorW[0],"misidentified",comparatorW[1],"Number of fibers in GT", comparatorW[3],"Number of fibers detected by watershed ",comparatorW[4], "notdetected", comparatorW[5] )
end = time.time()
print(end - start)
#comparatorS=ComparatorOf2(matrix,matrix3)
#print(" For Stardist: identified", comparatorS[0],"misidentified",comparatorS[1])

