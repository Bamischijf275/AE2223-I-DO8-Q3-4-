# read image through command line
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#from scipy import ndimage
import numpy as np
#import argparse
#import imutils
import cv2
import pandas as pd

N = 3  # Rows 155

M = 3 # Columns 120

def existingFiberIds(arr, greaterthanany):  #####Checking what Fiber IDs are actually existing, returns array of the ids
    mp = {k: 0 for k in range(int(greaterthanany)+1)}
    #print(mp)
    # Store frequency of elements
    # in matrix
    maxi=0
    existingfibers=[]
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                mp[int(arr[i][j])] += 1

    for i in range(int(greaterthanany)+1):
        if mp[i]!=0:
            existingfibers=np.append(existingfibers, i, axis=None)

    return existingfibers

# Function to find count of all
# majority elements in a Matrix
def majorityInMatrix(arr,greaterthanany):  ###### Returns the number that happens the most times in a matrix
    # we take length equal to max
    # value in array

    mp = {k: 0 for k in range(int(greaterthanany)+1)}
    #print(mp)
    # Store frequency of elements
    # in matrix
    maxi=0
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                mp[int(arr[i][j])] += 1
                #print(mp)
                if mp[int(arr[i][j])]>maxi:
                    maxi=arr[i][j]

    return maxi

def GTpixels(matrix,matrix2,number): #Finds the corresponding id number in the other matrix (the ids do not directly correspond)

    imax = 0
    jmax = 0
    imin = len(matrix)
    jmin = len(matrix[0])
    thereisanumber=0
    #print(imin,jmin)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                thereisanumber = 1
                #print(matrix[i][j],"great")
                if i > imax: imax = i
                if i < imin: imin = i
                if j> jmax: jmax = j
                if j < jmin: jmin = j
    jmax=jmax+1
    imax=imax+1
    #print(imax, imin, jmax, jmin,number)
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
    #print("cut out pixels of a specific fiber of GT:\n",newmatrix)
    return newmatrix2

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


matrix = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\labels.csv', delimiter=",")
matrix2= np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\labels.csv', delimiter=',')
###### Delete only if watershed
#matrix2 = np.delete(matrix2, (0), axis=0)
#matrix2 = np.delete(matrix2,(0), axis=1)
#matrix = np.delete(matrix, (0), axis=0)
#matrix = np.delete(matrix,(0), axis=1)

def ComparatorOf2(matrix,matrix2): ###compares 2 matrices, returnes the number of identified and misidentified fibers
    numero=findlargestIDnumber(matrix)
    identified=0
    misidentified=0
    mini=findlowestIDnumber(matrix)
    number=mini
    print(numero)
    numero=numero+1
    thereisanumber=0
    exist=existingFiberIds(matrix,numero)
    for number in exist:
        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(GTpixels(matrix,matrix2,number), numero)
        pixelb=CountPixels(matrix2,a)
        pixela=CountPixels(gtpixel,a)
        ratio=pixela/pixelb
        if ratio>0.8:
            identified=identified+1
        else:
            misidentified=misidentified+1
    #print(number)
    #number=number+1
    #print("The ratio of pixels\n", pixela/pixelb)

    return identified, misidentified
comparatorW=ComparatorOf2(matrix,matrix2)
print(" For Watershed: identified", comparatorW[0],"misidentified",comparatorW[1])
#comparatorS=ComparatorOf2(matrix,matrix3)
#print(" For Stardist: identified", comparatorS[0],"misidentified",comparatorS[1])

