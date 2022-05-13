import numpy as np
#import argparse
#import imutils
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import Plotting
#from Plotting import generateResultsChart

def existingFiberIds(arr, greaterthanany):  #####Checking what Fiber IDs are actually existing, returns array of the ids
    mp = {k: 0 for k in range(int(greaterthanany)+5)}
    #print(greaterthanany)
    existingfibers=[]
    #print(len(arr), len(arr[0]),"pog")
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            #print(i,j,"ij")
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
    mp = {k: 0 for k in range(int(greaterthanany)+1)}
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
    #print(newmatrix,"\n", newmatrix2)
    return newmatrix2#, newmatrix3

def CountPixels(matrix,number):    #nuber of pixels with a specidic id
    pixelnumber=0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                pixelnumber=pixelnumber+1
    return pixelnumber

def findhighestIDnumber(matrix): ###largest id number in the matrix
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]>numero: numero=matrix[i][j]
    return numero
def findlowestIDnumber(matrix): ### lowest id number in the matrix
    numero= findhighestIDnumber(matrix)
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


def ComparatorOf2(matrix,matrix2): ###compares 2 matrices, returnes the number of identified and misidentified fibers
    numero=findhighestIDnumber(matrix)
    print(numero)
    numero2=findhighestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=findlowestIDnumber(matrix)
    print(numero,numero2)
    numero=numero+1
    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)
    notdetected = 0
    mp = {k: 0 for k in range(int(numero2) + 5)}
    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(gtpixel, numero2)
        if a!=0:

            pixelb=CountPixels(matrix2,a) ##########number of pixels of ID2 in the compared matrix
            pixela=CountPixels(gtpixel,a) ##########number of pixels of ID2 in the original fiber
            pixelc=CountPixels(matrix, number) ##########number of pixels of ID1 in the original fiber
            mp[a]+=1
            ratio=pixela/pixelb
            ratio2=pixela/pixelc
            print(pixela, pixelb,pixelc, ratio, ratio2, number)
            if ratio>0.8:
                identified=identified+1
            else:
                misidentified=misidentified+1
        else:
            misidentified=misidentified+1
            notdetected=notdetected+1
    mui=0
    mp2 = {k: 0 for k in range(int(numero2) + 5)}
    for i in range(int(numero2)+5):
        if mp[i]>1:
            mui=mui+1
            mp2[mp[i]]=mp2[mp[i]]+1
    array=[]
    for i in range(int(numero2)+5):
        if mp2[i]>0:
            array.append((mp2[i],"groups of", i))
    alpha=identified/numberoffibers2
    beta=misidentified/(misidentified+identified)
    gamma=notdetected/numberoffibers2
    delta=mui/numberoffibers2
    return identified, misidentified, numero, numberoffibers2, numberoffibers22, notdetected, alpha, beta, gamma, delta,array

def ComparatorM(matrix,matrix2):

    ###### Delete only if watershed
    #matrix2 = np.delete(matrix2, (0), axis=0)
    #matrix2 = np.delete(matrix2,(0), axis=1)
    #matrix = np.delete(matrix, (0), axis=0)
    #matrix = np.delete(matrix,(0), axis=1)
    print(matrix.shape)
    matrix2 = np.pad(matrix2, ((0, 2), (0, 2)))
    #matrix = np.pad(matrix, ((0, 2), (0, 2)))
    matrix = matrix.astype(int)
    matrix2= matrix2.astype(int)


    start = time.time()
    comparatorW=ComparatorOf2(matrix,matrix2)
    print("For Watershed: identified", comparatorW[0],"misidentified",comparatorW[1],"Number of fibers in GT", comparatorW[3],"Number of fibers detected by watershed ",comparatorW[4], "notdetected", comparatorW[5] )
    print("alpha",comparatorW[6],"beta",comparatorW[7] , "gamma", comparatorW[8], "delta",comparatorW[9])
    print(comparatorW[10])
#plotting = Plotting()
#generateResultsChart([0.6,0.7,0.8,0.4],[0.6,0.5,0.8,0.6],[0.6,0.6,0.8,0.4],[0.9,0.9,0.9,0.4])
#plotting.generateResultsChart
    end = time.time()
    print("Time",end - start)

#comparatorS=ComparatorOf2(matrix,matrix3)
#print(" For Stardist: identified", comparatorS[0],"misidentified",comparatorS[1])
def DELTA(matrix,matrix2):
    print(matrix.shape)
    matrix2 = np.pad(matrix2, ((0, 2), (0, 2)))
    # matrix = np.pad(matrix, ((0, 2), (0, 2)))
    matrix = matrix.astype(int)
    matrix2 = matrix2.astype(int)
    numero=findhighestIDnumber(matrix)
    print(numero)
    numero2=findhighestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=findlowestIDnumber(matrix)
    print(numero,numero2)
    numero=numero+1
    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)
    notdetected = 0
    mp = {k: 0 for k in range(int(numero2) + 5)}
    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(gtpixel, numero2)
        if a!=0:
            mp[a] += 1
    mui = 0
    mp2 = {k: 0 for k in range(int(numero2) + 5)}
    for i in range(int(numero2)+5):
        if mp[i]>1:
            mui=mui+1
            mp2[mp[i]]=mp2[mp[i]]+1
    array=[]
    for i in range(int(numero2)+5):
        if mp2[i]>0:
            array.append((mp2[i],"groups of", i))
    delta = mui / numberoffibers2
    return delta

matrix = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data Processed\Annotated\GroundTruth\Tape_B_1_4.csv', delimiter=",")
matrix2 = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data Processed\Watershed\Tape_B_1_4.csv', delimiter=',')
start=time.time()
print(DELTA(matrix,matrix2))
end = time.time()
print("Time",end - start)