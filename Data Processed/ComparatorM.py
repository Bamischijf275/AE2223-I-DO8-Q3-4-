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


# Function to find count of all
# majority elements in a Matrix
def majorityInMatrix(arr,greaterthanany):
    # we take length equal to max
    # value in array
    mp = {i: 0 for i in range(greaterthanany)}

    # Store frequency of elements
    # in matrix
    maxi=0

    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i][j]!=0:
                mp[arr[i][j]] += 1
                if mp[arr[i][j]]>maxi:
                    maxi=arr[i][j]
    # loop to iteratre through map
    #countMajority = 0
    #for key, value in mp.items():

        # check if frequency is greater than
        # or equal to (N*M)/2
    #    if (value >= (int((N * M) / 2))):
     #       countMajority += 1

    return maxi



# Driver Code
#if __name__ == '__main__':
##    greaterthanany=7
  #  mat = [[0, 0, 6],
   #        [0, 0, 2],
  #         [1, 6, 6]]
    #print(majorityInMatrix(mat,greaterthanany))


def GTpixels(matrix,matrix2,number):
    imax = 0
    jmax = 0
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
    print(imin,imax,jmin,jmax)
    newmatrix=np.zeros((imax-imin,jmax-jmin),dtype="int")
    newmatrix2 = np.zeros((imax - imin, jmax - jmin), dtype="int")
    #print(newmatrix)
    for i in range(imin,imax):
        for j in range(jmin,jmax):
            if matrix[i][j]==number:
                newmatrix[i-imin][j-jmin]=matrix[i][j]
            #if matrix[i][j] ==number:
                newmatrix2[i-imin][j-jmin]=matrix2[i][j]
    print(newmatrix)
    return newmatrix2

def CountPixels(matrix,number):
    pixelnumber=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                pixelnumber=pixelnumber+1
    return pixelnumber






matrix = np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\labels.csv', delimiter=",")
matrix2= np.genfromtxt(r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\Data Processed\Watershed\Watershed.csv', delimiter=',')
matrix2 = np.delete(matrix2, (0), axis=0)
matrix2 = np.delete(matrix2,(0), axis=1)
#print(csv)

#print(matrix2)
#matrix = np.read_csv (r'C:\Users\mikol\PycharmProjects\AE2223-I-DO8-Q3-4-\labels.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
number=112
#print(matrix)
#print(len(matrix[0]))
print(CountPixels(matrix,517))
gtpixel=GTpixels(matrix,matrix2,number)
print(gtpixel)
a=majorityInMatrix(GTpixels(matrix,matrix2,number), len(matrix[0]))
print(a)## change to number of fibers
pixelb=CountPixels(matrix2,a)
pixela=CountPixels(gtpixel,a)
print(pixelb)
print(pixela)
print("The ratio of pixels", pixela/pixelb)