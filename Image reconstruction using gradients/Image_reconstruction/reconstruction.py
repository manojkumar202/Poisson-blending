import numpy as np
import cv2
import os
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

if __name__ == '__main__':
    ##read source image
    source_path = '/Users/manojkumar/Downloads/Assignment 1 submission/part2/Image_reconstruction/large.jpg'
    source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

    H = source_image.shape[0]
    W = source_image.shape[1]
    #source_image = cv2.resize(source_image, (H, W))
    number_of_pixels = H * W
    
    A = np.zeros((number_of_pixels, number_of_pixels))
    b = np.zeros(H*W)
    equation_number = 0
    for i in range(H):
        for j in range(W):
            if i==0 and j==0:
                A[equation_number][i*W+j] = 1
                b[equation_number] = 100

            elif i==H-1 and j==0:
                A[equation_number][i*W+j] = 1
                b[equation_number] = 110

            elif i==0 and j==W-1:
                A[equation_number][i*W+j] = 1
                b[equation_number] = 200

            elif i==H-1 and j==W-1:
                A[equation_number][i*W+j] = 1
                b[equation_number] = 100
                
            elif i==0 or i==H-1:
                A[equation_number][i*W+j] = 2
                A[equation_number][i*W+(j-1)] = -1
                A[equation_number][i*W+(j+1)] = -1
                b[equation_number] = 2*source_image[i][j] - source_image[i][j-1] - source_image[i][j+1]
            elif j==0 or j==W-1:
                A[equation_number][i*W+j] = 2
                A[equation_number][(i-1)*W+j] = -1
                A[equation_number][(i+1)*W+j] = -1
                b[equation_number] = 2*source_image[i][j] - source_image[i-1][j] - source_image[i+1][j]
            else:
                A[equation_number][i*W+j] = 4
                A[equation_number][i*W+(j-1)] = -1
                A[equation_number][i*W+(j+1)] = -1
                A[equation_number][(i-1)*W+j] = -1
                A[equation_number][(i+1)*W+j] = -1
                b[equation_number] = 4*source_image[i][j] - source_image[i-1][j] - source_image[i+1][j] - source_image[i][j-1] - source_image[i][j+1]
            equation_number = equation_number + 1
    
    print("Before linear equation formation")
    sp_matrix_A = csr_matrix(A)
    sp_matrix_b = csr_matrix(b)
    image_reconstruction = spsolve(sp_matrix_A, sp_matrix_b.transpose())
    least_square_error = np.linalg.norm((sp_matrix_A.dot(image_reconstruction) - sp_matrix_b), ord=1)
    print("The least square error is ", least_square_error)
    #image_reconstruction = np.linalg.pinv(A) @ b
    print("Successfully completed")
    image_hat = image_reconstruction.reshape(H, W)
    
    cv2.imwrite("large_output.jpg", image_hat)
    cv2.imshow('image', image_hat)
    cv2.waitKey(0)