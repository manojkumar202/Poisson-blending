from audioop import avg
import cv2
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from align_target import align_target
from scipy.sparse.linalg import spsolve

def poisson_blending(im_source, target_image, mask):
    Height = mask.shape[0]
    Width = mask.shape[1]

    pst_var_by_xy_index = {}
    index=0
    for i in range(Height):
        for j in range(Width):
            if mask[i][j]==1:
                pst_var_by_xy_index[(i,j)] = index
                index = index+1
               
    #number_of_patch_pixels = index
    for i in range(Height):
        for j in range(Width):
            if mask[i][j]==1 and i-1>=0 and mask[i-1][j]==0 :
                im_source[i-1][j] = im_source[i][j]
                if not pst_var_by_xy_index.get((i-1,j)):
                    pst_var_by_xy_index[(i-1,j)] = index
                    index = index+1
     
            if mask[i][j]==1 and i+1<Height and mask[i+1][j]==0:
                im_source[i+1][j] = im_source[i][j]
                if not pst_var_by_xy_index.get((i+1,j)):
                    pst_var_by_xy_index[(i+1,j)] = index
                    index = index+1
                
            if mask[i][j]==1 and j-1>=0 and mask[i][j-1]==0:
                im_source[i][j-1] = im_source[i][j]
                if not pst_var_by_xy_index.get((i,j-1)):
                    pst_var_by_xy_index[(i,j-1)] = index
                    index = index+1
                
            if mask[i][j]==1 and j+1<Width and mask[i][j+1]==0:
                im_source[i][j+1] = im_source[i][j]
                if not pst_var_by_xy_index.get((i,j+1)):
                    pst_var_by_xy_index[(i,j+1)] = index
                    index = index+1
                
    no_inpainting_pixels_nboundary = index

    matrix_A = np.zeros((no_inpainting_pixels_nboundary, no_inpainting_pixels_nboundary))
    matrix_b = np.zeros(no_inpainting_pixels_nboundary)
    equation_number = 0
    
    for coordinates in pst_var_by_xy_index:
        i = coordinates[0]
        j = coordinates[1]
        if mask[i][j]==1:
            matrix_A[equation_number][pst_var_by_xy_index[(i,j)]] = 4
            matrix_A[equation_number][pst_var_by_xy_index[(i,j-1)]] = -1
            matrix_A[equation_number][pst_var_by_xy_index[(i,j+1)]] = -1
            matrix_A[equation_number][pst_var_by_xy_index[(i-1,j)]] = -1
            matrix_A[equation_number][pst_var_by_xy_index[(i+1,j)]] = -1
            matrix_b[equation_number] = 4*im_source[i][j] - im_source[i-1][j] - im_source[i+1][j] - im_source[i][j-1] - im_source[i][j+1]
        else:
            matrix_A[equation_number][pst_var_by_xy_index[(i,j)]] = 1
            matrix_b[equation_number] = target_image[i][j]
        equation_number = equation_number + 1
    
    print("**************** Solving ax = b ****************")
    sp_matrix_A = csr_matrix(matrix_A)
    sp_matrix_b = csr_matrix(matrix_b)
    image_reconstruction = spsolve(sp_matrix_A, sp_matrix_b.transpose())
    image_reconstruction_unchanged = image_reconstruction.copy()
    image_reconstruction[image_reconstruction<0] = 0
    image_reconstruction[image_reconstruction>255] = 255
    #image_reconstruction = np.linalg.pinv(matrix_A) @ matrix_b
    print("**************** Solved  ax = b ****************")
    print(len(image_reconstruction))
    #cv2.imshow("before_final", im_source_original)
    #cv2.waitKey(0)
    ind = 0
    for i in range(Height):
        for j in range(Width):
            if mask[i][j]==1:
                target_image[i][j] = image_reconstruction[pst_var_by_xy_index[(i,j)]]
                ind = ind + 1
    #cv2.imshow("after_final", target_image)
    #cv2.waitKey(0)
    least_square_error = np.linalg.norm((sp_matrix_A.dot(image_reconstruction_unchanged) - sp_matrix_b), ord=1)
    return target_image, least_square_error

source_path = 'source1.jpg'
target_path = 'target.jpg'

source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

#align target image
im_source, mask = align_target(source_image, target_image)
cv2.imwrite("sample_imsource.jpg", im_source)
#cv2.imshow("im_source", im_source)
#cv2.waitKey(0)

target_image[:,:,0], e1 = poisson_blending(im_source[:,:,0], target_image[:,:,0],  mask)
target_image[:,:,1], e2 = poisson_blending(im_source[:,:,1], target_image[:,:,1], mask)
target_image[:,:,2], e3 = poisson_blending(im_source[:,:,2], target_image[:,:,2], mask)
cv2.imwrite("sample_output_report_4.jpg", target_image)

print("The least square error is ", (e1+e2+e3)/3)

cv2.imshow("FINAL_OUTPUT", target_image)

cv2.waitKey(0)