# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

# Student: Han Le - 111679478

from __future__ import print_function
from __future__ import division
import os
import sys
import cv2
import numpy as np


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):
    # Write histogram equalization here
    # Write histogram equalization here
    b, g, r = cv2.split(img_in)  # split 3 channels to work on them separately
    img_all = [b, g, r]  # save into 1 list to iterate
    img_out = []  # output image
    height, width = np.shape(img_all[0])[0], np.shape(img_all[0])[1]
    for i in xrange(0, 3):
        # calculate the hist of the image.
        hist, _ = np.histogram(
            img_all[i], 256, [0,
                              256])  # calculate the histograme of each channel

        # calculate the hist of the image.
        cdf_in = hist.cumsum()  # calculate the cdf of the histogram

        # calculate the modified cdf
        # formula is from Prof's reading material: http://cache.freescale.com/files/dsp/doc/app_note/AN4318.pdf
        cdf_modified = np.uint8(
            (cdf_in - cdf_in.min()) * 255 / (height * width - cdf_in.min()))

        temp = cdf_modified[img_all[
            i]]  # mapping the pixel values from the original image to the new one
        img_out.append(temp)  # save the result of each channel, merge later

    img_out = cv2.merge(img_out)

    # validate result:
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    img_val = cv2.merge((blue, green, red))
    # print("validation: ", np.abs(np.sum(img_out - img_val)))
    # print("validation: ", img_val)
    # print("img_out: ", img_out)
    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):

    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    img_in = img_in[:, :, 0]
    # Write low pass filter here
    fft_img = ft(img_in)
    window = 20  # LPF window of 20x20
    f_height, f_width = np.shape(fft_img)

    f_shift_LPF = fft_img.copy()
    mask_LPF = np.zeros([f_height, f_width])
    h_2, w_2 = np.int(f_height / 2), np.int(f_width / 2)
    window_2 = np.int(window / 2)

    mask_LPF[h_2 - window_2:h_2 + window_2, w_2 - window_2:w_2 + window_2] = 1
    f_shift_LPF = f_shift_LPF * mask_LPF
    img_out = np.uint8(ift(f_shift_LPF))

    return True, img_out


def high_pass_filter(img_in):

    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    img_in = img_in[:, :, 0]
    # Write high pass filter here
    fft_img = ft(img_in)
    window = 20  # LPF window of 20x20
    f_height, f_width = np.shape(fft_img)
    h_2, w_2 = np.int(f_height / 2), np.int(f_width / 2)
    window_2 = np.int(window / 2)

    f_shift_HPF = fft_img.copy()
    mask_HPF = np.ones([f_height, f_width])
    mask_HPF[h_2 - window_2:h_2 + window_2, w_2 - window_2:w_2 + window_2] = 0
    f_shift_HPF = f_shift_HPF * mask_HPF
    img_out = np.uint8(ift(f_shift_HPF))
    return True, img_out


def deconvolution(img_in):

    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    # Write deconvolution codes here
    blurred_fft = ft(img_in)
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T  # create the square kernel
    gkf = ft(gk, (np.shape(img_in)[0],
                  np.shape(img_in)[1]))  # so we can multiple easily

    # de-blur the input image by dividing the the blurred on to the kernel
    img_out = np.uint8((ift(blurred_fft / gkf)) * 255)

    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    # input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(
        sys.argv[3], cv2.IMREAD_ANYCOLOR |
        cv2.IMREAD_ANYDEPTH)  # new command to read .exr image

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here
    A = img_in1
    B = img_in2
    # resize the image A to have same size as image B
    # height, width = np.shape(B)[0], np.shape(B)[1]
    # A = cv2.resize(A, (width, height))

    A = A[:, :A.shape[0]]
    B = B[:A.shape[0], :A.shape[0]]

    deep_level = 5
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(0, deep_level + 1):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(0, deep_level + 1):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A, which contains high freq elements
    lpA = [gpA[deep_level]]
    for i in xrange(deep_level, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
        height, width = np.shape(gpA[i - 1])[0], np.shape(gpA[i - 1])[1]
        if not np.shape(GE) == np.shape(gpA[i - 1]):
            L = cv2.subtract(gpA[i - 1], cv2.resize(GE, (width, height)))
        else:
            L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[deep_level]]
    for i in xrange(deep_level, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
        height, width = np.shape(gpB[i - 1])[0], np.shape(gpB[i - 1])[1]
        if not np.shape(GE) == np.shape(gpB[i - 1]):
            L = cv2.subtract(gpB[i - 1], cv2.resize(GE, (width, height)))
        else:
            L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = np.shape(la)
        ls = np.hstack((la[:, 0:np.int(cols / 2), :],
                        lb[:, np.int(cols / 2):, :]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]  # LS[0] is the LOW freq element
    for i in xrange(1, deep_level + 1):
        ls_ = cv2.pyrUp(ls_)

        # addition requires same size matrix --> resize 1 image to have the same size if necessary
        height, width = np.shape(LS[i])[0], np.shape(LS[i])[1]
        if not np.shape(ls_) == np.shape(LS[i]):
            ls_ = cv2.add(cv2.resize(ls_, (width, height)), LS[i])
        else:
            ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:, :np.int(cols / 2)], B[:, np.int(cols / 2):]))

    return True, ls_


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
