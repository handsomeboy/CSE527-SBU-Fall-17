from __future__ import division
import  numpy as np
import cv2
# from matplotlib import pyplot as plt
# from scipy import signal


def Question3_func():
    A = cv2.imread("input3A.jpg", cv2.IMREAD_COLOR)
    B = cv2.imread("input3B.jpg", cv2.IMREAD_COLOR)

    # A = cv2.imread("apple.png", cv2.IMREAD_COLOR)
    # B = cv2.imread("orange.png", cv2.IMREAD_COLOR)

    # height, width = np.shape(A)[0], np.shape(A)[1]
    # B = cv2.resize(B, (width, height))

    # resize the image A to have same size as image B
    height, width = np.shape(B)[0], np.shape(B)[1]
    A = cv2.resize(A, (width, height))

    deep_level = 5
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(0, deep_level):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(0, deep_level):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A, which contains high freq elements
    lpA = [gpA[deep_level]]
    for i in xrange(deep_level, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        # subtraction requires same size matrix --> resize 1 image to have the same size if necessary
        height, width = np.shape(gpA[i - 1])[0], np.shape(gpA[i - 1])[1]
        if not np.shape(GE) == np.shape(gpA[i-1]):
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
        if not np.shape(GE) == np.shape(gpB[i-1]):
            L = cv2.subtract(gpB[i - 1], cv2.resize(GE, (width, height)))
        else:
            L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = np.shape(la)
        ls = np.hstack((la[:, 0:cols / 2, :], lb[:, cols / 2:, :]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]         # LS[0] is the LOW freq element
    for i in xrange(1, deep_level + 1):
        ls_ = cv2.pyrUp(ls_)

        # addition requires same size matrix --> resize 1 image to have the same size if necessary
        height, width = np.shape(LS[i])[0], np.shape(LS[i])[1]
        if not np.shape(ls_) == np.shape(LS[i]):
            ls_ = cv2.add(cv2.resize(ls_, (width, height)), LS[i])
        else:
            ls_ = cv2.add(ls_, LS[i])

    # image with direct connecting each half
    real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))

    # cv2.imwrite('output3.png', ls_)

    # cv2.imshow("Real", real)
    cv2.imshow("pyramid", ls_)

    cv2.waitKey(0)

def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)

def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def Question2_func():
    img = cv2.imread('input2.png')
    img = img[:,:,0]

    fft_img = ft(img)
    window = 20                     # LPF window of 20x20
    f_height, f_width = np.shape(fft_img)

    """Low Pass filter"""
    f_shift_LPF = fft_img.copy()
    mask_LPF = np.zeros([f_height, f_width])
    mask_LPF[f_height/2 - window/2:f_height/2 + window/2, f_width/2 - window/2: f_width/2 + window//2] = 1
    f_shift_LPF = f_shift_LPF*mask_LPF
    LPF = np.uint8(ift(f_shift_LPF))

    # cv2.imwrite("output2LPF.png", LPF)
    cv2.imshow("LPF", LPF)
    # cv2.imshow("intput", img)

    """High Pass filter"""
    f_shift_HPF = fft_img.copy()
    mask_HPF = np.ones([f_height, f_width])
    mask_HPF[f_height/2 - window/2:f_height/2 + window/2, f_width/2 - window/2: f_width/2 + window//2] = 0
    f_shift_HPF = f_shift_HPF*mask_HPF
    HPF = np.uint8(ift(f_shift_HPF))

    # cv2.imwrite("output2HPF.png", HPF)
    # cv2.imshow("LPF", LPF)
    cv2.imshow("HPF", HPF)
    # cv2.imshow("intput", img)
    #
    # LPF = cv2.imread("output2LPF.png", 0)
    # HPF = cv2.imread("output2HPF.png", 0)
    #
    # fft = np.fft.fft2(LPF) + np.fft.fft2(HPF)
    # img_reconst = np.uint8(np.abs(np.fft.ifft2(fft)))
    # cv2.imshow("reconstruction", img_reconst)



    """ Deconvolution"""
    img_blurred = cv2.imread("blurred2.png", 0)

    # img_blurred = cv2.imread("blurred2.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    print "min of Blurred: ", np.min(img_blurred)
    print "max of Blurred: ", np.max(img_blurred)


    print "\n\n img_blurred: ", img_blurred
    print "size of im_blurred: ", np.shape(img_blurred)
    blurred_fft = ft(img_blurred)

    gk = cv2.getGaussianKernel(21, 5)
    print "size of gk: ", np.shape(gk)
    gk = gk * gk.T          # create the square kernel

    print "size of gk 2: ", np.shape(gk)
    # gkf = ft(gk, (np.shape(img_blurred)[0], np.shape(img_blurred)[1]))  # so we can multiple easily
    #
    # # de-blur the input image by dividing the the blurred on to the kernel
    # img_deBlurred = np.uint8((ift(blurred_fft/gkf)))
    #
    # cv2.imshow("de-blurred", img_deBlurred)
    #
    # cv2.imwrite("deblurred_result.png", img_deBlurred)



    # img_2 = cv2.imread("input2.png", 0)
    # gkf = ft(gk, (np.shape(img_2)[0], np.shape(img_2)[1]))
    # blur = ift((ft(img_2)*gkf))
    # blur_fft = ft(blur)
    # de_blur = (blur_fft/gkf)
    # print "de_blur: ", np.shape(de_blur)
    # de_blur = np.uint8(ift(de_blur))
    # cv2.imshow("blur", de_blur)

    cv2.waitKey(0)


def Question1_func():
    img_in = cv2.imread("input1.jpg", cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img_in)     # split 3 channels to work on them separately
    img_all = [b, g, r]             # save into 1 list to iterate
    img_out = []                    # output image
    height, width = np.shape(img_all[0])[0], np.shape(img_all[0])[1]
    for i in xrange(0,3):
        # calculate the hist of the image.
        hist,_ = np.histogram(img_all[i],256,[0,256])   # calculate the histograme of each channel

        # calculate the hist of the image.
        cdf_in = hist.cumsum()                          # calculate the cdf of the histogram

        # calculate the modified cdf
        # formula is from Prof's reading material: http://cache.freescale.com/files/dsp/doc/app_note/AN4318.pdf
        cdf_modified = np.uint8((cdf_in - cdf_in.min()) * 255 / (height*width - cdf_in.min()))

        temp = cdf_modified[img_all[i]]         # mapping the pixel values from the original image to the new one
        img_out.append(temp)                    # save the result of each channel, merge later

    #     hist_out,_ = np.histogram(temp,256,[0,256])
    #     cdf_out = hist_out.cumsum()
    #     cdf_out = cdf_out * hist_out.max() / cdf_out.max()  # this line not necessary.
    #     plt.plot(cdf_out)
    #
    # plt.xlim([0,256])
    # plt.legend(("Blue channel", "Green channel", "Red channel"), loc = 'upper left')
    # plt.show()

    # validate result:
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    img_val = cv2.merge((blue, green, red))
    cv2.imshow("validation", img_val)

    img_out = cv2.merge(img_out)

    cv2.imshow("input", img_in)
    cv2.imshow("output", img_out)
    cv2.imwrite("output1.png", img_out)

    print "validataion: ", np.sum(img_val - img_out)

    cv2.waitKey(0)

    # hist_out,_ = np.histogram(img_out,256,[0,256])
    # cdf_out = hist_out.cumsum()
    # cdf_out = cdf_out * hist_out.max() / cdf_out.max()  # this line not necessary.
    # plt.subplot(211)
    # plt.plot(hist)
    # plt.plot(cdf_in, '--*')
    # plt.legend(("Histogram", "cdf"), loc = 'upper left')
    # plt.title("input image")
    # plt.xlim([0,256])
    # plt.subplot(212)
    # plt.plot(hist_out)
    # plt.plot(cdf_out, '--*')
    # plt.title("output image")
    # plt.legend(("Histogram", "cdf"), loc = 'upper left')
    # plt.xlim([0,256])
    # plt.show()


Question1_func()
# Question2_func()
# Question3_func()

