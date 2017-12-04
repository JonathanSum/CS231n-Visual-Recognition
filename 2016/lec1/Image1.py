from scipy.misc import imread, imsave, imresize

img = imread('cat1.jpg')
print(img.dtype, img.shape)
img_tinted = img *[1,0.95,0.9]
array1=img.shape
print("This is array shape" + str(array1) )
img_tinted = imresize(img_tinted,(300,300))
print(img_tinted.shape)
imsave('cat_tinted.jpg', img_tinted)
