# Importing Image from PIL package 
from PIL import Image

# Create a list whith the grayscale of each pixel of an image
def imgPxList(path):
    pxList = []
    img = Image.open(path)
    # Loops through each pixel of an image
    for x in range(img.width):
        for y in range(img.height):
            pixel = {"x": x, "y": y}
            pxValue = readPixel(path, pixel)
            grayScalePx = calcGrayscale(pxValue)
            pxList.append(grayScalePx)
    return pxList

def readPixel(path, pixel):
    # Creating an image object
    img = Image.open(path)
    # Convert multi layer images into single layer images
    grayscaleImage = img.convert(mode="L")
    px = grayscaleImage.load()
    # Returns the pixel value
    return (px[pixel["x"], pixel["y"]])

# Converts a 0 to 255 scale into a 0 to 1 scale
def calcGrayscale(i):
    return i/255

#imagePath = "image.jpg"
#print(imgPxList(imagePath))