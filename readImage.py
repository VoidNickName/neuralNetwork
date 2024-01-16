# Importing Image from PIL package 
from PIL import Image

def imgPxList(path):
    pxList = []
    for x in range(28):
        for y in range(28):
            pixel = {
                "x": x,
                "y": y
            }
            pxValue = readPixel(path, pixel)
            grayScalePx = calcGrayScale(pxValue)
            pxList.append(grayScalePx)
    return pxList


def readPixel(path, pixel):
    # creating a image object
    img = Image.open(path)
    if img.width >= pixel["x"] and img.height >= pixel["y"]:
        px = img.load()
        return (px[pixel["x"], pixel["y"]])

def calcGrayScale(i):
    return i/255

imagePath = "image.jpg"
print(imgPxList(imagePath))