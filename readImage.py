# Importing Image from PIL package 
from PIL import Image
import os
import pickle

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

def createImgList(path):
    if os.path.isfile(path) == False:
        i = 0
        imgList = {}
        for directory in os.listdir("trainingSet"):

            for file in os.listdir("trainingSet/" + directory):
                imgList[i] = {}
                filePath = "trainingSet/" + directory + "/" + file
                imgList[i]["number"] = directory
                imgList[i]["pixelList"] = imgPxList(filePath)
                i += 1
                if i % 100 == 0:
                    print(i)

        with open(path, 'wb') as f:
            # serialize the pixel list to the file
            pickle.dump(imgList, f)
            f.close()

def createTinyImgList(path):
    if os.path.isfile(path) == False:
        imgList = {
            0: {
                "number": 0,
                "pixelList": [0, 0]
            },
            1: {
                "number": 1,
                "pixelList": [1, 0]
            },
            2: {
                "number": 0,
                "pixelList": [0, 0]
            },
            3: {
                "number": 0,
                "pixelList": [1, 1]
            },
            4: {
                "number": 1,
                "pixelList": [0, 1]
            },
            5: {
                "number": 1,
                "pixelList": [0, 1]
            },
            6: {
                "number": 0,
                "pixelList": [0, 0]
            },
            7: {
                "number": 0,
                "pixelList": [1, 1]
            },
            8: {
                "number": 1,
                "pixelList": [1, 0]
            },
            9: {
                "number": 1,
                "pixelList": [1, 0]
            }
        }
        print(imgList)
        with open(path, 'wb') as f:
            # serialize the pixel list to the file
            pickle.dump(imgList, f)
            f.close()