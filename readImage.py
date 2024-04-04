# Importing Image from PIL package 
from PIL import Image
import os
import pickle
import createWeightsAndBiases

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

def createImgList(path, testSize):
    settings = createWeightsAndBiases.returnJsonFileData()
    testSize = int(1 / testSize)
    testFileImgList = settings["testImgListFile"]
    directory = settings["trainingSetDirectory"]
    if os.path.isfile(path) == False or os.path.isfile(testFileImgList) == False:
        t = 0
        i = 0
        j = 0
        imgList = {}
        testImgList = {}
        for folder in os.listdir(directory):
            directory = directory + "/"

            for file in os.listdir(directory + folder):
                if t % testSize != 0:
                    imgList[i] = {}
                    filePath = directory + folder + "/" + file
                    imgList[i]["number"] = folder
                    imgList[i]["pixelList"] = imgPxList(filePath)
                    i += 1
                elif t % testSize == 0:
                    testImgList[j] = {}
                    filePath = directory + folder + "/" + file
                    testImgList[j]["number"] = folder
                    testImgList[j]["pixelList"] = imgPxList(filePath)
                    j += 1

                t += 1
                if (t) % 100 == 0:
                        print(t)

        with open(path, 'wb') as f:
            # serialize the pixel list to the file
            pickle.dump(imgList, f)
            f.close()

        with open(testFileImgList, 'wb') as f:
            # serialize the pixel list to the file
            pickle.dump(testImgList, f)
            f.close()