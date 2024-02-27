import random
import createWeightsAndBiases
import readImage
from runNeuralNetwork import runNeuralNetwork

file = "weightsAndBiases.pkl"
# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)

def main(weightsAndBiases):
    trainingList = list(range(42000))
    random.shuffle(trainingList)

    for i in trainingList:
        sgd(weightsAndBiases, findNumber(i), i)

def findNumber(index):
    for n in range(10):
        try:
            open(f"trainingSet\{n}\img_{index}.jpg")
            return n
        except FileNotFoundError:
            pass

def sgd(weightsAndBiases, number, index):
    imagePath = f"trainingSet\{number}\img_{index}.jpg"
    # Get list of pixel values from image
    imgPixelList = readImage.imgPxList(imagePath)

    #valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)
    print(runNeuralNetwork(weightsAndBiases, imgPixelList))

main(weightsAndBiases)