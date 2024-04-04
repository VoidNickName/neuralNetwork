import random
import createWeightsAndBiases
import readImage
import mathFunctions
from runNeuralNetwork import runNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

def main():
    settings = createWeightsAndBiases.returnJsonFileData()
    fileWeightsAndBiases = settings["weightsAndBiasesFile"]
    fileImgList = settings["imgListFile"]
    testFileImgList = settings["testImgListFile"]
    costListFile = settings["costListFile"]

    learningRate = settings["learningRate"]
    testSize = settings["testSize"]
    minCostDifference = settings["minCostDifference"]

    try:
        costListData = createWeightsAndBiases.returnFileData(costListFile)
    except FileNotFoundError:
        costListData = {}
        costListData["costList"] = {}

    while len(costListData["costList"]) < 2 or costListData["costList"][-2] - costListData["costList"][-1] >= minCostDifference:
        print("#" * 20)
        train(fileWeightsAndBiases, fileImgList, testSize, testFileImgList, costListFile, learningRate)
        costListData = createWeightsAndBiases.returnFileData(costListFile)

    showGraph(costListFile)

def train(fileWeightsAndBiases, fileImgList, testSize, testFileImgList, costListFile, learningRate):

    # Get weights and biases
    weightsAndBiases = createWeightsAndBiases.getWeightsAndBiases()

    # Retreve traingList from file
    # When the file does not exist create a new one
    try:
        trainingList = createWeightsAndBiases.returnFileData(fileImgList)
    except FileNotFoundError:
        readImage.createImgList(fileImgList, testSize)
        trainingList = createWeightsAndBiases.returnFileData(fileImgList)
    
    # Create a list for the order of the training images
    trainingListOrder = (list(range(len(trainingList))))
    # Shuffle the training list
    random.shuffle(trainingListOrder)

    try:
        testList = createWeightsAndBiases.returnFileData(testFileImgList)
    except FileNotFoundError:
        readImage.createImgList(fileImgList, testSize)
        testList = createWeightsAndBiases.returnFileData(testFileImgList)

    try:
        costListData = createWeightsAndBiases.returnFileData(costListFile)
        
    except FileNotFoundError:
        totalCorrect = 0
        costSum = 0
        for i, _ in enumerate(testList):
            cost, correct = calcCost(weightsAndBiases, testList[i]["number"], testList[i]["pixelList"])
            costSum += cost
            totalCorrect += correct

        avgCost = costSum / len(testList)
        persentageCorrect = totalCorrect / len(testList) * 100

        costListData = {}
        costListData["costList"] = []
        costListData["persentageCorrect"] = []
        costListData["costList"].append(avgCost)
        costListData["persentageCorrect"].append(persentageCorrect)
        createWeightsAndBiases.createFile(costListFile, costListData)

    # Loop through the training list
    for progress, i in enumerate(trainingListOrder):
        # Calculate the gradient
        weightGradient, biasGradient = calcGradient(weightsAndBiases, trainingList[i]["number"], trainingList[i]["pixelList"])
        # Apply the gradient to the weights and biases
        weightsAndBiases = applyGradient(weightGradient, biasGradient, weightsAndBiases, learningRate)
        # Write the changes to the weights and biases to the file
        createWeightsAndBiases.alterFileData(fileWeightsAndBiases, weightsAndBiases)
        # Give a progress update every 100 iterations
        if (progress + 1) % 1000 == 0:
            print(progress + 1)

    calcAvgCost(testList, costListFile, weightsAndBiases)

def calcGradient(weightsAndBiases, number, imgPixelList):
    # Run the neural network and retreve the value of the neurons
    valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)

    weightGradient = {}
    biasGradient = {}
    activationGradient = {}

    # Loop through layers
    for layer in range(len(valueNeurons) - 1):
        weightGradient[layer] = {}
        biasGradient[layer] = {}
        activationGradient[layer] = {}
        
        if layer == 0:
            # Loop through the output layer
            for outputKey, output in valueNeurons[len(valueNeurons) - 1].items():
                weightGradient[layer][outputKey] = {}

                # Calculate the desired output of the current neuron
                desiredOutput = calcDesiredOutput(outputKey, number)

                # Calculate the derivative of the costfunction
                costDerivative = mathFunctions.calcCostDerivative(output, desiredOutput)

                # Calculate the delta
                delta = mathFunctions.calcDelta(costDerivative, output)

                # Set the bias gradient to the derivative of the bias which is equal to the delta
                biasGradient[layer][outputKey] = delta

                # Looping through the next layer
                for nKey, n in valueNeurons[len(valueNeurons) - 2].items():
                    # Retreve weight
                    weight = weightsAndBiases[len(valueNeurons) - 1][outputKey]["weights"][nKey]
                    # Calculate the gradient of the weight
                    weightGradient[layer][outputKey][nKey] = mathFunctions.weightDerivative(delta, n)

                    # Try to add the derivative of the activation to the activation gradient
                    # Else set the activation gradient to the derivative of the activation
                    try:
                        activationGradient[layer][nKey] += mathFunctions.activationDerivative(delta, weight)
                    except KeyError:
                        activationGradient[layer][nKey] = mathFunctions.activationDerivative(delta, weight)
        else:
            # Loop through the layer
            for outputKey, output in valueNeurons[len(valueNeurons) - 1 - layer].items():
                weightGradient[layer][outputKey] = {}

                # Calculate the activation gradient of current neuron
                activation = activationGradient[layer - 1][outputKey]

                # Calculate the delta
                delta = mathFunctions.calcDelta(activation, output)

                # Try to add the derivative of the bias to the bias gradient
                # Else set the bias gradient to the derivative of the bias which is equal to the delta
                try:
                    biasGradient[layer][outputKey] += delta
                except KeyError:
                    biasGradient[layer][outputKey] = delta

                # Looping through the next layer
                for nKey, n in valueNeurons[len(valueNeurons) - 2 - layer].items():

                    # Retreve weight
                    weight = weightsAndBiases[len(valueNeurons) - 1 - layer][outputKey]["weights"][nKey]
                    # Calculate the gradient of the weight
                    weightGradient[layer][outputKey][nKey] = mathFunctions.weightDerivative(delta, n)
                    
                    if len(valueNeurons) - layer != 2:

                        # Try to add the derivative of the activation to the activation gradient
                        # Else set the activation gradient to the derivative of the activation
                        try:
                            activationGradient[layer][nKey] += mathFunctions.activationDerivative(delta, weight)
                        except KeyError:
                            activationGradient[layer][nKey] = mathFunctions.activationDerivative(delta, weight)

    return weightGradient, biasGradient

def calcDesiredOutput(outputNumber, number):
    outputNumber = int(outputNumber)
    number = int(number)
    if outputNumber == number:
        return 1
    else:
        return 0

def applyGradient(weightGradient, biasGradient, weightsAndBiases, learningRate):
    # Loop through the layers of the weights and biases
    for layerKey, layer in weightsAndBiases.items():
        # Loop through the neurons of the layer
        for neuronKey, neuron in layer.items():
            # Calculate the new bias
            weightsAndBiases[layerKey][neuronKey]["bias"] = mathFunctions.changeWeightOrBias(neuron["bias"], learningRate, biasGradient[len(biasGradient) - layerKey][neuronKey])
            # Loop through the weights of the neuron
            for i, weight in enumerate(neuron["weights"]):
                # Calculate the new weight
                weightsAndBiases[layerKey][neuronKey]["weights"][i] = mathFunctions.changeWeightOrBias(weight, learningRate, weightGradient[len(weightGradient) - layerKey][neuronKey][i])
    return weightsAndBiases

def calcCost(weightsAndBiases, number, imgPixelList):
    number = int(number)
    # Run the neural network and retreve the value of the neurons
    valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)
    cost = 0

    for outputKey, output in valueNeurons[len(valueNeurons) - 1].items():
        desiredOutput = calcDesiredOutput(outputKey, number)
        cost += mathFunctions.calcCost(output, desiredOutput)
    
    output = valueNeurons[len(valueNeurons) -1]
    Keymax = max(zip(output.values(), output.keys()))[1]
    if number == Keymax:
        return cost, 1
    else:
        return cost, 0

def showGraph(costListFile):
    costListData = createWeightsAndBiases.returnFileData(costListFile)
    
    costList = costListData["costList"]
    ypoints = np.array(costList)
    plt.subplot(1, 2, 1)
    plt.plot(ypoints, marker = 'o')
    plt.title("Cost")
    plt.xlabel("Iterations")

    persentageCorrect = costListData["persentageCorrect"]
    ypoints = np.array(persentageCorrect)
    plt.subplot(1, 2, 2)
    plt.plot(ypoints, marker = 'o')
    plt.title("Persentage Correct")
    plt.xlabel("Iterations")

    plt.show()

def calcAvgCost(testList, costListFile, weightsAndBiases):
    totalCorrect = 0
    costSum = 0
    for i, _ in enumerate(testList):
        cost, correct = calcCost(weightsAndBiases, testList[i]["number"], testList[i]["pixelList"])
        costSum += cost
        totalCorrect += correct

    avgCost = costSum / len(testList)
    persentageCorrect = totalCorrect / len(testList) * 100

    costListData = createWeightsAndBiases.returnFileData(costListFile)
    costListData["costList"].append(avgCost)
    costListData["persentageCorrect"].append(persentageCorrect)
    createWeightsAndBiases.alterFileData(costListFile, costListData)