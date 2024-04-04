import random
import createWeightsAndBiases
import readImage
import mathFunctions
from runNeuralNetwork import runNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

settings = createWeightsAndBiases.returnJsonFileData()
canvasHight = settings["canvasHight"]
canvasWidth = settings["canvasWidth"]
fileWeightsAndBiases = settings["weightsAndBiasesFile"]
fileImgList = settings["imgListFile"]
testFileImgList = settings["testImgListFile"]
costList = settings["costList"]

learningRate = settings["learningRate"]
testSize = settings["testSize"]

inputLayer = canvasWidth * canvasHight
hiddenLayers = {}
for layerKey, layer in settings["hiddenLayers"].items():
    hiddenLayers[int(layerKey)] = layer

# Retreve weights and biases from file
# When the file does not exist create a new one
try:
    weightsAndBiases = createWeightsAndBiases.returnFileData(fileWeightsAndBiases)
except FileNotFoundError:
    createWeightsAndBiases.createWeightsAndBiasesFile(inputLayer, hiddenLayers, fileWeightsAndBiases)
    weightsAndBiases = createWeightsAndBiases.returnFileData(fileWeightsAndBiases)

def main(weightsAndBiases):
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
        costListData = createWeightsAndBiases.returnFileData(costList)
        
    except FileNotFoundError:
        costSum = 0
        for i, _ in enumerate(testList):
            costSum += calcCost(weightsAndBiases, testList[i]["number"], testList[i]["pixelList"])
        avgCost = costSum / len(testList)

        costListData = []
        costListData.append(avgCost)
        createWeightsAndBiases.createFile(costList, costListData)

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
        
    costSum = 0
    for i, _ in enumerate(testList):
        costSum += calcCost(weightsAndBiases, testList[i]["number"], testList[i]["pixelList"])
    avgCost = costSum / len(testList)

    costListData = createWeightsAndBiases.returnFileData(costList)
    costListData.append(avgCost)
    createWeightsAndBiases.alterFileData(costList, costListData)

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
    # Run the neural network and retreve the value of the neurons
    valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)
    cost = 0

    for outputKey, output in valueNeurons[len(valueNeurons) - 1].items():
        desiredOutput = calcDesiredOutput(outputKey, number)
        cost += mathFunctions.calcCost(output, desiredOutput)
    
    return cost
            

for _ in range(1):
    print("#" * 20)
    main(weightsAndBiases)

costListData = createWeightsAndBiases.returnFileData(costList)
ypoints = np.array(costListData)

plt.plot(ypoints)
plt.show()