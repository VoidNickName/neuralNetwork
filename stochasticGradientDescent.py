import random
import createWeightsAndBiases
import readImage
import mathFunctions
from runNeuralNetwork import runNeuralNetwork

file = "weightsAndBiases.pkl"
trainingListLenght = 42000
#trainingListLenght = 1
# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)
learningRate = 0.01

def main(weightsAndBiases):
    trainingList = list(range(trainingListLenght))
    random.shuffle(trainingList)

    progress = 0
    for i in trainingList:
        progress += 1
        weightGradient, biasGradient = calcGradient(weightsAndBiases, findNumber(i), i)
        weightsAndBiases = applyGradient(weightGradient, biasGradient, weightsAndBiases, learningRate)
        createWeightsAndBiases.alterFileData(file, weightsAndBiases)
        print(progress)


def findNumber(index):
    for n in range(10):
        try:
            open(f"trainingSet\{n}\img_{index}.jpg")
            return n
        except FileNotFoundError:
            pass

def calcGradient(weightsAndBiases, number, index):
    imagePath = f"trainingSet\{number}\img_{index}.jpg"
    # Get list of pixel values from image
    imgPixelList = readImage.imgPxList(imagePath)

    valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)

    weightGradient = {}
    biasGradient = {}
    activationGradient = {}

    for layer in range(len(valueNeurons) - 1):
        weightGradient[layer] = {}
        biasGradient[layer] = {}
        activationGradient[layer] = {}
        
        if layer == 0:
            for outputKey, output in valueNeurons[len(valueNeurons) - 1].items():
                weightGradient[layer][outputKey] = {}

                if outputKey not in biasGradient[layer]:
                    biasGradient[layer][outputKey] = 0

                desiredOutput = calcDesiredOutput(outputKey - 1, number)
                biasGradient[layer][outputKey] += biasDerivativeWithCost(output, desiredOutput)

                for nKey, n in valueNeurons[len(valueNeurons) - 2].items():
                    weight = weightsAndBiases[len(valueNeurons) - 1][outputKey]["weights"][nKey]

                    if nKey not in activationGradient[layer]:
                        activationGradient[layer][nKey] = 0

                    weightGradient[layer][outputKey][nKey] = weightDerivativeWithCost(output, n, desiredOutput)
                    activationGradient[layer][nKey] += activationDerivativeWithCost(output, desiredOutput, weight)
        else:
            for outputKey, output in valueNeurons[len(valueNeurons) - 1 - layer].items():
                weightGradient[layer][outputKey] = {}

                if outputKey not in biasGradient[layer]:
                    biasGradient[layer][outputKey] = 0

                desiredOutput = calcDesiredOutput(outputKey, number)
                activation = activationGradient[layer - 1][outputKey]

                biasGradient[layer][outputKey] += biasDerivative(activation, output)

                for nKey, n in valueNeurons[len(valueNeurons) - 2 - layer].items():
                    weight = weightsAndBiases[len(valueNeurons) - 1 - layer][outputKey]["weights"][nKey]

                    weightGradient[layer][outputKey][nKey] = weightDerivative(activation, output, n)
                    if layer != len(valueNeurons) - 2:

                        if nKey not in activationGradient[layer]:
                            activationGradient[layer][nKey] = 0

                        activationGradient[layer][nKey] += activationDerivative(activation, output, weight)
    return weightGradient, biasGradient

def calcDesiredOutput(outputNumber, number):
    if outputNumber == number:
        return 1
    else:
        return 0
    
def weightDerivativeWithCost(output, n, desiredOutput):
    return 2 * (output - desiredOutput) * (output * (1 - output)) * n

def biasDerivativeWithCost(output, desiredOutput):
    return 2 * (output - desiredOutput) * (output * (1 - output))

def activationDerivativeWithCost(output, desiredOutput, weight):
    return 2 * (output - desiredOutput) * (output * (1 - output)) * weight

def weightDerivative(activation, output, n):
    return activation * (output * (1 - output)) * n

def biasDerivative(activation, output):
    return activation * (output * (1 - output))

def activationDerivative(activation, output, weight):
    return activation * (output * (1 - output)) * weight

def applyGradient(weightGradient, biasGradient, weightsAndBiases, learningRate):
    for layerKey, layer in weightsAndBiases.items():
        for neuronKey, neuron in layer.items():
            neuron["bias"] = mathFunctions.changeWeightOrBias(neuron["bias"], learningRate, biasGradient[len(biasGradient) - layerKey][neuronKey])
            i = 0
            for weight in neuron["weights"]:
                weight = mathFunctions.changeWeightOrBias(weight, learningRate, weightGradient[len(weightGradient) - layerKey][neuronKey][i])
                i += 1
    return weightsAndBiases

main(weightsAndBiases)