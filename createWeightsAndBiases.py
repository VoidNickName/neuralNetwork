import pickle
import os
import random
import mathFunctions
import json

# Create a nested dictionarie with the weights and biases
def setWeightsAndBiases(inputLayer, hiddenLayers):
    weightsAndBiases = {}
        # Loop through the layers in the neuralnetwork
    for layerKey, neurons in hiddenLayers.items():
        weightsAndBiases[layerKey] = {}
            # Loop through the neurons in the layer
        for neuron in range(neurons):
            weightsAndBiases[layerKey][neuron] = {}
                # Create bias for the neuron
            weightsAndBiases[layerKey][neuron]["bias"] = 0
            weightsAndBiases[layerKey][neuron]["weights"] = []
            if layerKey == 1:
                    # Define weight range with Xavier Weight Initialization
                weightRange = mathFunctions.XavierWeightInitialization(inputLayer)
                    # Create the weights for all the connected input neurons of this neuron
                for _ in range(inputLayer):
                    # Create random weight within given weight range
                    weight = random.uniform(weightRange["min"], weightRange["max"])
                    weightsAndBiases[layerKey][neuron]["weights"].append(weight)
            else:
                # Define weight range with Xavier Weight Initialization
                weightRange = mathFunctions.XavierWeightInitialization(hiddenLayers[layerKey - 1])
                    # Create the weights for all the connected neurons of this neuron
                for _ in range(hiddenLayers[layerKey - 1]):
                    # Create random weight within given weight range
                    weight = random.uniform(weightRange["min"], weightRange["max"])
                    weightsAndBiases[layerKey][neuron]["weights"].append(weight)
    return weightsAndBiases

def createWeightsAndBiasesFile(inputLayer, hiddenLayers, file):
    if os.path.isfile(file) == False:
        # open a file
        with open(file, 'wb') as f:
            weightsAndBiases = setWeightsAndBiases(inputLayer, hiddenLayers)
            # serialize the weights and biases to the file
            pickle.dump(weightsAndBiases, f)
            f.close()
        return True
    else:
        return False

def returnFileData(file):
    with open(file, 'rb') as f:
        # deserialize using load()
        data = pickle.load(f)
        f.close()
        return data
    
    
def alterFileData(file, newData):
    if os.path.isfile(file) == True:
        # open a file
        with open(file, 'wb') as f:
            # serialize the data to the file
            pickle.dump(newData, f)
            f.close()
        return True
    else:
        return False
    
def createFile(file, data):
    if os.path.isfile(file) == False:
        # open a file
        with open(file, 'wb') as f:
            # serialize the data to the file
            pickle.dump(data, f)
            f.close()
        return True
    else:
        return False
    
def returnJsonFileData():
    with open("settings.json", 'rb') as f:
        # deserialize using load()
        data = json.load(f)
        f.close()
        return data
    
def getWeightsAndBiases():
    settings = returnJsonFileData()
    canvasHight = settings["canvasHight"]
    canvasWidth = settings["canvasWidth"]
    file = settings["weightsAndBiasesFile"]

    inputLayer = canvasWidth * canvasHight
    hiddenLayers = {}
    for layerKey, layer in settings["hiddenLayers"].items():
        hiddenLayers[int(layerKey)] = layer

    # Retreve weights and biases from file
    # When the file does not exist create a new one
    try:
        weightsAndBiases = returnFileData(file)
    except FileNotFoundError:
        createWeightsAndBiasesFile(inputLayer, hiddenLayers, file)
        weightsAndBiases = returnFileData(file)
    
    return weightsAndBiases