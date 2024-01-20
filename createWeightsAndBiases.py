import pickle
import os
import random
import mathFunctions

inputLayer = 784
hiddenLayers = {
    1: 16,
    2: 16,
    # Output layer
    3: 10
}

file = "weightsAndBiases.pkl"
weightsAndBiases = {}
# Create a nested dictionarie with the weights and baises
def setWeightsAndBiases(inputLayer, hiddenLayers):
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
            setWeightsAndBiases(inputLayer, hiddenLayers)
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
        return data

#print(returnFileData(file))    
#print(createWeightsAndBiasesFile(inputLayer, hiddenLayers, file))