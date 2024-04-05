import mathFunctions

def runNeuralNetwork(weightsAndBiases, imgPixelList):
    valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)
    return valueNeurons

def runNeuralNetwork(weightsAndBiases, imgPixelList):
    valueNeurons = {}
    valueNeurons[0] = {}

    # This is a list and needs to be a dictonary
    for i, pixelValue in enumerate(imgPixelList):
        valueNeurons[0][i] = pixelValue
        
    # Loop through layers
    for layerKey, layer in weightsAndBiases.items():
        valueNeurons[layerKey] = {}
        # Loop through the neurons of the layer
        for neuron, weightsAndBias in layer.items():
            valueNeuron = 0

            # Loop through weights of the connected neurons
            for i, weight in enumerate(weightsAndBias["weights"]):
                # Add the input of a neuron of the previouse layer to the current neurons value
                valueNeuron += weight * valueNeurons[layerKey - 1][i]

            # Add the bias of the curent neuron of it's value
            valueNeuron += weightsAndBias["bias"]
            
            # Use the sigmoid function to tranform the range of the value of the neuron to a number between 0 and 1.
            valueNeurons[layerKey][neuron] = mathFunctions.sigmoidFunction(valueNeuron)
    
    # Return the list of values of the neurons of the outputlayer
    return valueNeurons