import createWeightsAndBiases
import mathFunctions
import readImage

file = "weightsAndBiases.pkl"
imagePath = "testSet\img_1.jpg"

# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)
# Get list of pixel values from image
imgPixelList = readImage.imgPxList(imagePath)


def main(weightsAndBiases, imgPixelList):
    output = runNeuralNetwork(weightsAndBiases, imgPixelList)
    print(output)
    # Get the neuron with the highest value
    Keymax = max(zip(output.values(), output.keys()))[1]
    print(Keymax)

def runNeuralNetwork(weightsAndBiases, imgPixelList):
    valueNeurons = {}
    # Loop through layers
    for layerKey, layer in weightsAndBiases.items():
        valueNeurons[layerKey] = {}
        # Loop through the neurons of the layer
        for neuron, weightsAndBias in layer.items():
            valueNeuron = 0
            i = 0
            # Loop through weights of the connected neurons
            for weight in weightsAndBias["weights"]:
                if layerKey == 1:
                    # Add the input of a neuron of the previouse layer to the current neurons value
                    valueNeuron += weight * imgPixelList[i]
                else:
                    # Add the input of a neuron of the previouse layer to the current neurons value
                    valueNeuron += weight * valueNeurons[layerKey - 1][i]
                i += 1
            # Subtract the bias of the curent neuron of it's value
            valueNeuron -= weightsAndBias["bias"]
            # Use the sigmoid function to tranform the range of the value of the neuron to a number between 0 and 1.
            valueNeurons[layerKey][neuron] = mathFunctions.sigmoidFunction(valueNeuron)
    # Return the list of values of the neurons of the outputlayer
    return valueNeurons[len(valueNeurons)]

main(weightsAndBiases, imgPixelList)