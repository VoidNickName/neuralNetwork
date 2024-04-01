import createWeightsAndBiases
import readImage
from runNeuralNetwork import runNeuralNetwork

file = "weightsAndBiases.pkl"
#file = "tinyWeightsAndBiases.pkl"
imagePath = "testSet\img_2.jpg"

# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)
# Get list of pixel values from image
imgPixelList = readImage.imgPxList(imagePath)
#imgPixelList = [0, 1]

valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)

output = (valueNeurons[len(valueNeurons) - 1])
print(output)
# Get the neuron with the highest value
Keymax = max(zip(output.values(), output.keys()))[1]
print(Keymax)