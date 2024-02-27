import createWeightsAndBiases
import readImage
from runNeuralNetwork import runNeuralNetwork

file = "weightsAndBiases.pkl"
imagePath = "testSet\img_1.jpg"

# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)
# Get list of pixel values from image
imgPixelList = readImage.imgPxList(imagePath)

valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)

output = (valueNeurons[len(valueNeurons)])
print(output)
# Get the neuron with the highest value
Keymax = max(zip(output.values(), output.keys()))[1]
print(Keymax)