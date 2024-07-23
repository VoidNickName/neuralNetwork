import createWeightsAndBiases
import readImage
from runNeuralNetwork import runNeuralNetwork
import sys
import stochasticGradientDescent
import guiNeuralNetwork

settings = createWeightsAndBiases.returnJsonFileData()
file = settings["weightsAndBiasesFile"]
imagePath = settings["imagePath"]
costListFile = settings["costListFile"]

def main():
    arg = sys.argv

    if len(arg) < 2 or arg[1] == "help":
        help()
    elif arg[1] == "train":
        stochasticGradientDescent.main()
    elif arg[1] == "gui":
        guiNeuralNetwork.main()
    elif arg[1] == "graph":
        try:
            stochasticGradientDescent.showGraph(costListFile)
        except FileNotFoundError:
            print("")
            print(" ========================================ERROR=========================================")
            print("  |                                                                                   |")
            print("  |  The Neural Network has not been trained yet.                                     |")
            print("  |  In order to see the graph of the training the network first has to be trained.   |")
            print("  |  The neural network can be trained by running the following command:              |")
            print("  |        > python neuralNetwork.py train                                            |")
            print("  |                                                                                   |")
            print(" ======================================================================================")
            print("")
    elif arg[1] == "run":
        try: 
            if len(arg) > 2:
                # Get list of pixel values from image
                imgPixelList = readImage.imgPxList(arg[2])
            else:
                # Get list of pixel values from image
                imgPixelList = readImage.imgPxList(imagePath)

            # Retreve or create weights and biases
            weightsAndBiases = createWeightsAndBiases.getWeightsAndBiases()

            # Run the neural network
            valueNeurons = runNeuralNetwork(weightsAndBiases, imgPixelList)

            output = (valueNeurons[len(valueNeurons) - 1])
            # Get the neuron with the highest value
            Keymax = max(zip(output.values(), output.keys()))[1]
            print(Keymax)
        except FileNotFoundError:
            print("File not Found")
    else:
        help()

def help():
    print("")
    print(" =======================================HELP===========================================")
    print("  |                                                                                   |")
    print("  |  In order to use this neural network you can run the following commands:          |")
    print("  |                                                                                   |")
    print("\033[1m" + "  |  - help                                                                           |" + "\033[0m")
    print("  |  This command is used to get this message.                                        |")
    print("  |                                                                                   |")
    print("\033[1m" + "  |  - train                                                                          |" + "\033[0m")
    print("  |  With this command you can train the neural network.                              |")
    print("  |                                                                                   |")
    print("\033[1m" + "  |  - gui                                                                            |" + "\033[0m")
    print("  |  This command is used to get a graphical user interface.                          |")
    print("  |  With this gui you can draw your own images                                       |")
    print("  |  and let the neural network recognise them.                                       |")
    print("  |  This command should only be used after the neural network has been trained.      |")
    print("  |                                                                                   |")
    print("\033[1m" + "  |  - run                                                                            |" + "\033[0m")
    print("  |  With this command you can run the neural network.                                |")
    print("  |  This command has a second optional command.                                      |")
    print("  |  If you want to run another image through the neural network                      |")
    print("  |  then the one specified in the settings,                                          |")
    print("  |  you can run te following command:                                                |")
    print("  |        > python neuralNetwork.py run <filepath>                                   |")
    print("  |                                                                                   |")
    print(" ======================================================================================")
    print("")
main()