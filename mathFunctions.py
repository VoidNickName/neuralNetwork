# Import math Library
import math

def sigmoidFunction(x):
    x = float(x)
    return (1 / (1 + (math.e ** (-x))))

def derivativeSigmoidFunction(x):
    return x * (1 - x)

# Xavier Weight Initialization
def XavierWeightInitialization(n):
    weightRange = {
        "min": -(1/math.sqrt(n)),
        "max": (1/math.sqrt(n))
    }
    return weightRange

def calcCost(output, desiredOutput):
    return (output - desiredOutput) ** 2

def calcCostDerivative(output, desiredOutput):
    return 2 * (output - desiredOutput)

def calcDelta(x, output):
    return x * derivativeSigmoidFunction(output)

def weightDerivative(delta, n):
    return delta * n

def activationDerivative(delta, weight):
    return delta * weight

def changeWeightOrBias(currentEstimate, constant, gradient):
    return (currentEstimate - (constant * gradient))