# Import math Library
import math

def sigmoidFunction(x):
    x = float(x)
    return (1 / (1 + (math.e ** (-x))))

def derivativeSigmoidFunction(x):
    return sigmoidFunction(x) * (1 - sigmoidFunction(x))

#number = -100
#print(sigmoidFunction(number))


# Xavier Weight Initialization
def XavierWeightInitialization(n):
    weightRange = {
        "min": -(1/math.sqrt(n)),
        "max": (1/math.sqrt(n))
    }
    return weightRange

def changeWeightOrBias(currentEstimate, constant, gradient):
    return (currentEstimate - constant * gradient)