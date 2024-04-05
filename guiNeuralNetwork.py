# Import Module
from tkinter import *

# import messagebox from tkinter module 
import tkinter.messagebox 

import createWeightsAndBiases
from runNeuralNetwork import runNeuralNetwork

def main():

    settings = createWeightsAndBiases.returnJsonFileData()
    canvasHight = settings["canvasHight"]
    canvasWidth = settings["canvasWidth"]
    pixelSize = settings["pixelSize"]

    global canvasPixel
    canvasPixel = {}

    # create root window
    root = Tk()
    # Set the background color of the window
    root.configure(background='gray')
    global canvas
    # Configure the size of the canvas
    canvas = Canvas(root, width=canvasWidth * pixelSize + 2 * pixelSize, height=canvasHight * pixelSize + 2 * pixelSize, bd=-2)
    # Set the background color of the canvas
    canvas.configure(background='gray')

    # root window title and dimension
    root.title("Neural Network")

    # Loop through the canvas and create a canvas full of pixels
    for x in range(canvasWidth):
        canvasPixel[x] = {}
        for y in range(canvasHight):
            canvasPixel[x][y] = canvas.create_rectangle((pixelSize + 1 + x * pixelSize),(pixelSize + 1 + y * pixelSize),(pixelSize + 1 + x * pixelSize + pixelSize),(pixelSize + 1 + y * pixelSize + pixelSize),fill="black", width=0)

    # Sets a continues check for a button press and mouse movement
    canvas.bind("<B1-Motion>", callback)
    canvas.bind("<Button-1>", callback)
    canvas.grid(column=0, row=2)

    # Create a reset button to clear the canvas
    btn1 = Button(root, text="Reset canvas", command=lambda: reset(canvasWidth, canvasHight, canvas, canvasPixel))
    btn1.grid(column=0, row=3)

    # Create a button to run the neural network on the drawn picture
    btn1 = Button(root, text="Run Neural Network", command=lambda: run(canvasWidth, canvasHight, canvas, canvasPixel))
    btn1.grid(column=0, row=4, pady=pixelSize)

    # Execute Tkinter
    root.mainloop()

def callback(event):
    settings = createWeightsAndBiases.returnJsonFileData()
    canvasHight = settings["canvasHight"]
    canvasWidth = settings["canvasWidth"]
    pixelSize = settings["pixelSize"]

    # Colors the pixel where the mouse is at at the moment
    if 10 <= event.x <= (canvasWidth * pixelSize + pixelSize) and 10 <= event.y <= (canvasHight * pixelSize + pixelSize):
        x = int((event.x - pixelSize - 1) / pixelSize)
        y = int((event.y - pixelSize - 1) / pixelSize)
        canvas.itemconfig(canvasPixel[x][y], fill="white")

def reset(canvasWidth, canvasHight, canvas, canvasPixel):
    # Makes the intire canvas black again
    for x in range(canvasWidth):
        for y in range(canvasHight):
            canvas.itemconfig(canvasPixel[x][y], fill="black")

def run(canvasWidth, canvasHight, canvas, canvasPixel):
    # Get weights and biases
    weightsAndBiases = createWeightsAndBiases.getWeightsAndBiases()

    # Creates a list of the pixels of the canvas
    pixelColor = []
    for x in range(canvasWidth):
        for y in range(canvasHight):
            color = canvas.itemcget(canvasPixel[x][y], "fill")
            if color == "black":
                pixelColor.append(0)
            else:
                pixelColor.append(1)

    # Runs the list through the neural network
    valueNeurons = runNeuralNetwork(weightsAndBiases, pixelColor)

    output = (valueNeurons[len(valueNeurons) - 1])
    # Get the neuron with the highest value
    Keymax = max(zip(output.values(), output.keys()))[1]
    tkinter.messagebox.showinfo("Output", Keymax) 
    print(Keymax)