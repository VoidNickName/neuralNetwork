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
    file = settings["weightsAndBiasesFile"]

    global canvasPixel
    canvasPixel = {}

    # create root window
    root = Tk()
    root.configure(background='gray')
    global canvas
    canvas = Canvas(root, width=canvasWidth * pixelSize + 2 * pixelSize, height=canvasHight * pixelSize + 2 * pixelSize, bd=-2)
    canvas.configure(background='gray')

    # root window title and dimension
    root.title("Neural Network")

    for x in range(canvasWidth):
        canvasPixel[x] = {}
        for y in range(canvasHight):
            canvasPixel[x][y] = canvas.create_rectangle((pixelSize + 1 + x * pixelSize),(pixelSize + 1 + y * pixelSize),(pixelSize + 1 + x * pixelSize + pixelSize),(pixelSize + 1 + y * pixelSize + pixelSize),fill="black", width=0)

    canvas.bind("<B1-Motion>", callback)
    canvas.bind("<Button-1>", callback)
    canvas.grid(column=0, row=2)

    btn1 = Button(root, text="Reset canvas", command=lambda: reset(canvasWidth, canvasHight, canvas, canvasPixel))
    btn1.grid(column=0, row=3)

    btn1 = Button(root, text="Run Neural Network", command=lambda: run(canvasWidth, canvasHight, canvas, canvasPixel, file))
    btn1.grid(column=0, row=4, pady=pixelSize)

    # Execute Tkinter
    root.mainloop()

def callback(event):
    settings = createWeightsAndBiases.returnJsonFileData()
    canvasHight = settings["canvasHight"]
    canvasWidth = settings["canvasWidth"]
    pixelSize = settings["pixelSize"]

    if 10 <= event.x <= (canvasWidth * pixelSize + pixelSize) and 10 <= event.y <= (canvasHight * pixelSize + pixelSize):
        x = int((event.x - pixelSize - 1) / pixelSize)
        y = int((event.y - pixelSize - 1) / pixelSize)
        canvas.itemconfig(canvasPixel[x][y], fill="white")

def reset(canvasWidth, canvasHight, canvas, canvasPixel):
    for x in range(canvasWidth):
        for y in range(canvasHight):
            canvas.itemconfig(canvasPixel[x][y], fill="black")

def run(canvasWidth, canvasHight, canvas, canvasPixel, file):
    # Retreve weights and biases from file
    weightsAndBiases = createWeightsAndBiases.returnFileData(file)

    pixelColor = []
    for x in range(canvasWidth):
        for y in range(canvasHight):
            color = canvas.itemcget(canvasPixel[x][y], "fill")
            if color == "black":
                pixelColor.append(0)
            else:
                pixelColor.append(1)

    valueNeurons = runNeuralNetwork(weightsAndBiases, pixelColor)

    output = (valueNeurons[len(valueNeurons) - 1])
    print(output)
    # Get the neuron with the highest value
    Keymax = max(zip(output.values(), output.keys()))[1]
    tkinter.messagebox.showinfo("Output", Keymax) 
    print(Keymax)

main()