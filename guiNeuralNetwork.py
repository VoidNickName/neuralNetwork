# Import Module
from tkinter import *

# import messagebox from tkinter module 
import tkinter.messagebox 

import createWeightsAndBiases
from runNeuralNetwork import runNeuralNetwork

file = "weightsAndBiases.pkl"

# Retreve weights and biases from file
weightsAndBiases = createWeightsAndBiases.returnFileData(file)

canvasHight = 28
canvasWith = 28
pixelSize = 15

canvasPixel = {}

# create root window
root = Tk()
root.configure(background='gray')
canvas = Canvas(root, width=canvasWith * pixelSize + 2 * pixelSize, height=canvasHight * pixelSize + 2 * pixelSize, bd=-2)
canvas.configure(background='gray')

# root window title and dimension
root.title("Neural Network")

def key(event):
    print("pressed", repr(event.char))

def callback(event):
    if 10 <= event.x <= (canvasWith * pixelSize + pixelSize) and 10 <= event.y <= (canvasHight * pixelSize + pixelSize):
        x = int((event.x - pixelSize - 1) / pixelSize)
        y = int((event.y - pixelSize - 1) / pixelSize)
        canvas.itemconfig(canvasPixel[x][y], fill="white")

def reset():
    for x in range(canvasWith):
        for y in range(canvasHight):
            canvas.itemconfig(canvasPixel[x][y], fill="black")

def run():
    pixelColor = []
    for x in range(canvasWith):
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

for x in range(canvasWith):
    canvasPixel[x] = {}
    for y in range(canvasHight):
        canvasPixel[x][y] = canvas.create_rectangle((pixelSize + 1 + x * pixelSize),(pixelSize + 1 + y * pixelSize),(pixelSize + 1 + x * pixelSize + pixelSize),(pixelSize + 1 + y * pixelSize + pixelSize),fill="black", width=0)

canvas.bind("<Key>", key)
canvas.bind("<B1-Motion>", callback)
canvas.bind("<Button-1>", callback)
canvas.grid(column=0, row=2)

btn1 = Button(root, text="Reset canvas", command=reset)
btn1.grid(column=0, row=3)

btn1 = Button(root, text="Run Neural Network", command=run)
btn1.grid(column=0, row=4, pady=pixelSize)

# Execute Tkinter
root.mainloop()