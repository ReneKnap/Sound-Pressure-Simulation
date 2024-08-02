import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


posStepSize = 0.05
timeStepSize = 0.00005
SPEED_OF_SOUND = 346.3  # speed of sound in air in m/s at 25°C

roomWidth = 5.0  # in meters
roomHeight = 3.6   # in meters

numDiscretePosX = int(roomWidth / posStepSize) + 1
numDiscretePosY = int(roomHeight / posStepSize) + 1

pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))

speakerSize = 0.3  # in meters
speakerPosX = numDiscretePosY // 4
speakerPosY = numDiscretePosX // 2
frequency = 500  # in Hz
omega = 2 * np.pi * frequency

animRunning = True
currentPhase = 0 

dpi = 100 
figWidth = 400 / dpi
figHeight = 300 / dpi

root = tk.Tk()
root.title('Sound Pressure Simulation')
root.geometry("1050x900+10+10")

frameAnimation = tk.Frame(root, width=400, height=300)
frameAnimation.pack(side=tk.TOP, pady=10)

fig, ax = plt.subplots(figsize=(figWidth, figHeight), dpi=dpi)
image = ax.imshow(pressureField, cmap='viridis', vmin=-0.1, vmax=0.1, animated=True)
ax.set_title('Sound Pressure Simulation')
ax.set_xlabel('X in meter')
ax.set_ylabel('Y in meter')

xTicks = np.arange(0, numDiscretePosX, int(1 / posStepSize))
xLabels = np.round(xTicks * posStepSize, 1)
ax.set_xticks(xTicks)
ax.set_xticklabels(xLabels)

yTicks = np.arange(0, numDiscretePosY, int(1 / posStepSize))
yLabels = np.round(yTicks * posStepSize, 1)
ax.set_yticks(yTicks)
ax.set_yticklabels(yLabels)

speakerCircle = Circle(
    (speakerPosY, speakerPosX), radius=speakerSize / posStepSize / 2, color='orange', fill=False, linewidth=2)
ax.add_patch(speakerCircle)

canvas = FigureCanvasTkAgg(fig, master=frameAnimation)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.NONE, expand=False)
canvas.get_tk_widget().pack_propagate(0)
canvas.get_tk_widget().config(width=1000, height=750)

frameControls = tk.Frame(root)
frameControls.pack(side=tk.TOP, pady=10)

frequencyLabel = tk.Label(frameControls, text="Frequency (Hz)")
frequencyLabel.pack(side=tk.LEFT, padx=5)

frequencySlider = tk.Scale(frameControls, from_=10, to=1000, orient=tk.HORIZONTAL, length=200)
frequencySlider.set(frequency)
frequencySlider.pack(side=tk.LEFT, padx=5)

resetButton = tk.Button(frameControls, text="Reset", command=lambda: reset(None))
resetButton.pack(side=tk.LEFT, padx=15)

stopButton = tk.Button(frameControls, text="Stop", command=lambda: stop(None))
stopButton.pack(side=tk.LEFT, padx=15)


def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, speakerSize, currentPhase):
    # update velocity fields
    velocityFieldX[1:, :] -= timeStepSize / posStepSize * (pressureField[1:, :] - pressureField[:-1, :])
    velocityFieldY[:, 1:] -= timeStepSize / posStepSize * (pressureField[:, 1:] - pressureField[:, :-1])

    # reflective boundary conditions
    velocityFieldX[0, :] = 0
    velocityFieldX[-1, :] = 0
    velocityFieldY[:, 0] = 0
    velocityFieldY[:, -1] = 0

    pressureField[:-1, :] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldX[1:, :] - velocityFieldX[:-1, :])
    pressureField[:, :-1] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldY[:, 1:] - velocityFieldY[:, :-1])

    # simple sound source
    updateSpeakerPressure(pressureField, speakerPosX, speakerPosY, speakerSize, currentPhase)
    currentPhase += omega * timeStepSize
    currentPhase %= 2 * np.pi

    return pressureField, velocityFieldX, velocityFieldY, currentPhase

def updateSpeakerPressure(pressureField, centerX, centerY, size, phase):
    radius = int(size / posStepSize / 2)
    for y in range(centerX - radius, centerX + radius):
        for x in range(centerY - radius, centerY + radius):
            if 0 <= x < numDiscretePosX and 0 <= y < numDiscretePosY:
                if (x - centerY)**2 + (y - centerX)**2 <= radius**2:
                    pressureField[y, x] += np.sin(phase)

def update(frame):
    global pressureField, velocityFieldX, velocityFieldY, animRunning, speakerSize, currentPhase

    if animRunning:
        pressureField, velocityFieldX, velocityFieldY, currentPhase = calcFiniteDifferenceTimeDomain(
            pressureField, velocityFieldX, velocityFieldY, speakerSize, currentPhase)

        image.set_array(pressureField[:-1, :-1])

        # draw Speaker
        speakerCircle.set_radius(speakerSize / posStepSize / 2)
        ax.add_patch(speakerCircle)
    return [image, speakerCircle]

def updateFrequency(val):
    global frequency, omega
    frequency = frequencySlider.get()
    omega = 2 * np.pi * frequency

def reset(event):
    global pressureField, velocityFieldX, velocityFieldY, currentPhase
    pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
    velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
    velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))
    currentPhase = 0

def stop(event):
    global animRunning
    animRunning = not animRunning
    if animRunning:
        stopButton.config(text='Stop')
    else:
        stopButton.config(text='Start')


frequencySlider.bind("<Motion>", updateFrequency) 

animation = FuncAnimation(fig, update, blit=True, interval=1, cache_frame_data=False)

root.mainloop()
