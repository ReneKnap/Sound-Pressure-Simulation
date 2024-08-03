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
REFERENCE_PRESSURE = 20e-6  # reference pressure in Pascals for 0 dB

roomWidth = 5.0  # in meters
roomHeight = 3.6   # in meters
wallThickness = 0.2

# room size plus 2 times wall thickness
numDiscretePosX = int(roomWidth / posStepSize) + 1 + int(wallThickness / posStepSize) * 2
numDiscretePosY = int(roomHeight / posStepSize) + 1 + int(wallThickness / posStepSize) * 2

pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))

speakerSize = 0.3  # in meters
speakerPosX = numDiscretePosY // 2
speakerPosY = numDiscretePosX // 2
frequency = 500  # in Hz
omega = 2 * np.pi * frequency
volume = 60  # in dB

wallReflectionCoefficient = 0.8 # Proportion of reflection (0.0 to 1.0)

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

volumeLabel = tk.Label(frameControls, text="Volume (dB)")
volumeLabel.pack(side=tk.LEFT, padx=5)

volumeSlider = tk.Scale(frameControls, from_=0, to=120, orient=tk.HORIZONTAL, length=200)
volumeSlider.set(volume) 
volumeSlider.pack(side=tk.LEFT, padx=5)

resetButton = tk.Button(frameControls, text="Reset", command=lambda: reset(None))
resetButton.pack(side=tk.LEFT, padx=15)

stopButton = tk.Button(frameControls, text="Stop", command=lambda: stop(None))
stopButton.pack(side=tk.LEFT, padx=15)


def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, speakerSize, volume, currentPhase):
    # update velocity fields
    velocityFieldX[1:, :] -= timeStepSize / posStepSize * (pressureField[1:, :] - pressureField[:-1, :])
    velocityFieldY[:, 1:] -= timeStepSize / posStepSize * (pressureField[:, 1:] - pressureField[:, :-1])

    pressureField[:-1, :] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldX[1:, :] - velocityFieldX[:-1, :])
    pressureField[:, :-1] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldY[:, 1:] - velocityFieldY[:, :-1])

    applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY)

    # simple sound source
    updateSpeakerPressure(pressureField, speakerPosX, speakerPosY, speakerSize, volume, currentPhase)
    currentPhase += omega * timeStepSize
    currentPhase %= 2 * np.pi

    return pressureField, velocityFieldX, velocityFieldY, currentPhase


def applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY):
    absorptionCoefficient = 1.0 - wallReflectionCoefficient
    for i in range(int(wallThickness / posStepSize)):

        pressureField[i, :] *= absorptionCoefficient
        pressureField[-i-2, :] *= absorptionCoefficient
        pressureField[:, i] *= absorptionCoefficient
        pressureField[:, -i-2] *= absorptionCoefficient

    velocityFieldX[0, :] = 0
    velocityFieldX[-1, :] = 0
    velocityFieldY[:, 0] = 0
    velocityFieldY[:, -1] = 0

def updateSpeakerPressure(pressureField, centerX, centerY, size, volume, phase):
    amplitude = REFERENCE_PRESSURE * (10 ** (volume / 20))

    radius = size / (2 * posStepSize)
    for y in range(centerX - int(radius) - 0, centerX + int(radius) + 0):
        for x in range(centerY - int(radius) - 0, centerY + int(radius) + 0):
            if 0 <= x < numDiscretePosX and 0 <= y < numDiscretePosY:
                distance = np.sqrt((x - centerY) ** 2 + (y - centerX) ** 2)
                if distance <= radius:
                    pressureField[y, x] += amplitude * np.sin(phase)

def update(frame):
    global pressureField, velocityFieldX, velocityFieldY, animRunning, speakerSize, volume, currentPhase

    if animRunning:
        pressureField, velocityFieldX, velocityFieldY, currentPhase = calcFiniteDifferenceTimeDomain(
            pressureField, velocityFieldX, velocityFieldY, speakerSize, volume, currentPhase)

        image.set_array(pressureField[:-1, :-1])

        # draw Speaker
        speakerCircle.set_radius(speakerSize / posStepSize / 2)
        ax.add_patch(speakerCircle)
    return [image, speakerCircle]

def updateFrequency(val):
    global frequency, omega
    frequency = frequencySlider.get()
    omega = 2 * np.pi * frequency

def updateVolume(val):
    global volume
    volume = volumeSlider.get()

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


frequencySlider.bind("<B1-Motion>", updateFrequency) 
volumeSlider.bind("<B1-Motion>", updateVolume) 

animation = FuncAnimation(fig, update, blit=True, interval=1, cache_frame_data=False)

root.mainloop()
