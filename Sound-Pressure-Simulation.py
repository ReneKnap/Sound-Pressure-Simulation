import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import NamedTuple

class Position(NamedTuple):
    x: float
    y: float

class Size(NamedTuple):
    width: float
    height: float

posStepSize = 0.05
SPEED_OF_SOUND = 346.3  # Speed of sound in air in m/s at 25°C
timeStepSize = 0.1 * posStepSize / SPEED_OF_SOUND
REFERENCE_PRESSURE = 20e-6  # Reference pressure in Pascals for 0 dB

roomWidth = 5.0  # m
roomHeight = 3.6   # m
wallThickness = 0.2

# Room size plus 2 times wall thickness
numDiscretePosX = int(roomWidth / posStepSize) + 1 + int(wallThickness / posStepSize) * 2
numDiscretePosY = int(roomHeight / posStepSize) + 1 + int(wallThickness / posStepSize) * 2

pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))

speakerRadius = 0.3  # m
speakerPos = Position(0.5, 0.5)  # m
speakerFrequency = 30  # Hz
speakerVolume = 70  # dB
omega = 2 * np.pi * speakerFrequency

wallReflectionCoefficient = 0.8 # Proportion of reflection (0.0 to 1.0)
wallPressureAbsorptionCoefficient = 0.2 # Proportion of pressure absorption (0.0 to 1.0)
wallVelocityAbsorptionCoefficient = 0.2 # Proportion of velocity absorption (0.0 to 1.0)

animRunning = True
simulatedTime = 0.0  # ms
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
image = ax.imshow(pressureField, cmap='viridis', vmin=-0.0025, vmax=0.0025, animated=True)
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

wallColor = 'gray'
wallRects = [
    Rectangle((-0.2, -0.2), wallThickness/posStepSize-0.4, numDiscretePosY-1+0.4, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((numDiscretePosX-wallThickness/posStepSize-1+0.4, -0.2), wallThickness/posStepSize-0.2, numDiscretePosY-1+0.4, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((-0.2, -0.2), numDiscretePosX-1+0.4, wallThickness/posStepSize-0.2, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((-0.2, numDiscretePosY-wallThickness/posStepSize-1+0.4), numDiscretePosX-1+0.4, wallThickness/posStepSize-0.2, color=wallColor, fill=False, linewidth=2, hatch='////')
]

for wall in wallRects:
    ax.add_patch(wall)

speakerCircle = Circle(
    (speakerPos.x / posStepSize , speakerPos.y / posStepSize), radius=speakerRadius / posStepSize / 2, color='orange', fill=False, linewidth=2)
ax.add_patch(speakerCircle)

timeAnnotation = ax.text(0.05, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top')

cbar = fig.colorbar(image, ax=ax, orientation='vertical')
cbar.set_label('Druck (Pa)')

canvas = FigureCanvasTkAgg(fig, master=frameAnimation)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.NONE, expand=False)
canvas.get_tk_widget().pack_propagate(0)
canvas.get_tk_widget().config(width=1000, height=750)

frameControls = tk.Frame(root)
frameControls.pack(side=tk.TOP, pady=10)

speakerFrequencyLabel = tk.Label(frameControls, text="Frequency (Hz)")
speakerFrequencyLabel.pack(side=tk.LEFT, padx=5)

speakerFrequencyEntry = tk.Entry(frameControls, width=5)
speakerFrequencyEntry.insert(0, str(speakerFrequency))
speakerFrequencyEntry.pack(side=tk.LEFT, padx=5)

speakerFrequencySlider = tk.Scale(frameControls, from_=10, to=1000, orient=tk.HORIZONTAL, length=200)
speakerFrequencySlider.set(speakerFrequency)
speakerFrequencySlider.pack(side=tk.LEFT, padx=5)

speakerVolumeLabel = tk.Label(frameControls, text="Volume (dB)")
speakerVolumeLabel.pack(side=tk.LEFT, padx=5)

speakerVolumeEntry = tk.Entry(frameControls, width=5)
speakerVolumeEntry.insert(0, str(speakerVolume))
speakerVolumeEntry.pack(side=tk.LEFT, padx=5)

speakerVolumeSlider = tk.Scale(frameControls, from_=0, to=120, orient=tk.HORIZONTAL, length=200)
speakerVolumeSlider.set(speakerVolume) 
speakerVolumeSlider.pack(side=tk.LEFT, padx=5)

resetButton = tk.Button(frameControls, text="Reset", command=lambda: resetSimulation(None))
resetButton.pack(side=tk.LEFT, padx=15)

stopButton = tk.Button(frameControls, text="Stop", command=lambda: toggleSimulation(None))
stopButton.pack(side=tk.LEFT, padx=15)

fieldTarget = tk.StringVar(root)
fieldTarget.set("Pressure")

fieldOptions = ["Pressure", "Velocity X", "Velocity Y"]
fieldMenu = tk.OptionMenu(frameControls, fieldTarget, *fieldOptions)
fieldMenu.pack(side=tk.LEFT, padx=15)


class Absorber:
    def __init__(self, position, size, absorptionPressure = 0.0, absorptionVelocity = 0.0):
        self.position = position
        self.size = size
        self.absorptionPressure = absorptionPressure
        self.absorptionVelocity = absorptionVelocity
        self.patch = None # Store the Rectangle object
        self.startX = int(position.x / posStepSize)
        self.endX = int((position.x + size.width) / posStepSize)
        self.startY = int(position.y / posStepSize)
        self.endY = int((position.y + size.height) / posStepSize)

    def applyAbsorption(self, pressureField, velocityFieldX, velocityFieldY):
        pressureField[self.startY:self.endY, self.startX:self.endX] *= (1 - self.absorptionPressure)
        velocityFieldX[self.startY:self.endY, self.startX:self.endX] *= (1 - self.absorptionVelocity)
        velocityFieldY[self.startY:self.endY, self.startX:self.endX] *= (1 - self.absorptionVelocity)

    def draw(self, ax):
        absorberRect = Rectangle((self.position.x / posStepSize, self.position.y / posStepSize),
                                  self.size.width / posStepSize, self.size.height / posStepSize,
                                  color='red', fill=False)
        ax.add_patch(absorberRect)
    
    def getPatch(self):
        if not self.patch:
            self.patch = Rectangle((self.position.x / posStepSize, self.position.y / posStepSize),
                                   self.size.width / posStepSize, self.size.height / posStepSize,
                                   color='red', fill=False)
        return self.patch

absorbers = [
    Absorber(Position(4.6, 0.2), Size(0.6, 0.6), absorptionPressure=0.0, absorptionVelocity=0.1),
    Absorber(Position(4.6, 3.2), Size(0.6, 0.6), absorptionPressure=0.0, absorptionVelocity=0.1)
]

absorberPatches = [ax.add_patch(absorber.getPatch()) for absorber in absorbers]

def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY):
    velocityFieldX[1:, :] -= timeStepSize / posStepSize * (pressureField[1:, :] - pressureField[:-1, :])
    velocityFieldY[:, 1:] -= timeStepSize / posStepSize * (pressureField[:, 1:] - pressureField[:, :-1])

    pressureField[:-1, :] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldX[1:, :] - velocityFieldX[:-1, :])
    pressureField[:, :-1] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldY[:, 1:] - velocityFieldY[:, :-1])


def applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY):
    for i in range(int(wallThickness / posStepSize)):
        pressureField[i, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[-i-2, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, i] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, -i-2] *= 1 - wallPressureAbsorptionCoefficient

        velocityFieldX[i, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[-i-1, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, i] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, -i-1] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[i, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[-i-1, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, i] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, -i-1] *= 1 - wallVelocityAbsorptionCoefficient

    wallReflexionLayer = int(wallThickness / posStepSize) - 1  
    velocityFieldX[wallReflexionLayer+1, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldX[-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, -wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient

    velocityFieldX[0, :] = 0
    velocityFieldX[-1, :] = 0
    velocityFieldY[:, 0] = 0
    velocityFieldY[:, -1] = 0

def updateSpeakerPressure(pressureField, phase):
    amplitude = REFERENCE_PRESSURE * (10 ** (speakerVolume / 20))
    radius = speakerRadius / (2 * posStepSize)

    for y in range(int(speakerPos.y / posStepSize  - radius), int(speakerPos.y / posStepSize + radius)):
        for x in range(int(speakerPos.x / posStepSize  - radius), int(speakerPos.x / posStepSize + radius)):
            if 0 <= x < numDiscretePosX and 0 <= y < numDiscretePosY:
                distance = np.sqrt((x - speakerPos.x / posStepSize ) ** 2 + (y - speakerPos.y / posStepSize ) ** 2)
                if distance <= radius:
                    pressureField[y, x] += amplitude * np.sin(phase)

def updateSimulation(pressureField, velocityFieldX, velocityFieldY, currentPhase):
    calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY)
    applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY)
    updateSpeakerPressure(pressureField, currentPhase)
    for absorber in absorbers:
        absorber.applyAbsorption(pressureField, velocityFieldX, velocityFieldY)


    currentPhase += omega * timeStepSize
    currentPhase %= 2 * np.pi

    return currentPhase

def updateDisplayedField():
    selectedField = fieldTarget.get()
    if selectedField == "Pressure":
        image.set_array(pressureField[:-1, :-1])
        image.set_clim(-0.25, 0.25)
        cbar.set_label('Pressure (Pa)')
    elif selectedField == "Velocity X":
        image.set_array(velocityFieldX[:-1, :-1])
        image.set_clim(-0.001, 0.001)
        cbar.set_label('Particle velocity in X (m/s)')
    elif selectedField == "Velocity Y":
        image.set_array(velocityFieldY[:-1, :-1])
        image.set_clim(-0.001, 0.001)
        cbar.set_label('Particle velocity in Y (m/s)')

def updateLegends(*args):
    canvas.draw_idle()

fieldTarget.trace_add("write", updateLegends)


def update(frame):
    global pressureField, velocityFieldX, velocityFieldY, animRunning, currentPhase, simulatedTime
    if animRunning:
        for _ in range(4):
            currentPhase = updateSimulation(pressureField, velocityFieldX, velocityFieldY, currentPhase)
            simulatedTime += timeStepSize * 1000

        timeAnnotation.set_text(f'Time: {simulatedTime:.2f} ms')


    updateDisplayedField()
    return [image, speakerCircle, timeAnnotation] + wallRects + absorberPatches

def updateFrequency(val):
    global speakerFrequency, omega
    speakerFrequency = speakerFrequencySlider.get()
    omega = 2 * np.pi * speakerFrequency
    speakerFrequencyEntry.delete(0, tk.END)
    speakerFrequencyEntry.insert(0, str(speakerFrequency))

def updateFrequencyFromEntry(event):
    try:
        val = speakerFrequencyEntry.get()
        if 10 <= val <= 1000:
            speakerFrequencySlider.set(val)
            updateFrequency(None)
    except ValueError:
        pass

def updateVolume(val):
    global speakerVolume
    speakerVolume = speakerVolumeSlider.get()
    speakerVolumeEntry.delete(0, tk.END)
    speakerVolumeEntry.insert(0, str(speakerVolume))


def updateVolumeFromEntry(event):
    try:
        val = speakerVolumeEntry.get()
        if 0 <= val <= 120:
            speakerVolumeSlider.set(val)
            updateVolume(None)
    except ValueError:
        pass

def resetSimulation(event):
    global pressureField, velocityFieldX, velocityFieldY, currentPhase
    pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
    velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
    velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))
    currentPhase = 0
    simulatedTime = 0.0

def toggleSimulation(event):
    global animRunning
    animRunning = not animRunning
    if animRunning:
        stopButton.config(text='Stop')
    else:
        stopButton.config(text='Start')


speakerFrequencySlider.bind("<B1-Motion>", updateFrequency) 
speakerVolumeSlider.bind("<B1-Motion>", updateVolume)
speakerFrequencyEntry.bind("<Return>", updateFrequencyFromEntry)
speakerVolumeEntry.bind("<Return>", updateVolumeFromEntry)

animation = FuncAnimation(fig, update, blit=True, interval=1, cache_frame_data=False)

root.mainloop()
