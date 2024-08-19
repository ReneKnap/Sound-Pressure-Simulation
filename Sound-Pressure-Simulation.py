import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
from matplotlib.path import Path as mplPath
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import NamedTuple
from enum import Enum
from matplotlib.colors import Normalize
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial import Delaunay, ConvexHull
import scipy.interpolate as interp
import scipy.signal as signal
from numba import njit
import threading
import time

class Position(NamedTuple):
    x: float
    y: float
    z: float

class Size(NamedTuple):
    length: float
    width: float
    height: float

class Axis(Enum):
    X = 1
    Y = 2
    Z = 3

posStepSize = np.float32(0.05)  # m
SPEED_OF_SOUND = np.float32(346.3)  # Speed of sound in air in m/s at 25°C
timeStepSize = np.float32(0.1 * posStepSize / SPEED_OF_SOUND)  # s
REFERENCE_PRESSURE = np.float32(20e-6)  # Reference pressure in Pascals for 0 dB
lowestFrequency = np.float32(20.0)  # Hz
highestFrequency = np.float32(2000.0)  # Hz

roomLength = np.float32(5.30)  # m #5.05
roomWidth = np.float32(3.55)   # m
roomHeight = np.float32(2.55)   # m

wallThickness = np.float32(0.2)  # m

numDiscretePosX = int(round(roomLength / posStepSize)) + int(round(wallThickness / posStepSize)) * 2
numDiscretePosY = int(round(roomWidth / posStepSize)) + int(round(wallThickness / posStepSize)) * 2
numDiscretePosZ = int(round(roomHeight / posStepSize)) + int(round(wallThickness / posStepSize)) * 2

pressureField = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
velocityFieldX = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
velocityFieldY = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
velocityFieldZ = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)

wallReflectionCoefficient = np.float32(0.47) # Proportion of reflection (0.0 to 1.0)
wallPressureAbsorptionCoefficient = np.float32(0.02) # Proportion of pressure absorption (0.0 to 1.0)
wallVelocityAbsorptionCoefficient = np.float32(0.94) # Proportion of velocity absorption (0.0 to 1.0)

animRunning = True
simulatedTime = np.float32(0.0)  # ms
timeSkip = 4

pressureHistoryDuration = np.float32(1.0 / lowestFrequency) # s
pressureHistoryLength = int(round(round(pressureHistoryDuration / timeStepSize)))
pressureHistory = np.zeros((pressureHistoryLength, numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
pressureIndex = 0
pressure_dB_cache = None

dpi = 100 
figWidth = 700 / dpi
figHeight = 300 / dpi

root = tk.Tk()
root.title('Sound Pressure Simulation')
root.geometry("1050x1280+10+10")

frameAnimation = tk.Frame(root, width=500, height=350)
frameAnimation.pack(side=tk.TOP, pady=0)

frameTimePlot = tk.Frame(root, width=500, height=350)
frameTimePlot.pack(side=tk.TOP, pady=0)

fig, ax = plt.subplots(figsize=(figWidth, figHeight), dpi=dpi)
image = ax.imshow(pressureField[30, :, :], cmap='viridis', vmin=-0.0025, vmax=0.0025, animated=True)
ax.set_title('Sound Pressure Simulation')
ax.set_xlabel('X in meter')
ax.set_ylabel('Y in meter')

xTicks = np.arange(0, numDiscretePosX, int(round(1 / posStepSize)))
xLabels = np.round(xTicks * posStepSize, 1)
ax.set_xticks(xTicks)
ax.set_xticklabels(xLabels)

yTicks = np.arange(0, numDiscretePosY, int(round(1 / posStepSize)))
yLabels = np.round(yTicks * posStepSize, 1)
ax.set_yticks(yTicks)
ax.set_yticklabels(yLabels)

lighterPurpleMicColor = (0.86, 0.44, 0.84)
defaultSpeakerColor = 'orange'
selectedSpeakerColor = 'blue'
wallColor = 'gray'

wallRects = [
    Rectangle((-0.2, -0.2), wallThickness/posStepSize-0.4, numDiscretePosY-1+0.4, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((numDiscretePosX-wallThickness/posStepSize-1+0.4, -0.2), wallThickness/posStepSize-0.2, numDiscretePosY-1+0.4, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((-0.2, -0.2), numDiscretePosX-1+0.4, wallThickness/posStepSize-0.2, color=wallColor, fill=False, linewidth=2, hatch='////'),
    Rectangle((-0.2, numDiscretePosY-wallThickness/posStepSize-1+0.4), numDiscretePosX-1+0.4, wallThickness/posStepSize-0.2, color=wallColor, fill=False, linewidth=2, hatch='////')
]

for wall in wallRects:
    ax.add_patch(wall)


max_dB_marker, = ax.plot([], [], 'x', color=(0.3, 0.5, 1), markersize=10, markeredgewidth=2, label='Max dB')
min_dB_marker, = ax.plot([], [], 'rx', markersize=10, markeredgewidth=2, label='Min dB')

cbar = fig.colorbar(image, ax=ax, orientation='vertical')
cbar.set_label('Druck (Pa)')

canvas = FigureCanvasTkAgg(fig, master=frameAnimation)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.NONE, expand=False)
canvas.get_tk_widget().pack_propagate(0)
canvas.get_tk_widget().config(width=1000, height=750)

frameMain = tk.Frame(root)
frameMain.pack(side=tk.BOTTOM, pady=10)

frameControls = tk.Frame(frameMain)
frameControls.pack(side=tk.TOP, pady=5)

frameSlider= tk.Frame(frameMain)
frameSlider.pack(side=tk.TOP, pady=5)

class RectangleShape:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self._discretePositions = None
        self._discretePositionsAndAdjacentX = None
        self._discretePositionsAndAdjacentY = None
        self._discretePositionsAndAdjacentZ = None

    def getDiscretePositions(self, axis=None):
        if axis is None and self._discretePositions is not None:
            return self._discretePositions
        if axis == Axis.X and self._discretePositionsAndAdjacentX is not None:
            return self._discretePositionsAndAdjacentX
        if axis == Axis.Y and self._discretePositionsAndAdjacentY is not None:
            return self._discretePositionsAndAdjacentY
        if axis == Axis.Z and self._discretePositionsAndAdjacentZ is not None:
            return self._discretePositionsAndAdjacentZ

        mask = np.zeros_like(pressureField, dtype=bool)
    
        startX = int(round(self.position.x / posStepSize))
        endX = int(round((self.position.x + self.size.length) / posStepSize))
        startY = int(round(self.position.y / posStepSize))
        endY = int(round((self.position.y + self.size.width) / posStepSize))
        startZ = int(round(self.position.z / posStepSize))
        endZ = int(round((self.position.z + self.size.height) / posStepSize))
    
        if axis is None:
            mask[startZ:endZ, startY:endY, startX:endX] = True
            self._discretePositions = mask
        elif axis == Axis.X:
            mask[startZ:endZ + 1, startY:endY, startX:endX] = True
            self._discretePositionsAndAdjacentX = mask
        elif axis == Axis.Y:
            mask[startZ:endZ, startY:endY + 1, startX:endX] = True
            self._discretePositionsAndAdjacentY = mask
        elif axis == Axis.Z:
            mask[startZ:endZ, startY:endY, startX:endX + 1] = True
            self._discretePositionsAndAdjacentZ = mask
    
        return mask

class EllipseShape:
    def __init__(self, position, radiusX, radiusY, radiusZ):
        self.position = position
        self.radiusX = radiusX
        self.radiusY = radiusY
        self.radiusZ = radiusZ
        self._discretePositions = None
        self._discretePositionsAndAdjacentX = None
        self._discretePositionsAndAdjacentY = None
        self._discretePositionsAndAdjacentZ = None

    def getDiscretePositions(self, axis=None):
        if axis is None and self._discretePositions is not None:
            return self._discretePositions
        if axis == Axis.X and self._discretePositionsAndAdjacentX is not None:
            return self._discretePositionsAndAdjacentX
        if axis == Axis.Y and self._discretePositionsAndAdjacentY is not None:
            return self._discretePositionsAndAdjacentY
        if axis == Axis.Z and self._discretePositionsAndAdjacentZ is not None:
            return self._discretePositionsAndAdjacentZ

        mask = np.zeros_like(pressureField, dtype=bool)
    
        startX = int(round((self.position.x - self.radiusX) / posStepSize))
        endX = int(round((self.position.x + self.radiusX) / posStepSize))
        startY = int(round((self.position.y - self.radiusY) / posStepSize))
        endY = int(round((self.position.y + self.radiusY) / posStepSize))
        startZ = int(round((self.position.z - self.radiusZ) / posStepSize))
        endZ = int(round((self.position.z + self.radiusZ) / posStepSize))
    
        centerX = self.position.x / posStepSize
        centerY = self.position.y / posStepSize
        centerZ = self.position.z / posStepSize
    
        for z in range(startZ, endZ):
            for y in range(startY, endY):
                for x in range(startX, endX):
                    ellipsoidEq = (
                        (x + 0.5 - centerX) ** 2 / (self.radiusX / posStepSize) ** 2 +
                        (y + 0.5 - centerY) ** 2 / (self.radiusY / posStepSize) ** 2 +
                        (z + 0.5 - centerZ) ** 2 / (self.radiusZ / posStepSize) ** 2)
                    if abs(ellipsoidEq - 1e-9) <= 1:
                        mask[z, y, x] = True

        if axis is not None:
            adjacentMask = np.zeros_like(mask, dtype=bool)
            if axis == Axis.X:
                adjacentMask[1:, :, :] = mask[:-1, :, :]
            elif axis == Axis.Y:
                adjacentMask[:, 1:, :] = mask[:, :-1, :]
            elif axis == Axis.Z:
                adjacentMask[:, :, 1:] = mask[:, :, :-1]
            
            mask |= adjacentMask

        if axis is None:
            self._discretePositions = mask
        elif axis == Axis.X:
            self._discretePositionsAndAdjacentX = mask
        elif axis == Axis.Y:
            self._discretePositionsAndAdjacentY = mask
        elif axis == Axis.Z:
            self._discretePositionsAndAdjacentZ = mask

        return mask


class PolygonShape:
    def __init__(self, position, vertices):
        self.position = position
        self.vertices = np.array(vertices)
        self._discretePositions = None
        self._discretePositionsAndAdjacentX = None
        self._discretePositionsAndAdjacentY = None
        self._discretePositionsAndAdjacentZ = None


    def getDiscretePositions(self, axis=None):
        if axis is None and self._discretePositions is not None:
            return self._discretePositions
        if axis == Axis.X and self._discretePositionsAndAdjacentX is not None:
            return self._discretePositionsAndAdjacentX
        if axis == Axis.Y and self._discretePositionsAndAdjacentY is not None:
            return self._discretePositionsAndAdjacentY
        if axis == Axis.Z and self._discretePositionsAndAdjacentZ is not None:
            return self._discretePositionsAndAdjacentZ

        mask = np.zeros_like(pressureField, dtype=bool)

        absoluteVertices = self.vertices + np.array([self.position.x, self.position.y, self.position.z])
        hull = Delaunay(absoluteVertices)

        minBounds = np.min(absoluteVertices, axis=0)
        maxBounds = np.max(absoluteVertices, axis=0)

        minIndex = np.floor(minBounds / posStepSize).astype(int)
        maxIndex = np.ceil(maxBounds / posStepSize).astype(int)

        minIndex = np.maximum(minIndex, [0, 0, 0])
        maxIndex = np.minimum(maxIndex, [numDiscretePosX - 1, numDiscretePosY - 1, numDiscretePosZ - 1])

        for z in range(minIndex[2], maxIndex[2] + 1):
            for y in range(minIndex[1], maxIndex[1] + 1):
                for x in range(minIndex[0], maxIndex[0] + 1):
                    point = np.array([(x + 0.5) * posStepSize, 
                                      (y + 0.5) * posStepSize, 
                                      (z + 0.5) * posStepSize])
                    if np.all(point >= minBounds) and np.all(point <= maxBounds):
                        if hull.find_simplex(point) >= 0:
                            mask[z, y, x] = True

        if axis is not None:
            adjacentMask = np.zeros_like(mask, dtype=bool)
            if axis == Axis.X:
                adjacentMask[1:, :, :] = mask[:-1, :, :]
            elif axis == Axis.Y:
                adjacentMask[:, 1:, :] = mask[:, :-1, :]
            elif axis == Axis.Z:
                adjacentMask[:, :, 1:] = mask[:, :, :-1]
            
            mask |= adjacentMask

        if axis is None:
            self._discretePositions = mask
        elif axis == Axis.X:
            self._discretePositionsAndAdjacentX = mask
        elif axis == Axis.Y:
            self._discretePositionsAndAdjacentY = mask
        elif axis == Axis.Z:
            self._discretePositionsAndAdjacentZ = mask

        return mask


class Absorber:
    def __init__(self, shape, absorptionPressure=0.0, absorptionVelocity=0.0):
        self.shape = shape
        self.absorptionPressure = absorptionPressure
        self.absorptionVelocity = absorptionVelocity

        self.shape.position = Position(
            self.shape.position.x + wallThickness,
            self.shape.position.y + wallThickness,
            self.shape.position.z + wallThickness)

    def applyAbsorption(self, pressureField, velocityFieldX, velocityFieldY, velocityFieldZ):
        pressureField[self.shape.getDiscretePositions()] *= (1 - self.absorptionPressure)
        velocityFieldX[self.shape.getDiscretePositions(Axis.X)] *= (1 - self.absorptionVelocity)
        velocityFieldY[self.shape.getDiscretePositions(Axis.Y)] *= (1 - self.absorptionVelocity)
        velocityFieldZ[self.shape.getDiscretePositions(Axis.Z)] *= (1 - self.absorptionVelocity)

    def getPatch(self):
        if isinstance(self.shape, RectangleShape):
            startX, startY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            length, width = self.shape.size.length / posStepSize, self.shape.size.width / posStepSize
            return Rectangle((startX - 0.6, startY - 0.6), length, width, color='red', fill=False)
        elif isinstance(self.shape, EllipseShape):
            centerX, centerY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            length, width = self.shape.radiusX * 2 / posStepSize, self.shape.radiusY * 2 / posStepSize
            return Ellipse((centerX - 0.6, centerY - 0.6), length, width, color='red', fill=False)
        elif isinstance(self.shape, PolygonShape):
            vertices_2d = self.shape.vertices[:, :2] + np.array([self.shape.position.x, self.shape.position.y])
            hull = ConvexHull(vertices_2d)
            hullVertices = vertices_2d[hull.vertices]
            return Polygon(hullVertices / posStepSize, color='red', fill=False)


absorbers = [
    # Corner Absorber
    Absorber(PolygonShape(Position(0.05, 0.05, 0.0), vertices=[[0, 0, 0], [0, 0.55, 0], [0.5, 0.0, 0], [0.5, 0, 2.55], [0, 0, 2.55], [0, 0.55, 2.55]]), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(PolygonShape(Position(0.05, 2.9, 0.0), vertices=[[0, 0, 0], [0, 0.55, 0], [0.5, 0.55, 0], [0.5, 0.55, 2.55], [0, 0.55, 2.55], [0, 0, 2.55]]), absorptionPressure=0.005, absorptionVelocity=0.05),
    # Side Absorber first reflection point
    Absorber(RectangleShape(Position(0.55, 0.05, 0.45), Size(0.8, 0.1, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(1.35, 0.05, 0.45), Size(0.8, 0.1, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(0.55, 3.35, 0.45), Size(0.8, 0.1, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(1.35, 3.35, 0.45), Size(0.8, 0.1, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    # Chimney
    Absorber(RectangleShape(Position(4.60, 0.0, 0.0), Size(0.40, 0.25, 2.55)), absorptionPressure=0.02, absorptionVelocity=0.94),
    # Backwall Absorber 
    Absorber(RectangleShape(Position(4.60, 0.25, 0.0), Size(0.20, 1.60, 0.8)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(4.60, 0.25, 0.0), Size(0.20, 0.8, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(4.60, 1.05, 0.0), Size(0.20, 0.8, 1.6)), absorptionPressure=0.005, absorptionVelocity=0.05),
    # Couch: l 2.00 h 0.66 w 0.13 
    Absorber(RectangleShape(Position(4.45, 0.0, 0.50), Size(0.15, 2.00, 0.60)), absorptionPressure=0.005, absorptionVelocity=0.05),
    Absorber(RectangleShape(Position(3.80, 0.0, 0.35), Size(0.80, 2.00, 0.15)), absorptionPressure=0.005, absorptionVelocity=0.05),

    #Absorber(PolygonShape(Position(2.8, 2.5, 1.2), vertices=[[0, 0, 0], [0.5, 0.2, 0], [0.3, 0.8, 0], [0, 0.6, 0], [0.2, 0.2, 0.5]]), absorptionPressure=0.01, absorptionVelocity=1)
]

absorberPatches = [ax.add_patch(absorber.getPatch()) for absorber in absorbers]

class Speaker:
    def __init__(self, name, shape, frequency, volume, minFrequency, maxFrequency, color=defaultSpeakerColor):
        self.name = name
        self.shape = shape
        self.frequency = frequency
        self.volume = volume
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.omega = 2 * np.pi * self.frequency
        self.currentPhase = 0
        self.color = color

        self.shape.position = Position(
            self.shape.position.x + wallThickness,
            self.shape.position.y + wallThickness,
            self.shape.position.z + wallThickness
        )

    def updateFrequency(self, frequency):
        self.frequency = frequency
        self.omega = 2 * np.pi * self.frequency

    def updateVolume(self, volume):
        self.volume = volume

    def updatePressure(self, pressureField, timeStepSize):
        if not (self.minFrequency <= self.frequency <= self.maxFrequency):
            return
        if self.volume == 0:
            return

        amplitude = REFERENCE_PRESSURE * (10 ** (self.volume / 20))

        previousPhase = self.currentPhase
        self.currentPhase += self.omega * timeStepSize
        self.currentPhase %= 2 * np.pi

        # Instead of giving the speakers a fixed pressure, you should leave the possibility of superimposition by adding only the difference. In this case, however, the volume is too low if you calculate it this way.
        #pressureChange = amplitude * np.sin(self.currentPhase) - amplitude * np.sin(previousPhase)
        #pressureField[self.shape.getDiscretePositions()] += pressureChange

        pressureField[self.shape.getDiscretePositions()] = amplitude * np.sin(self.currentPhase)
 

    def getPatch(self):
        if isinstance(self.shape, RectangleShape):
            startX, startY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            length, width = self.shape.size.length / posStepSize, self.shape.size.width / posStepSize
            return Rectangle((startX - 0.6, startY - 0.6), length, width, color=self.color, fill=False, linewidth=2)
        elif isinstance(self.shape, EllipseShape):
            centerX, centerY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            length, width = self.shape.radiusX * 2 / posStepSize, self.shape.radiusY * 2 / posStepSize
            return Ellipse((centerX, centerY), length, width, color=self.color, fill=False, linewidth=2)
        elif isinstance(self.shape, PolygonShape):
            vertices_2d = self.shape.vertices[:, :2] + np.array([self.shape.position.x, self.shape.position.y])
            hull = ConvexHull(vertices_2d)
            hullVertices = vertices_2d[hull.vertices]
            return Polygon(hullVertices / posStepSize, color=self.color, fill=False, linewidth=2)

    def updateColor(self, newColor):
        self.color = newColor



speakers = [
    Speaker("Main Speaker Right", EllipseShape(Position(0.25, 0.7, 1.3), 0.1, 0.1, 0.1), frequency=215, volume=85.0, minFrequency=20.0, maxFrequency=20000.0),
    Speaker("Main Speaker Left", EllipseShape(Position(0.25, 2.85, 1.3), 0.1, 0.1, 0.1), frequency=215, volume=85.0, minFrequency=20.0, maxFrequency=20000.0),
]

speakerNames = [speaker.name for speaker in speakers]

speakerPatches = [ax.add_patch(speaker.getPatch()) for speaker in speakers]


resetButton = tk.Button(frameSlider, text="Reset", command=lambda: resetSimulation(None))
resetButton.pack(side=tk.LEFT, padx=15)

stopButton = tk.Button(frameSlider, text="Stop", command=lambda: toggleSimulation(None))
stopButton.pack(side=tk.LEFT, padx=15)

fieldTarget = tk.StringVar(root)
fieldTarget.set("Pressure")

fieldOptions = ["Pressure", "Velocity X", "Velocity Y", "dB Level"]
fieldMenu = tk.OptionMenu(frameControls, fieldTarget, *fieldOptions)
fieldMenu.pack(side=tk.LEFT, padx=15)

def updatePressureHistory(pressureField):
    global pressureHistory, pressureIndex
    pressureHistory[pressureIndex] = pressureField.copy()
    pressureIndex = (pressureIndex + 1) % len(pressureHistory)

def calcPressure_dB():
    #  The textbook method is to calculate the rms pressure to calculate the dB over two periods of the lowest expected frequency. With rms the volume is underestimated by 10%, therefore the substitute calculation with min and max value is used which works in this controlled environment.
    #rmsPressure = np.sqrt(np.mean(pressureHistory**2, axis=0))
    maxPressure = np.amax(pressureHistory, axis=0)
    minPressure = np.amin(pressureHistory, axis=0)
    difPressure = abs(maxPressure - minPressure)/2
    pressure_dB = 20 * np.log10(difPressure / REFERENCE_PRESSURE + 1e-12) # + 1e-12 to avoid log(0)
    return pressure_dB


def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ):
    factorP = np.float32(timeStepSize * SPEED_OF_SOUND**2 / posStepSize)
    factorV = np.float32(timeStepSize / posStepSize)
    
    pressureField[:-1, :, :] -= factorP * (velocityFieldX[1:, :, :] - velocityFieldX[:-1, :, :])
    pressureField[:, :-1, :] -= factorP * (velocityFieldY[:, 1:, :] - velocityFieldY[:, :-1, :])
    pressureField[:, :, :-1] -= factorP * (velocityFieldZ[:, :, 1:] - velocityFieldZ[:, :, :-1])
    
    velocityFieldX[1:, :, :] -= factorV * (pressureField[1:, :, :] - pressureField[:-1, :, :])
    velocityFieldY[:, 1:, :] -= factorV * (pressureField[:, 1:, :] - pressureField[:, :-1, :])
    velocityFieldZ[:, :, 1:] -= factorV * (pressureField[:, :, 1:] - pressureField[:, :, :-1])


def applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ):
    for i in range(int(round(wallThickness / posStepSize))):
        pressureField[i, :, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[-i-2, :, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, i, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, -i-2, :] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, :, i] *= 1 - wallPressureAbsorptionCoefficient
        pressureField[:, :, -i-2] *= 1 - wallPressureAbsorptionCoefficient

        velocityFieldX[i, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[-i-1, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, i, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, -i-1, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, :, i] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldX[:, :, -i-1] *= 1 - wallVelocityAbsorptionCoefficient

        velocityFieldY[i, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[-i-1, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, i, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, -i-1, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, :, i] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldY[:, :, -i-1] *= 1 - wallVelocityAbsorptionCoefficient

        velocityFieldZ[i, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldZ[-i-1, :, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldZ[:, i, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldZ[:, -i-1, :] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldZ[:, :, i] *= 1 - wallVelocityAbsorptionCoefficient
        velocityFieldZ[:, :, -i-1] *= 1 - wallVelocityAbsorptionCoefficient

    wallReflexionLayer = int(round(wallThickness / posStepSize)) - 1  
    
    velocityFieldX[wallReflexionLayer+1, wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldX[-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, -wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldZ[wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldZ[wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2, -wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient

    velocityFieldX[0, :, :] = 0
    velocityFieldX[-1, :, :] = 0
    velocityFieldY[:, 0, :] = 0
    velocityFieldY[:, -1, :] = 0
    velocityFieldZ[:, :, 0] = 0
    velocityFieldZ[:, :, -1] = 0


def updateSimulation(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ):
    global simulatedTime

    updatePressureHistory(pressureField)
    calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)
    applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)

    for absorber in absorbers:
        absorber.applyAbsorption(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)

    for speaker in speakers:
        speaker.updatePressure(pressureField, timeStepSize)
    simulatedTime += timeStepSize * 1000

def updateDisplayedField():
    global pressure_dB_cache
    selectedField = fieldTarget.get()
    if selectedField == "Pressure":
        image.set_array(pressureField[30, :-1, :-1]) # 5 - 53
        image.set_clim(-0.25, 0.25)
        cbar.set_label('Pressure (Pa)')
    elif selectedField == "Velocity X":
        image.set_array(velocityFieldX[30, :-1, :-1])
        image.set_clim(-0.001, 0.001)
        cbar.set_label('Particle velocity in X (m/s)')
    elif selectedField == "Velocity Y":
        image.set_array(velocityFieldY[30, :-1, :-1])
        image.set_clim(-0.001, 0.001)
        cbar.set_label('Particle velocity in Y (m/s)')
    elif selectedField == "dB Level":
        if pressureIndex % 100 < 4:
            pressure_dB_cache = calcPressure_dB()
        if pressure_dB_cache is not None and pressure_dB_cache.ndim == 3:
            image.set_array(pressure_dB_cache[30, :-1, :-1])
        norm = Normalize(vmin=50, vmax=90)
        image.set_norm(norm)
        image.set_clim(50, 90)
        cbar.set_label('Sound Pressure Level (dB)')

def updateLegends(*args):
    canvas.draw_idle()

fieldTarget.trace_add("write", updateLegends)

def createExclusionMask():
    mask = np.ones(pressureField.shape, dtype=bool)
    return np.ones(pressureField.shape, dtype=bool)
    
    thickness = int(round(wallThickness / posStepSize))
    mask[:thickness, :] = False
    mask[-thickness-1:, :] = False
    mask[:, :thickness] = False
    mask[:, -thickness-1:] = False

    for absorber in absorbers:
        discretePositions = absorber.shape.getDiscretePositions()
        for x, y in discretePositions:
            mask[y, x] = False

    for speaker in speakers:
        discretePositions = speaker.shape.getDiscretePositions()
        for x, y in discretePositions:
            mask[y, x] = False

    return mask

exclusionMask = createExclusionMask()


textElements = [
    ax.text(0.01, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # time
    ax.text(0.29, 0.99, '', transform=ax.transAxes, color=(0.3, 0.5, 1), fontsize=12, weight='bold', va='top'),  # max
    ax.text(0.37, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # maxValue
    ax.text(0.54, 0.99, '', transform=ax.transAxes, color='red', fontsize=12, weight='bold', va='top'),  # min
    ax.text(0.61, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # minValue
    ax.text(0.77, 0.99, '', transform=ax.transAxes, color=lighterPurpleMicColor, fontsize=12, weight='bold', va='top'),  # mic dB
    ax.text(0.84, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # mic dB Value
]


def updateText(simulatedTime, pressure_dB_cache):
    selectedField = fieldTarget.get()
    if selectedField == "dB Level" and pressure_dB_cache is not None:
        max_dB = np.max(np.where(exclusionMask, pressure_dB_cache, -np.inf))
        min_dB = np.min(np.where(exclusionMask, pressure_dB_cache, np.inf))

        centerMicX = int(round(micX.get() / posStepSize))
        centerMicY = int(round(micY.get() / posStepSize))
        centerMicZ = int(round(micZ.get() / posStepSize))
        centerMicX = np.clip(centerMicX, 0, numDiscretePosX - 1)
        centerMicY = np.clip(centerMicY, 0, numDiscretePosY - 1)
        centerMicZ = np.clip(centerMicZ, 0, numDiscretePosZ - 1)

        mic_dB = pressure_dB_cache[centerMicZ, centerMicY, centerMicX]

        max_dB = max(max_dB, 0)
        min_dB = max(min_dB, 0)
        mic_dB = max(mic_dB, 0)

        textElements[0].set_text(f'Time: {simulatedTime:2.2f} ms, ')
        textElements[1].set_text('Max: ')
        textElements[2].set_text(f'{max_dB:2.2f} dB, ')
        textElements[3].set_text('Min: ')
        textElements[4].set_text(f'{min_dB:2.2f} dB, ')
        textElements[5].set_text('Mic: ')
        textElements[6].set_text(f'{mic_dB:2.2f} dB')
    else:
        textElements[0].set_text(f'Time: {simulatedTime:2.2f} ms')
        textElements[1].set_text('')
        textElements[2].set_text('')
        textElements[3].set_text('')
        textElements[4].set_text('')
        textElements[5].set_text('')
        textElements[6].set_text('')

def updateMarkers(pressure_dB_cache):
    selectedField = fieldTarget.get()
    if selectedField == "dB Level" and pressure_dB_cache is not None:
        validPressure_dB = np.where(exclusionMask, pressure_dB_cache, -np.inf)
        max_dB_position = np.unravel_index(np.argmax(validPressure_dB, axis=None), validPressure_dB.shape)
        validPressure_dB = np.where(exclusionMask, pressure_dB_cache, np.inf)
        min_dB_position = np.unravel_index(np.argmin(validPressure_dB, axis=None), validPressure_dB.shape)

        max_dB_marker.set_data([max_dB_position[1]], [max_dB_position[0]])
        min_dB_marker.set_data([min_dB_position[1]], [min_dB_position[0]])

        max_dB_marker.set_visible(True)
        min_dB_marker.set_visible(True)
    else:
        max_dB_marker.set_visible(False)
        min_dB_marker.set_visible(False)

ax2_xMin = -1000 * timeStepSize * 1000 * timeSkip
ax2_xMax = 0
fig2, ax2 = plt.subplots(figsize=(figWidth, figHeight), dpi=dpi)
ax2.set_title('Pressure change over time')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Pressure (Pa)')
ax2.set_ylim(-0.4, 0.4)
ax2.set_xlim(ax2_xMin, ax2_xMax)
ax2.grid(True)
ax2.axhline(0, color='black', linewidth=2)

timeLabels = np.arange(ax2_xMin, ax2_xMax + 1, 400 * timeStepSize * 1000)
ax2.set_xticks(timeLabels)
ax2.set_xticklabels(np.round(timeLabels, 1))

timeData = np.linspace(ax2_xMin, ax2_xMax, 1000)
pressureDataSpeaker = [0] * 1000
pressureDataMic = [0] * 1000
micOffsetInSteps = 0
lineSpeaker, = ax2.plot(timeData, pressureDataSpeaker, lw=2, color=selectedSpeakerColor, label="Speaker")
lineMic, = ax2.plot(timeData, pressureDataMic, lw=2, color=lighterPurpleMicColor, label="Microphone")


def onMicPositionChange(*args):
    updateMicMarker()

def validatePositiveOffset(valueIfAllowed):
    if valueIfAllowed.isdigit() or (valueIfAllowed.replace('.', '', 1).isdigit() and valueIfAllowed.count('.') < 2):
        if 50.0 >= float(valueIfAllowed) >= 0:
            return True
    return False


dbAnalyseButton = tk.Button(frameControls, text="Freq Analyse", command=lambda: performDBAnalyse())
dbAnalyseButton.pack(side=tk.LEFT, padx=15)
responseAnalyseButton = tk.Button(frameControls, text="Response Analyse", command=lambda: performFrequencyResponseAnalysisSweep())
responseAnalyseButton.pack(side=tk.LEFT, padx=15)
micX = tk.DoubleVar(value=1.7 + wallThickness)
micY = tk.DoubleVar(value=1.8 + wallThickness)
micZ = tk.DoubleVar(value=1.3 + wallThickness)
micOffset = tk.DoubleVar(value=0.0)
micPositionLabel = tk.Label(frameControls, text="Mikrofon Position (X, Y):")
micPositionLabel.pack(side=tk.LEFT, padx=5)
micEntryX = tk.Entry(frameControls, textvariable=micX, width=5)
micEntryX.pack(side=tk.LEFT, padx=5)
micEntryY = tk.Entry(frameControls, textvariable=micY, width=5)
micEntryY.pack(side=tk.LEFT, padx=5)
micEntryZ = tk.Entry(frameControls, textvariable=micZ, width=5)
micEntryZ.pack(side=tk.LEFT, padx=5)

micOffsetLabel = tk.Label(frameControls, text="Offset (ms):")
micOffsetLabel.pack(side=tk.LEFT, padx=5)
vcmd = (root.register(validatePositiveOffset), '%P')
micOffsetEntry = tk.Entry(frameControls, textvariable=micOffset, width=5, validate='key', validatecommand=vcmd)
micOffsetEntry.pack(side=tk.LEFT, padx=5)

micX.trace_add("write", onMicPositionChange)
micY.trace_add("write", onMicPositionChange)
micZ.trace_add("write", onMicPositionChange)
micOffset.trace_add("write", lambda *args: applyMicOffset())

micMarker = Circle((micX.get() / posStepSize, micY.get() / posStepSize), radius=1, color=lighterPurpleMicColor, fill=False, linewidth=2)
ax.add_patch(micMarker)

def performDBAnalyse():
    if animRunning:
        toggleSimulation(None)

    centerMicX = int(round(micX.get() / posStepSize))
    centerMicY = int(round(micY.get() / posStepSize))
    centerMicZ = int(round(micZ.get() / posStepSize))
    centerMicX = np.clip(centerMicX, 0, numDiscretePosX - 1)
    centerMicY = np.clip(centerMicY, 0, numDiscretePosY - 1)
    centerMicZ = np.clip(centerMicZ, 0, numDiscretePosZ - 1)

    pressureAtMic = np.array([history[centerMicY, centerMicX, centerMicZ] for history in pressureHistory])

    N = len(pressureAtMic)
    T = timeStepSize
    yf = fft(pressureAtMic)
    xf = fftfreq(N, T)[:N//2]
    amplitudes = 2.0 / N * np.abs(yf[:N//2])

    logFreqs = np.logspace(np.log10(lowestFrequency), np.log10(20000), num=200)
    interpFunc = interp1d(xf, amplitudes, kind='linear', bounds_error=False, fill_value=0)
    logAmplitudes = interpFunc(logFreqs)

    pressure_dB = 20 * np.log10(logAmplitudes / REFERENCE_PRESSURE + 1e-12)
    
    analysisWindow = tk.Toplevel()
    analysisWindow.title("Frequency analysis")
    analysisWindow.geometry("1000x800")

    fig = plt.Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(logFreqs, pressure_dB, width=np.diff(logFreqs, append=logFreqs[-1]), align='center', log=True)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Volume (dB)')
    ax.set_title('Frequency analysis')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yscale('linear')
    ax.set_ylim(10, 90)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=analysisWindow)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    analysisWindow.update_idletasks()

def performFrequencyResponseAnalysisSweep():
    startFrequency = 20.0
    endFrequency = 2000.0
    sweepDuration = 2.0
    samplingRate = 1.0 / timeStepSize

    def generateSweepTone(startFreq, endFreq, duration, samplingRate):
        t = np.linspace(0, duration, int(samplingRate * duration))
        sweepTone = np.sin(2 * np.pi * startFreq * t * (endFreq / startFreq) ** (t / duration))
        return sweepTone

    def smooth(y, boxPts):
        box = np.ones(boxPts) / boxPts
        smoothY = np.convolve(y, box, mode='same')
        return smoothY

    def butterLowpassFilter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normalCutoff = cutoff / nyquist
        b, a = butter(order, normalCutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def smoothLogScale(x, y, xMin=10, xMax=2000, numPoints=4096, windowSize=31, extensionFactor=0.1):
        mask = (x >= xMin) & (x <= xMax)
        xFocus = x[mask]
        yFocus = y[mask]

        xExtensionLeft = xMin * (1 - extensionFactor)
        xExtensionRight = xMax * (1 + extensionFactor)
        xExtended = np.concatenate(([xExtensionLeft], xFocus, [xExtensionRight]))
        yExtended = np.concatenate(([yFocus[0]], yFocus, [yFocus[-1]]))

        logX_extended = np.log10(xExtended)
        logX_new = np.linspace(logX_extended.min(), logX_extended.max(), num=numPoints)
        yInterp = interp.interp1d(logX_extended, yExtended, kind='linear')(logX_new)
        ySmooth = np.convolve(yInterp, np.ones(windowSize)/windowSize, mode='same')

        validMask = (logX_new >= np.log10(xMin)) & (logX_new <= np.log10(xMax))
        xSmooth = 10**logX_new[validMask]
        ySmooth = ySmooth[validMask]

        return xSmooth, ySmooth

    def runAnalysis():
        sweepTone = generateSweepTone(startFrequency, endFrequency, sweepDuration, samplingRate)

        originalVolume = speakers[0].volume
        originalFrequency = speakers[0].frequency
        for speaker in speakers:
            speaker.updateFrequency(startFrequency)
            speaker.updateVolume(85.0)  

        pressureAtMic = []
        centerMicX = int(round(micX.get() / posStepSize))
        centerMicY = int(round(micY.get() / posStepSize))
        centerMicZ = int(round(micZ.get() / posStepSize))       
        centerMicX = np.clip(centerMicX, 0, numDiscretePosX - 1)
        centerMicY = np.clip(centerMicY, 0, numDiscretePosY - 1)
        centerMicZ = np.clip(centerMicZ, 0, numDiscretePosZ - 1)

        for i, tone in enumerate(sweepTone):
            for speaker in speakers:
                speaker.updateFrequency(startFrequency + (endFrequency - startFrequency) * (i / len(sweepTone)))
                speaker.currentPhase = 2 * np.pi * speaker.frequency * (i / samplingRate)
                speaker.updatePressure(pressureField, timeStepSize)

            updateSimulation(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)
            pressureAtMic.append(pressureField[centerMicZ, centerMicY, centerMicX])

            progress['value'] = i + 1
            progressWindow.update_idletasks()

        resetSimulation(None)
        progressWindow.destroy()

        N = len(pressureAtMic)
        yf = fft(pressureAtMic)
        xf = fftfreq(N, 1 / samplingRate)[:N // 2]

        amplitudes = 2.0 / N * np.abs(yf[:N // 2]) 
        amplitudes_dB = 20 * np.log10(amplitudes / REFERENCE_PRESSURE + 1e-12)
        xf_smooth, smoothedAmplitudes_dB = smoothLogScale(xf, amplitudes_dB)
        smoothedAmplitudes_dB = butterLowpassFilter(smoothedAmplitudes_dB, 0.2, 1.0, 3)

        analysisWindow = tk.Toplevel()
        analysisWindow.title("Frequency Response Analysis - Sweep")
        analysisWindow.geometry("1000x800")

        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(xf_smooth, smoothedAmplitudes_dB, color='blue')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title('Frequency Response Analysis - Sweep')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax.set_ylim(0, 60)
        ax.set_yscale('linear')
        ax.set_xlim([startFrequency, endFrequency])

        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        canvas = FigureCanvasTkAgg(fig, master=analysisWindow)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        analysisWindow.update_idletasks()

    if animRunning:
        toggleSimulation(None)
    resetSimulation(None)

    progressWindow = tk.Toplevel()
    progressWindow.title("Progress")
    progressWindow.geometry("300x100")
    progressLabel = tk.Label(progressWindow, text="Performing Frequency Response Analysis - Sweep...")
    progressLabel.pack(pady=10)

    progress = ttk.Progressbar(progressWindow, orient=tk.HORIZONTAL, length=250, mode='determinate', maximum=samplingRate * sweepDuration)
    progress.pack(pady=10)

    analysisThread = threading.Thread(target=runAnalysis)
    analysisThread.start()

def performFrequencyResponseAnalysis():
    startFrequency = 20.0  
    endFrequency = 2000.0  
    numSteps = 200
    duration = 0.05

    def runAnalysis():
        frequencies = np.logspace(np.log10(startFrequency), np.log10(endFrequency), numSteps)
        responses = []

        for speaker in speakers: 
            originalVolume = speaker.volume
            originalFrequency = speaker.frequency

        centerMicX = int(round(micX.get() / posStepSize))
        centerMicY = int(round(micY.get() / posStepSize))
        centerMicX = np.clip(centerMicX, 0, numDiscretePosX - 1)
        centerMicY = np.clip(centerMicY, 0, numDiscretePosY - 1)

        for i, freq in enumerate(frequencies):
            resetSimulation(None)
            for speaker in speakers: 
                speaker.updateFrequency(freq)
            for _ in range(int(duration / timeStepSize)):
                updateSimulation(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)

            pressure_dB_cache = calcPressure_dB()
            mic_dB = pressure_dB_cache[centerMicY, centerMicX]
            responses.append(mic_dB)

            progress['value'] = i + 1
            progressWindow.update_idletasks()

        resetSimulation(None)
        progressWindow.destroy()
        
        for speaker in speakers: 
            speaker.updateVolume(originalVolume)
            speaker.updateFrequency(originalFrequency)
    
        analysisWindow = tk.Toplevel()
        analysisWindow.title("Frequency Response Analysis")
        analysisWindow.geometry("1000x800")
    
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(frequencies, responses, marker='o', linestyle='-', color='blue')
        ax.set_xscale('log')
        ax.set_ylim(10, 90)
        ax.set_yscale('linear')
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim([startFrequency, endFrequency])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Volume (dB)')
        ax.set_title('Frequency Response Analysis')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
        fig.tight_layout()
    
        canvas = FigureCanvasTkAgg(fig, master=analysisWindow)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        analysisWindow.update_idletasks()

    if animRunning:
        toggleSimulation(None)
    resetSimulation(None)

    progressWindow = tk.Toplevel()
    progressWindow.title("Progress")
    progressWindow.geometry("300x100")
    progressLabel = tk.Label(progressWindow, text="Performing Frequency Response Analysis...")
    progressLabel.pack(pady=10)

    progress = ttk.Progressbar(progressWindow, orient=tk.HORIZONTAL, length=250, mode='determinate', maximum=numSteps)
    progress.pack(pady=10)

    analysisThread = threading.Thread(target=runAnalysis)
    analysisThread.start()



def updateMicMarker():
    try:
        micValueX = micX.get()
        micValueY = micY.get()
        if micValueX == "" or micValueY == "":
            return 
        micMarker.set_center((float(micValueX) / posStepSize, float(micValueY) / posStepSize))
    except tk.TclError:
        return


def applyMicOffset():
    global pressureDataMic, micOffsetInSteps
    scaledOffset = micOffset.get() / (timeSkip * timeStepSize * 1000)
    newOffsetInSteps = int(round(scaledOffset))
    if newOffsetInSteps > micOffsetInSteps:
        shift = newOffsetInSteps - micOffsetInSteps
        pressureDataMic = pressureDataMic[shift:] + [0] * shift
    elif newOffsetInSteps < micOffsetInSteps:
        shift = micOffsetInSteps - newOffsetInSteps
        pressureDataMic = [0] * shift + pressureDataMic[:-shift]

    micOffsetInSteps = newOffsetInSteps


def updateTimePlot():
    global pressureDataMic, pressureDataSpeaker
    index = speakerNames.index(selectedSpeaker.get())
    speaker = speakers[index]

    centerSpeakerX = int(round(speaker.shape.position.x / posStepSize))
    centerSpeakerY = int(round(speaker.shape.position.y / posStepSize))
    centerSpeakerZ = int(round(speaker.shape.position.z / posStepSize))
    
    currentPressureSpeaker = pressureField[centerSpeakerZ, centerSpeakerY, centerSpeakerX]

    centerMicX = int(round(micX.get() / posStepSize))
    centerMicY = int(round(micY.get() / posStepSize))
    centerMicZ = int(round(micZ.get() / posStepSize))

    centerMicX = np.clip(centerMicX, 0, numDiscretePosX - 1)
    centerMicY = np.clip(centerMicY, 0, numDiscretePosY - 1)
    centerMicZ = np.clip(centerMicZ, 0, numDiscretePosZ - 1)

    currentPressureMic = pressureField[centerMicZ, centerMicY, centerMicX]   
      
    pressureDataMic.pop(0)
    if micOffsetInSteps == 0:
        pressureDataMic.append(currentPressureMic)
    else:
        pressureDataMic.insert(-micOffsetInSteps, currentPressureMic)
    pressureDataSpeaker.append(currentPressureSpeaker)
    pressureDataSpeaker.pop(0)

    lineSpeaker.set_ydata(pressureDataSpeaker)
    lineMic.set_ydata(pressureDataMic)

    handles, labels = ax2.get_legend_handles_labels()
    if labels:
        ax2.legend(loc="upper right")


def update(frame):
    global pressureField, velocityFieldX, velocityFieldY, velocityFieldZ, animRunning
    if animRunning:
        #startzeit = time.time()
        for _ in range(4):
            updateSimulation(pressureField, velocityFieldX, velocityFieldY, velocityFieldZ)
        #endzeit = time.time()
        #dauer = endzeit - startzeit
        #print(f"Die Funktion benötigte {dauer:.6f} Sekunden.")

        updateTimePlot()
        updateText(simulatedTime, pressure_dB_cache)
        updateMarkers(pressure_dB_cache)
        updateMicMarker()

    updateDisplayedField()
    return [image] + speakerPatches + textElements + [max_dB_marker, min_dB_marker] + wallRects + absorberPatches + [lineSpeaker, lineMic, micMarker]

controlIndividualSpeakersFlag = tk.BooleanVar(value=False)

controlAllSpeakersToggle = tk.Checkbutton(frameControls, text="Individual Speaker Control", variable=controlIndividualSpeakersFlag)
controlAllSpeakersToggle.pack(side=tk.LEFT, padx=5)

selectedSpeaker = tk.StringVar()
selectedSpeaker.set(speakerNames[0])

speakerMenu = tk.OptionMenu(frameControls, selectedSpeaker, *speakerNames)
def toggleSpeakerMenu():
    if controlIndividualSpeakersFlag.get():
        speakerMenu.pack(side=tk.LEFT, padx=5)
    else:
        speakerMenu.pack_forget()

speakerFrequencyLabel = tk.Label(frameSlider, text="Frequency (Hz)")
speakerFrequencyLabel.pack(side=tk.LEFT, padx=5)

speakerFrequencyEntry = tk.Entry(frameSlider, width=8)
speakerFrequencyEntry.insert(0, str(speakers[0].frequency))
speakerFrequencyEntry.pack(side=tk.LEFT, padx=5)

speakerFrequencySlider = tk.Scale(frameSlider, from_=lowestFrequency, to=highestFrequency, orient=tk.HORIZONTAL, length=200, resolution=0.01)
speakerFrequencySlider.set(speakers[0].frequency)
speakerFrequencySlider.pack(side=tk.LEFT, padx=5)

speakerVolumeLabel = tk.Label(frameSlider, text="Volume (dB)")
speakerVolumeLabel.pack(side=tk.LEFT, padx=5)

speakerVolumeEntry = tk.Entry(frameSlider, width=8)
speakerVolumeEntry.insert(0, str(speakers[0].volume))
speakerVolumeEntry.pack(side=tk.LEFT, padx=5)

speakerVolumeSlider = tk.Scale(frameSlider, from_=0, to=120, orient=tk.HORIZONTAL, length=200, resolution=0.01)
speakerVolumeSlider.set(speakers[0].volume)
speakerVolumeSlider.pack(side=tk.LEFT, padx=5)

controlIndividualSpeakersFlag.trace_add("write", lambda *args: toggleSpeakerMenu())
toggleSpeakerMenu()


def updateSelectedSpeaker(*args):
    global speakerPatches

    for speaker in speakers:
        speaker.updateColor(defaultSpeakerColor)

    index = speakerNames.index(selectedSpeaker.get())
    speakers[index].updateColor(selectedSpeakerColor)

    for patch in speakerPatches:
        patch.remove()
    speakerPatches = [ax.add_patch(speaker.getPatch()) for speaker in speakers]

    if controlIndividualSpeakersFlag.get():
        index = speakerNames.index(selectedSpeaker.get())
        speaker = speakers[index]
        speakerFrequencySlider.set(speaker.frequency)
        speakerVolumeSlider.set(speaker.volume)
        speakerFrequencyEntry.delete(0, tk.END)
        speakerFrequencyEntry.insert(0, str(speaker.frequency))
        speakerVolumeEntry.delete(0, tk.END)
        speakerVolumeEntry.insert(0, str(speaker.volume))

selectedSpeaker.trace_add("write", updateSelectedSpeaker)


def updateFrequency(event):
    newFrequency = float(speakerFrequencySlider.get())
    if controlIndividualSpeakersFlag.get():
        index = speakerNames.index(selectedSpeaker.get())
        speakers[index].updateFrequency(newFrequency)
    else:
        for speaker in speakers:
            speaker.updateFrequency(newFrequency)

    speakerFrequencyEntry.delete(0, tk.END)
    speakerFrequencyEntry.insert(0, str(newFrequency))

def updateFrequencyFromEntry(event):
    try:
        val = float(speakerFrequencyEntry.get())
        if lowestFrequency <= val <= highestFrequency:
            speakerFrequencySlider.set(val)
            updateFrequency(None)
    except ValueError:
        pass

def updateVolume(event):
    newVolume = float(speakerVolumeSlider.get())
    if controlIndividualSpeakersFlag.get():
        index = speakerNames.index(selectedSpeaker.get())
        speakers[index].updateVolume(newVolume)
    else:
        for speaker in speakers:
            speaker.updateVolume(newVolume)

    speakerVolumeEntry.delete(0, tk.END)
    speakerVolumeEntry.insert(0, str(newVolume))


def updateVolumeFromEntry(event):
    try:
        val = float(speakerVolumeEntry.get())
        if 0.0 <= val <= 120.0:
            speakerVolumeSlider.set(val)
            updateVolume(None)
    except ValueError:
        pass

def resetSimulation(event):
    global pressureField, velocityFieldX, velocityFieldY, velocityFieldZ, simulatedTime, pressureDataMic, pressureDataSpeaker, pressureHistory, pressureIndex

   
    pressureField = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
    velocityFieldX = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
    velocityFieldY = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
    velocityFieldZ = np.zeros((numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
    simulatedTime = 0.0

    pressureDataMic = [0] * 1000
    pressureDataSpeaker = [0] * 1000
    lineSpeaker.set_ydata(pressureDataSpeaker)
    lineMic.set_ydata(pressureDataMic)
    canvas2.draw_idle()

    for speaker in speakers:
        speaker.currentPhase = 0

    pressureHistory = np.zeros((pressureHistoryLength, numDiscretePosZ, numDiscretePosY, numDiscretePosX), dtype=np.float32)
    pressureIndex = 0

def toggleSimulation(event):
    global animRunning
    animRunning = not animRunning
    if animRunning:
        stopButton.config(text='Stop')
    else:
        stopButton.config(text='Start')

speakerFrequencySlider.config(command=updateFrequency)
speakerVolumeSlider.config(command=updateVolume)
speakerFrequencyEntry.bind("<Return>", updateFrequencyFromEntry)
speakerVolumeEntry.bind("<Return>", updateVolumeFromEntry)

updateSelectedSpeaker()

animation = FuncAnimation(fig, update, blit=True, interval=1, cache_frame_data=False)

canvas2 = FigureCanvasTkAgg(fig2, master=frameTimePlot)
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.NONE, expand=False)
canvas2.get_tk_widget().pack_propagate(0)
canvas2.get_tk_widget().config(width=1000, height=400)

root.mainloop()
