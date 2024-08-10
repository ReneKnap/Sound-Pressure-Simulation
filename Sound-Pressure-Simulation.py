import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
from matplotlib.path import Path as mplPath
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import NamedTuple
from matplotlib.colors import Normalize

class Position(NamedTuple):
    x: float
    y: float

class Size(NamedTuple):
    width: float
    height: float

posStepSize = 0.05  # m
SPEED_OF_SOUND = 346.3  # Speed of sound in air in m/s at 25°C
timeStepSize = 0.1 * posStepSize / SPEED_OF_SOUND  # s
REFERENCE_PRESSURE = 20e-6  # Reference pressure in Pascals for 0 dB
lowestFrequency = 20.0  # Hz
highestFrequency = 2000.0  # Hz

roomWidth = 5.1  # m
roomHeight = 3.6   # m
wallThickness = 0.2  # m

# Room size plus 2 times wall thickness
numDiscretePosX = int(round(roomWidth / posStepSize)) + int(round(wallThickness / posStepSize)) * 2
numDiscretePosY = int(round(roomHeight / posStepSize)) + int(round(wallThickness / posStepSize)) * 2

pressureField = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldX = np.zeros((numDiscretePosY, numDiscretePosX))
velocityFieldY = np.zeros((numDiscretePosY, numDiscretePosX))

speakerRadius = 0.3  # m
speakerPos = Position(0.5, 2)  # m

wallReflectionCoefficient = 0.8 # Proportion of reflection (0.0 to 1.0)
wallPressureAbsorptionCoefficient = 0.2 # Proportion of pressure absorption (0.0 to 1.0)
wallVelocityAbsorptionCoefficient = 0.2 # Proportion of velocity absorption (0.0 to 1.0)

animRunning = True
simulatedTime = 0.0  # ms

pressureHistoryDuration = 1.0 / lowestFrequency  # s
pressureHistoryLength = int(round(round(pressureHistoryDuration / timeStepSize)))
pressureHistory = [np.zeros((numDiscretePosY, numDiscretePosX)) for _ in range(pressureHistoryLength)]
pressureIndex = 0
pressure_dB_cache = None

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

xTicks = np.arange(0, numDiscretePosX, int(round(1 / posStepSize)))
xLabels = np.round(xTicks * posStepSize, 1)
ax.set_xticks(xTicks)
ax.set_xticklabels(xLabels)

yTicks = np.arange(0, numDiscretePosY, int(round(1 / posStepSize)))
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
frameMain.pack(side=tk.TOP, pady=10)

frameControls = tk.Frame(frameMain)
frameControls.pack(side=tk.TOP, pady=5)

frameSlider= tk.Frame(frameMain)
frameSlider.pack(side=tk.TOP, pady=5)

class RectangleShape:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self._discretePositions = None

    def getDiscretePositions(self):
        if self._discretePositions is None:
            startX = int(round(self.position.x / posStepSize))
            endX = int(round((self.position.x + self.size.width) / posStepSize))
            startY = int(round(self.position.y / posStepSize))
            endY = int(round((self.position.y + self.size.height) / posStepSize))
            self._discretePositions = [(x, y) for y in range(startY, endY) for x in range(startX, endX)]
        return self._discretePositions

class EllipseShape:
    def __init__(self, position, radiusX, radiusY):
        self.position = position
        self.radiusX = radiusX
        self.radiusY = radiusY
        self._discretePositions = None

    def getDiscretePositions(self):
        if self._discretePositions is None:
            startX = int(round((self.position.x - self.radiusX) / posStepSize))
            endX = int(round((self.position.x + self.radiusX) / posStepSize))
            startY = int(round((self.position.y - self.radiusY) / posStepSize))
            endY = int(round((self.position.y + self.radiusY) / posStepSize))

            self._discretePositions = []
            centerX = self.position.x / posStepSize
            centerY = self.position.y / posStepSize
            for y in range(startY, endY):
                for x in range(startX, endX):
                    ellipse_eq = ((x + 0.5 - centerX) ** 2 / (self.radiusX / posStepSize) ** 2 +
                                  (y + 0.5 - centerY) ** 2 / (self.radiusY / posStepSize) ** 2)
                    if ellipse_eq <= 1:
                        self._discretePositions.append((x, y))
        return self._discretePositions

class PolygonShape:
    def __init__(self, position, vertices):
        self.position = position
        self.vertices = np.array(vertices)
        self._discretePositions = None

    def getDiscretePositions(self):
        if self._discretePositions is None:
            absoluteVertices = self.vertices + np.array([self.position.x, self.position.y])
            polygon_path = mplPath(absoluteVertices / posStepSize)
            self._discretePositions = []
            for y in range(numDiscretePosY):
                for x in range(numDiscretePosX):
                    if polygon_path.contains_point((x + 0.5, y + 0.5)):
                        self._discretePositions.append((x, y))
        return self._discretePositions


class Absorber:
    def __init__(self, shape, absorptionPressure=0.0, absorptionVelocity=0.0):
        self.shape = shape
        self.absorptionPressure = absorptionPressure
        self.absorptionVelocity = absorptionVelocity

        self.shape.position = Position(
            self.shape.position.x + wallThickness,
            self.shape.position.y + wallThickness)

    def applyAbsorption(self, pressureField, velocityFieldX, velocityFieldY):
        discrete_positions = self.shape.getDiscretePositions()
        for x, y in discrete_positions:
            pressureField[y, x] *= (1 - self.absorptionPressure)
            velocityFieldX[y, x] *= (1 - self.absorptionVelocity)
            velocityFieldY[y, x] *= (1 - self.absorptionVelocity)

    def getPatch(self):
        if isinstance(self.shape, RectangleShape):
            startX, startY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            width, height = self.shape.size.width / posStepSize, self.shape.size.height / posStepSize
            return Rectangle((startX - 0.6, startY - 0.6), width, height, color='red', fill=False)
        elif isinstance(self.shape, EllipseShape):
            centerX, centerY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            width, height = self.shape.radiusX * 2 / posStepSize, self.shape.radiusY * 2 / posStepSize
            return Ellipse((centerX - 0.6, centerY - 0.6), width, height, color='red', fill=False)
        elif isinstance(self.shape, PolygonShape):
            absoluteVertices = self.shape.vertices + np.array([self.shape.position.x, self.shape.position.y])
            return Polygon(absoluteVertices / posStepSize, color='red', fill=False)

absorbers = [
    Absorber(RectangleShape(Position(0.0, 0.0), Size(0.5, 0.5)), absorptionPressure=0.0, absorptionVelocity=0.2),
    Absorber(RectangleShape(Position(0.0, 3.1), Size(0.5, 0.5)), absorptionPressure=0.0, absorptionVelocity=0.2),
    Absorber(RectangleShape(Position(4.7, 0.0), Size(0.4, 1.9)), absorptionPressure=0.0, absorptionVelocity=0.2),
    Absorber(RectangleShape(Position(1.5, 3.45), Size(1.6, 0.15)), absorptionPressure=0.0, absorptionVelocity=0.2),
    Absorber(RectangleShape(Position(1.5, 0.0), Size(1.6, 0.15)), absorptionPressure=0.0, absorptionVelocity=0.2)
    #Absorber(RectangleShape(Position(0.2, 0.2), Size(0.5, 0.5)), absorptionPressure=0.0, absorptionVelocity=0.2),
    #Absorber(EllipseShape(Position(1.8, 1.6), radiusX=0.3, radiusY=0.5), absorptionPressure=0.0, absorptionVelocity=0.2),
    #Absorber(PolygonShape(Position(2.5, 2.5), vertices=[[0, 0], [0.5, 0.2], [0.3, 0.8], [0, 0.6]]), absorptionPressure=0.0, absorptionVelocity=0.2)
]

absorberPatches = [ax.add_patch(absorber.getPatch()) for absorber in absorbers]

class Speaker:
    def __init__(self, name, shape, frequency, volume, minFrequency, maxFrequency):
        self.name = name
        self.shape = shape
        self.frequency = frequency
        self.volume = volume
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.omega = 2 * np.pi * self.frequency
        self.currentPhase = 0

        self.shape.position = Position(
            self.shape.position.x + wallThickness,
            self.shape.position.y + wallThickness
        )

    def updateFrequency(self, frequency):
        self.frequency = frequency
        self.omega = 2 * np.pi * self.frequency

    def updateVolume(self, volume):
        self.volume = volume

    def updatePressure(self, pressureField, timeStepSize):
        if not (self.minFrequency <= self.frequency <= self.maxFrequency):
            return

        amplitude = REFERENCE_PRESSURE * (10 ** (self.volume / 20))
        discretePositions = self.shape.getDiscretePositions()

        self.currentPhase += self.omega * timeStepSize
        self.currentPhase %= 2 * np.pi

        for x, y in discretePositions:
            pressureField[y, x] = amplitude * np.sin(self.currentPhase)

    def getPatch(self):
        if isinstance(self.shape, RectangleShape):
            startX, startY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            width, height = self.shape.size.width / posStepSize, self.shape.size.height / posStepSize
            return Rectangle((startX - 0.6, startY - 0.6), width, height, color='orange', fill=False, linewidth=2)
        elif isinstance(self.shape, EllipseShape):
            centerX, centerY = self.shape.position.x / posStepSize, self.shape.position.y / posStepSize
            width, height = self.shape.radiusX * 2 / posStepSize, self.shape.radiusY * 2 / posStepSize
            return Ellipse((centerX, centerY), width, height, color='orange', fill=False, linewidth=2)
        elif isinstance(self.shape, PolygonShape):
            absoluteVertices = self.shape.vertices + np.array([self.shape.position.x, self.shape.position.y])
            return Polygon(absoluteVertices / posStepSize, color='orange', fill=False, linewidth=2)



speakers = [
    Speaker("Main Speaker", EllipseShape(Position(0.5, 1.8), 0.15, 0.15), frequency=100, volume=80.0, minFrequency=20.0, maxFrequency=20000.0),
    Speaker("Tweeter", EllipseShape(Position(3.0, 1.5), 0.2, 0.2), frequency=1000, volume=75.0, minFrequency=80.0, maxFrequency=20000.0),
    Speaker("Bass", EllipseShape(Position(4.0, 3.0), 0.1, 0.1), frequency=33.63, volume=85.0, minFrequency=20.0, maxFrequency=80.0),
]

speakerNames = [speaker.name for speaker in speakers]

speakerPatches = [ax.add_patch(speaker.getPatch()) for speaker in speakers]


resetButton = tk.Button(frameControls, text="Reset", command=lambda: resetSimulation(None))
resetButton.pack(side=tk.LEFT, padx=15)

stopButton = tk.Button(frameControls, text="Stop", command=lambda: toggleSimulation(None))
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
    pressureHistoryNp = np.array(pressureHistory) 
    maxPressure = np.amax(pressureHistoryNp, axis=0)
    minPressure = np.amin(pressureHistoryNp, axis=0)
    difPressure = abs(maxPressure - minPressure)/2
    pressure_dB = 20 * np.log10(difPressure / REFERENCE_PRESSURE + 1e-12) # + 1e-12 to avoid log(0)
    return pressure_dB

def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY):
    velocityFieldX[1:, :] -= timeStepSize / posStepSize * (pressureField[1:, :] - pressureField[:-1, :])
    velocityFieldY[:, 1:] -= timeStepSize / posStepSize * (pressureField[:, 1:] - pressureField[:, :-1])

    pressureField[:-1, :] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldX[1:, :] - velocityFieldX[:-1, :])
    pressureField[:, :-1] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldY[:, 1:] - velocityFieldY[:, :-1])


def applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY):
    for i in range(int(round(wallThickness / posStepSize))):
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

    wallReflexionLayer = int(round(wallThickness / posStepSize)) - 1  
    velocityFieldX[wallReflexionLayer+1, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldX[-wallReflexionLayer-2, wallReflexionLayer+1:-wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, wallReflexionLayer+1] *= 1 - 1.99 * wallReflectionCoefficient
    velocityFieldY[wallReflexionLayer+1:-wallReflexionLayer-2, -wallReflexionLayer-2] *= 1 - 1.99 * wallReflectionCoefficient

    velocityFieldX[0, :] = 0
    velocityFieldX[-1, :] = 0
    velocityFieldY[:, 0] = 0
    velocityFieldY[:, -1] = 0


def updateSimulation(pressureField, velocityFieldX, velocityFieldY):
    global simulatedTime

    updatePressureHistory(pressureField)
    calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY)
    applyBoundaryConditions(pressureField, velocityFieldX, velocityFieldY)

    for absorber in absorbers:
        absorber.applyAbsorption(pressureField, velocityFieldX, velocityFieldY)

    for speaker in speakers:
        speaker.updatePressure(pressureField, timeStepSize)
    simulatedTime += timeStepSize * 1000

def updateDisplayedField():
    global pressure_dB_cache
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
    elif selectedField == "dB Level":
        if pressureIndex % 100 < 4:
            pressure_dB_cache = calcPressure_dB()
        if pressure_dB_cache is not None and pressure_dB_cache.ndim == 2:
            image.set_array(pressure_dB_cache[:-1, :-1])
        norm = Normalize(vmin=50, vmax=90)
        image.set_norm(norm)
        image.set_clim(50, 90)
        cbar.set_label('Sound Pressure Level (dB)')

def updateLegends(*args):
    canvas.draw_idle()

fieldTarget.trace_add("write", updateLegends)

def createExclusionMask():
    mask = np.ones(pressureField.shape, dtype=bool)
    
    thickness = int(round(wallThickness / posStepSize))
    mask[:thickness, :] = False
    mask[-thickness-1:, :] = False
    mask[:, :thickness] = False
    mask[:, -thickness-1:] = False

    for absorber in absorbers:
        discrete_positions = absorber.shape.getDiscretePositions()
        for x, y in discrete_positions:
            mask[y, x] = False

    for speaker in speakers:
        discrete_positions = speaker.shape.getDiscretePositions()
        for x, y in discrete_positions:
            mask[y, x] = False

    return mask

exclusion_mask = createExclusionMask()

textElements = [
    ax.text(0.05, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # time
    ax.text(0.35, 0.99, '', transform=ax.transAxes, color=(0.3, 0.5, 1), fontsize=12, weight='bold', va='top'),  # max
    ax.text(0.43, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top'),  # maxValue
    ax.text(0.62, 0.99, '', transform=ax.transAxes, color='red', fontsize=12, weight='bold', va='top'),  # min
    ax.text(0.69, 0.99, '', transform=ax.transAxes, color='white', fontsize=12, weight='bold', va='top')  # minValue
]


def updateText(simulatedTime, pressure_dB_cache):
    selectedField = fieldTarget.get()
    if selectedField == "dB Level" and pressure_dB_cache is not None:
        max_dB = np.max(np.where(exclusion_mask, pressure_dB_cache, -np.inf))
        min_dB = np.min(np.where(exclusion_mask, pressure_dB_cache, np.inf))

        textElements[0].set_text(f'Time: {simulatedTime:2.2f} ms, ')
        textElements[1].set_text('Max: ')
        textElements[2].set_text(f'{max_dB:2.2f} dB, ')
        textElements[3].set_text('Min: ')
        textElements[4].set_text(f'{min_dB:2.2f} dB')
    else:
        textElements[0].set_text(f'Time: {simulatedTime:2.2f} ms')
        textElements[1].set_text('')
        textElements[2].set_text('')
        textElements[3].set_text('')
        textElements[4].set_text('')

def updateMarkers(pressure_dB_cache):
    selectedField = fieldTarget.get()
    if selectedField == "dB Level" and pressure_dB_cache is not None:
        valid_pressure_dB = np.where(exclusion_mask, pressure_dB_cache, -np.inf)
        max_dB_position = np.unravel_index(np.argmax(valid_pressure_dB, axis=None), valid_pressure_dB.shape)
        valid_pressure_dB = np.where(exclusion_mask, pressure_dB_cache, np.inf)
        min_dB_position = np.unravel_index(np.argmin(valid_pressure_dB, axis=None), valid_pressure_dB.shape)

        max_dB_marker.set_data([max_dB_position[1]], [max_dB_position[0]])
        min_dB_marker.set_data([min_dB_position[1]], [min_dB_position[0]])

        max_dB_marker.set_visible(True)
        min_dB_marker.set_visible(True)
    else:
        max_dB_marker.set_visible(False)
        min_dB_marker.set_visible(False)


def update(frame):
    global pressureField, velocityFieldX, velocityFieldY, animRunning
    if animRunning:
        for _ in range(4):
            updateSimulation(pressureField, velocityFieldX, velocityFieldY)

        updateText(simulatedTime, pressure_dB_cache)
        updateMarkers(pressure_dB_cache)

    updateDisplayedField()
    return [image] + speakerPatches + textElements + [max_dB_marker, min_dB_marker] + wallRects + absorberPatches

controlAllSpeakersFlag = tk.BooleanVar(value=True)

controlAllSpeakersToggle = tk.Checkbutton(frameControls, text="Control All Speakers", variable=controlAllSpeakersFlag)
controlAllSpeakersToggle.pack(side=tk.LEFT, padx=5)

selectedSpeaker = tk.StringVar()
selectedSpeaker.set(speakerNames[0])

speakerMenu = tk.OptionMenu(frameControls, selectedSpeaker, *speakerNames)
speakerMenu.pack(side=tk.LEFT, padx=5)

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

def updateSelectedSpeaker(*args):
    if not controlAllSpeakersFlag.get():  # Nur bei Einzelsteuerung aktualisieren
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
    if controlAllSpeakersFlag.get():
        for speaker in speakers:
            speaker.updateFrequency(newFrequency)
    else:
        index = speakerNames.index(selectedSpeaker.get())
        speakers[index].updateFrequency(newFrequency)

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
    if controlAllSpeakersFlag.get():
        for speaker in speakers:
            speaker.updateVolume(newVolume)
    else:
        index = speakerNames.index(selectedSpeaker.get())
        speakers[index].updateVolume(newVolume)

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

speakerFrequencySlider.config(command=updateFrequency)
speakerVolumeSlider.config(command=updateVolume)
speakerFrequencyEntry.bind("<Return>", updateFrequencyFromEntry)
speakerVolumeEntry.bind("<Return>", updateVolumeFromEntry)

updateSelectedSpeaker()

animation = FuncAnimation(fig, update, blit=True, interval=1, cache_frame_data=False)

root.mainloop()
