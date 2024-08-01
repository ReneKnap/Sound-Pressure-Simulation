import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



posStepSize = 0.1
timeStepSize = 0.0001
SPEED_OF_SOUND = 346.3  # speed of sound in air in m/s at 25°C

roomWidth = 10.0  # in meters
roomHeight = 10.0  # in meters

numDiscretePosX = int(roomWidth / posStepSize) + 1
numDiscretePosY = int(roomHeight / posStepSize) + 1

pressureField = np.zeros((numDiscretePosX, numDiscretePosY))
velocityFieldX = np.zeros((numDiscretePosX, numDiscretePosY))
velocityFieldY = np.zeros((numDiscretePosX, numDiscretePosY))

sourcePosX = numDiscretePosX // 2
sourcePosY = numDiscretePosY // 2
frequency = 100  # in Hz
omega = 2 * np.pi * frequency


fig, ax = plt.subplots()
image = ax.imshow(pressureField, cmap='viridis', vmin=-0.1, vmax=0.1, animated=True)
ax.set_title('Sound PressureField Simulation')
ax.set_xlabel('X in meter')
ax.set_ylabel('Y in meter')

# Set ticks and labels for x and y axes
xTicks = np.arange(0, numDiscretePosX, int(1 / posStepSize))
xLabels = np.round(xTicks * posStepSize, 1)
ax.set_xticks(xTicks)
ax.set_xticklabels(xLabels)

yTicks = np.arange(0, numDiscretePosY, int(1 / posStepSize))
yLabels = np.round(yTicks * posStepSize, 1)
ax.set_yticks(yTicks)
ax.set_yticklabels(yLabels)


def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, frame):
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
    pressureField[sourcePosX, sourcePosY] += np.sin(omega * frame * timeStepSize)



    return pressureField, velocityFieldX, velocityFieldY

def update(frame):
    global pressureField, velocityFieldX, velocityFieldY

    pressureField, velocityFieldX, velocityFieldY = calcFiniteDifferenceTimeDomain(
        pressureField, velocityFieldX, velocityFieldY, frame)

    image.set_array(pressureField[:-1, :-1])
    return [image]


animation = FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)
plt.show()
