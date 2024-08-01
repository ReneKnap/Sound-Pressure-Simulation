import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

posStepSize = 0.05
timeStepSize = 0.0001
SPEED_OF_SOUND = 346.3  # speed of sound in air in m/s at 25°C
numSteps = 300

roomWidth = 10.0  # in meters
roomHeight = 10.0  # in meters

numDiscretePosX = int(roomWidth / posStepSize)
numDiscretePosY = int(roomHeight / posStepSize)

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


def calcFiniteDifferenceTimeDomain(pressureField, velocityFieldX, velocityFieldY, frame):
    # update velocity fields
    velocityFieldX[1:, :] -= timeStepSize / posStepSize * (pressureField[1:, :] - pressureField[:-1, :])
    velocityFieldY[:, 1:] -= timeStepSize / posStepSize * (pressureField[:, 1:] - pressureField[:, :-1])

    # reflective boundary conditions
    velocityFieldX[0, :] = 0
    velocityFieldX[-1, :] = 0
    velocityFieldY[:, 0] = 0
    velocityFieldY[:, -1] = 0

    # update pressure field
    pressureField[:-1, :] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldX[1:, :] - velocityFieldX[:-1, :])
    pressureField[:, :-1] -= timeStepSize * SPEED_OF_SOUND**2 / posStepSize * (velocityFieldY[:, 1:] - velocityFieldY[:, :-1])

    # simple sound source
    pressureField[sourcePosX, sourcePosY] += np.sin(omega * frame * timeStepSize)

    return pressureField, velocityFieldX, velocityFieldY

def update(frame):
    global pressureField, velocityFieldX, velocityFieldY

    pressureField, velocityFieldX, velocityFieldY = calcFiniteDifferenceTimeDomain(
        pressureField, velocityFieldX, velocityFieldY, frame
    )

    image.set_array(pressureField)
    return [image]


animation = FuncAnimation(fig, update, frames=numSteps, blit=True, interval=20)
plt.show()
