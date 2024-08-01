import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

posStepSize = 0.1
timeStepSize = 0.0001
SPEED_OF_SOUND = 346.3  # speed of sound in air in m/s at 25°C
numSteps = 300

roomWidth = 10.0  # in meters
roomHeight = 10.0  # in meters

numDiscretePosX = int(roomWidth / posStepSize)
numDiscretePosY = int(roomHeight / posStepSize)

pressure = np.zeros((numDiscretePosX, numDiscretePosY))

fig, ax = plt.subplots()
image = ax.imshow(pressure, cmap='viridis', vmin=-0.1, vmax=0.1, animated=True)
ax.set_title('Sound Pressure Simulation')
ax.set_xlabel('X in meter')
ax.set_ylabel('Y in meter')

def update(frame):
	global pressure
	image.set_array(pressure)
	return [image]


animation = FuncAnimation(fig, update, frames=numSteps, blit=True, interval=20)
plt.show()
