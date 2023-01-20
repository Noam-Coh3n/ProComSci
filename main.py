import numpy as np
from diver import Diver
import matplotlib.pyplot as plt
import integration_error as err

STEP_SIZE = 0.005

# Setup variables
wind_list = np.array([[-30, 0], [10, 0], [30, 60]])
air_pres_list = [100000]
temp_list = [280]
h_vals = np.arange(0.001, 0.13, 0.01)

# Add a diver.
myDiver = Diver(x=np.array([0.,0.,3600.]), vel=np.array([0.,-600.,0.]),
            wind=wind_list, air_pressure=air_pres_list, temperature=temp_list,
            h_shute=200, stepsize=STEP_SIZE)
# Run model.
myDiver.simulate_trajectory('Pred-corr')

# Plot location.
for i, direction in enumerate(['x', 'y', 'z']):
    plt.figure(f"{direction}-direction, location")
    time = [i * STEP_SIZE for i in range(len(myDiver.x_list))]

    # Location.
    plt.plot(time, [x[i] for x in myDiver.x_list], label="location")

    plt.legend()
    plt.xlabel("time (sec)")

# Plot velocity and acceleration.
for i, direction in enumerate(['x', 'y', 'z']):
    plt.figure(f"{direction}-direction, velocity and acceleration")
    time = [i * STEP_SIZE for i in range(len(myDiver.x_list))]

    # Location.
    plt.plot(time, [v[i] for v in myDiver.v_list], label="velocity")
    plt.plot(time, [a[i] for a in myDiver.a_list], label="acceleration")

    plt.legend()
    plt.xlabel("time (sec)")

plt.show()

err.simulate_error(h_vals)

# from visual import Visual

# myVisual = Visual(wind_list, air_pres_list, temp_list)
# myVisual.add_diver(myDiver)
# myVisual.select_line_interval(300)
# myVisual.run(slowed=10)


# Pyplot en Pygame kunnen niet tegenlijkertijd, dan krijg je een error.