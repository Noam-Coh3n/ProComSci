import numpy as np
from diver import Diver
import matplotlib.pyplot as plt

# Setup variables
wind_list = np.array([[-30, 0], [10, 0], [30, 60]])
air_pres_list = [100000]
temp_list = [280]

# Add a diver.
myDiver = Diver(x=np.array([0.,0.,3600.]), vel=np.array([0.,-600.,0.]), 
            wind=wind_list, air_pressure=air_pres_list, temperature=temp_list,
            h_shute=200)
# Run model.
while True:
    if myDiver.x[2] <= 0: 
        break
    myDiver.move()

# Plot.
plt.figure(300)
plt.plot([x[2] for x in myDiver.x_list])
plt.show()


# from visual import Visual

# myVisual = Visual(wind_list, air_pres_list, temp_list)
# myVisual.add_diver(myDiver)
# myVisual.select_line_interval(300)
# myVisual.run(slowed=10)
