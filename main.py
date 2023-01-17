import numpy as np
from diver import Diver

wind_list = np.array([[-30, 0], [10, 0]])
air_pres_list = [100000]
temp_list = [280]


myDiver = Diver(x=np.array([0.,0.,3600.]), vel=np.array([0.,600.,0.]), 
            wind=wind_list, air_pressure=air_pres_list, temperature=temp_list,
            h_shute=200)
# Run
while True:
    if myDiver.x[2] <= 0: 
        break
    myDiver.move()

# print(myDiver.v)

from visual import Visual

myVisual = Visual(wind_list, air_pres_list, temp_list)
myVisual.add_diver(myDiver)
myVisual.select_line_interval(300)
myVisual.run(slowed=10)
