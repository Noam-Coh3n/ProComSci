import numpy as np
import matplotlib.pyplot as plt

TIME_INTERVALS = 0.1 # of a sec

class Diver:

    def __init__(self, initial_pos : np.array, initial_vel : np.array):
        self.pos = initial_pos
        self.vel = initial_vel

        self.massa = 80000 #gram?
        self.drag_coeff = 1
        self.air_density = 1.293
        self.projected_area = 0.7

    def _get_z_acceleration(self):
        speed_rel_to_air = self.vel[2] # als air geen verticale snelheid heeft
        gravity_force = self.massa * 9.81
        drag_force = 0.5 * self.drag_coeff * self.air_density \
                          * speed_rel_to_air**2 * self.projected_area
        acceleration = (drag_force - gravity_force ) / self.massa
        return acceleration

    def move(self):
        """ Update the position with help from the velocity 
            and update the velocity with help from the acceleration.
        """
        self.pos += self.vel * TIME_INTERVALS
        self.vel += np.array([0, 0, self._get_z_acceleration()]) * TIME_INTERVALS
        
    # For a human, the drag coefficient Cd is about 1 in a belly down, horizontal orientation and 0.7 in head down position.
    # https://owlcation.com/stem/Drag-Force-and-the-Terminal-Velocity-of-a-Human
    # A is the projected area of the object, or area cross-section (~0.18 m2 for head down position, ~0.7 for belly-to-earth position)[
    # https://en.wikipedia.org/wiki/Speed_skydiving#Terminal_Velocity

def go():
    diver = Diver(np.array([0., 0., 3600.]), np.array([0.,0.,0.]))

    pos = [[], [], []]
    vel = [[], [], []]
    while True:

        if diver.pos[2] <= 0: break 
        diver.move()

        for i in range(len(pos)):
            pos[i].append(diver.pos[i])
            vel[i].append(diver.vel[i])

    fig, axs = plt.subplots(1, 2)

    axs[0].plot([i*TIME_INTERVALS for i in range(len(pos[2]))], pos[2], label="Height")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("height (meters)")

    axs[1].plot([i*TIME_INTERVALS for i in range(len(vel[2]))], [-x for x in vel[2]], label="Velocity")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Velocity (meters)")
    plt.show()
go()
            




    