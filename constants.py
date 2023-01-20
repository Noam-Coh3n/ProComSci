from collections import namedtuple

m_diver = 83
m_air = 4.81 * 10**(-26)
g = 9.81
kB = 1.38 * 10**(-23)

sides = namedtuple('Sides', ['front', 'side'])
C_diver = sides(1.18, 1.11)
C_chute = sides(1.68, 0.35)
A_diver = sides(0.55, 0.38)
A_chute = sides(47.8, 23.9)
