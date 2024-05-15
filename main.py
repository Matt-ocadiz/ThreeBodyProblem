import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import scipy.integrate
import numpy as np

# Define universal gravitation constant
G = 6.67408e-11  # N-m^2/kg^2

# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri


# Net constants
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1 = 1.1  # Alpha Centauri A
m2 = 0.907  # Alpha Centauri B
#Mass of the Third Star
m3 = 1.0 #Third Star


# Define initial position vectors
r1 = [-0.5, 0, 0]  # m
r2 = [0.5, 0, 0]  # m
#Position of the Third Star
r3 = [0,1,0] #m


# Convert pos vectors to arrays
r1 = np.array(r1, dtype="float64")
r2 = np.array(r2, dtype="float64")
r3 = np.array(r3,dtype="float64")
# Find Centre of Mass
# Update COM formula
r_com = (m1*r1+m2*r2+m3*r3)/(m1+m2+m3)

# Define initial velocities
v1 = [0.01, 0.01, 0]  # m/s
v2 = [-0.05, 0, -0.1]  # m/s

# Convert velocity vectors to arrays
v1 = np.array(v1, dtype="float64")
v2 = np.array(v2, dtype="float64")

#Velocity of the Third Star
v3=[0,-0.01,0]
v3=np.array(v3,dtype="float64")
# Find velocity of COM

#Update velocity of COM formula
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

# A function defining the equations of motion
def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs
# Package initial parameters
#Package initial parameters
init_params=np.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=np.linspace(0,20,500) #20 orbital periods and 500 points

# Define time span
time_span = np.linspace(0, 8, 500)  # 8 orbital periods and 500 points

# Run the ODE solver
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

# Extract solutions
r1_sol = three_body_sol[:,:3]
r2_sol = three_body_sol[:,3:6]
r3_sol = three_body_sol[:,6:9]

# Create figure
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")

# Initialize the lines and points for animation
line1, = ax.plot([], [], [], color="darkblue", label="Alpha Centauri A")
line2, = ax.plot([], [], [], color="tab:red", label="Alpha Centauri B")
point1, = ax.plot([], [], [], 'o', color="darkblue")
point2, = ax.plot([], [], [], 'o', color="tab:red")
line3, = ax.plot([], [], [], color="green", label="Third Star")
point3, = ax.plot([], [], [], 'o', color="green")

# Set the axes properties
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a Three-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

# Animation control variables
is_paused = False

def on_pause(event):
    global is_paused
    is_paused = not is_paused
    if is_paused:
        ani.event_source.stop()
    else:
        ani.event_source.start()

# Update function for animation
def update(num, r1_sol, r2_sol, r3_sol, line1, line2, line3, point1, point2, point3):
    if not is_paused:
        line1.set_data(r1_sol[:num, 0], r1_sol[:num, 1])
        line1.set_3d_properties(r1_sol[:num, 2])
        line2.set_data(r2_sol[:num, 0], r2_sol[:num, 1])
        line2.set_3d_properties(r2_sol[:num, 2])
        line3.set_data(r3_sol[:num, 0], r3_sol[:num, 1])
        line3.set_3d_properties(r3_sol[:num, 2])

        point1.set_data(r1_sol[num, 0], r1_sol[num, 1])
        point1.set_3d_properties(r1_sol[num, 2])
        point2.set_data(r2_sol[num, 0], r2_sol[num, 1])
        point2.set_3d_properties(r2_sol[num, 2])
        point3.set_data(r3_sol[num, 0], r3_sol[num, 1])
        point3.set_3d_properties(r3_sol[num, 2])
    return line1, line2, line3, point1, point2, point3


# Create the animation object
ani = FuncAnimation(fig, update, frames=len(time_span),
                    fargs=(r1_sol, r2_sol, r3_sol, line1, line2, line3, point1, point2, point3), interval=50,
                    blit=False)

# Add pause button
ax_pause = plt.axes([0.7, 0.02, 0.1, 0.075])
btn_pause = Button(ax_pause, 'Pause/Play')
btn_pause.on_clicked(on_pause)

plt.show()
