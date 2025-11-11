from qiskit import *
from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

from matplotlib.patches import Circle

from qiskit.visualization import *

import numpy as np
import imageio
from PIL import Image




def animate_rotation_1angle_simple(f_matrix, angle_range, seconds, fps, loop, reverse):


    theta_min, theta_max = angle_range
    dtheta = (theta_max - theta_min) / (fps*seconds)

    images = []
    for i in range(fps*seconds):
        qc = QuantumCircuit(3)
        qc.rx(np.pi/4, 0)
        qc.rx(np.pi/4, 1)
        qc.rx(np.pi/4, 2)
        qc.rx(i * dtheta, 0)
        qc.ry(i * dtheta, 1)
        qc.rz(i * dtheta, 2)
        f = plot_bloch_multivector( Statevector.from_instruction(qc) )

        f.canvas.draw()

        img = np.array(f.canvas.get_renderer().buffer_rgba())  # Get RGBA image
        img = Image.fromarray(img)  # Convert to PIL image
        images.append(img)  # Store image

        plt.close(f)

    if reverse:
        rev_imgs = images.copy()
        rev_imgs.reverse()
        images += rev_imgs

    imageio.mimsave("animation.mp4", images, fps=60)  # Save as GIF

    print("Animation saved as animation.gif")

        


    # visualize_transition(qc, saveas="Rx_2pi.mp4", fpg=288, spg=2)

    # fig, ax = plt.subplots()

    # artists =[]

    # for i in range(points_per_cycle):
    #     qc = QuantumCircuit(1)
    #     qc.rx(dtheta*i, 0)

    #     sv = Statevector.from_instruction(qc)

    #     # f = sv.draw(output='bloch')
    #     f = plot_bloch_multivector(sv)
    #     artists.append(f)

    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
    # plt.show()

        
    plt.waitforbuttonpress()

    # bloch_sphere = Circle((0,0), radius=1, facecolor=(1,1,1,0.5))

    # # Plot
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    # ax.add_patch(bloch_sphere)
    # art3d.pathpatch_2d_to_3d(bloch_sphere, z=0, zdir="z")


    # # Plot axes


    # # Plot Sphere
    # sphere = lambda t0, t1: [np.sin(t0)*np.cos(t1), np.sin(t0)*np.sin(t1), np.cos(t0)]
    # coordenadas = [sphere(t0, t1) for t0 in np.arange(0,2*np.pi, np.pi/50) for t1 in np.arange(0, 2*np.pi, np.pi/50)]

    # X, Y, Z = np.array(coordenadas).T

    # ax.plot(X, Y, Z, '-', color=(0.5,0.5,0.5,0.5))

    # # Plot dot
    # ax.plot(0,0,1, 'o', color='r')
    
    # plt.gca().set_aspect('equal')

    # plt.show()



def animate_rotation_per_angle(f_matrix, angle_ranges, points_per_cycle, loop, reverse):
    for theta_min, theta_max in angle_ranges:

        dtheta = (theta_max - theta_min) / points_per_cycle

        f_matrix



if __name__=='__main__':
    animate_rotation_1angle_simple(None, (0,2*np.pi), 5, 60, True, True)