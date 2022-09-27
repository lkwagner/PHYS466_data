from HW2_solutions import * 
import matplotlib.pyplot as plt


if __name__=="__main__":
    length=20
    coordinates = np.asarray([[0., 0., 0.],[0.,0.,2**(1/6)+0.1]])

    velocities = np.asarray([[0., 0., 0.],[0., 0., 0.]])

    # tables required to compute quantities like forces, energies
    displacements = displacement_table(coordinates, length)
    mass = 48
    timestep = 0.01
    cutoff = length/4
    T = 100
    nsteps = int(T/timestep)

    distances = np.linalg.norm(displacements, axis=-1)
    KE = []
    PE = []
    for _ in range(nsteps):
        coordinates, velocities, displacements, distances = advance(coordinates,\
                velocities, mass, timestep, displacements, distances, cutoff,\
                length)
        PE.append(potential(distances, cutoff))
        KE.append(kinetic(mass, velocities))


    fig, axes = plt.subplots(1,3, figsize=(16,9))
    axes[0].plot(PE)
    axes[0].set_ylabel("Potential energy")
    axes[1].plot(KE)
    axes[1].set_ylabel("Kinetic energy")
    axes[2].plot(np.asarray(PE)+np.asarray(KE))
    axes[2].set_ylabel("Total energy")
    plt.tight_layout()
    plt.savefig("twoparticle.pdf", bbox_inches='tight')