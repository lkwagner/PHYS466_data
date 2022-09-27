import numpy as np

def minimum_image(r, L):
    """
    required for: displacement_table(), advance()
    args:
        r: array of any shape
        L: side length of cubic box
    returns:
        array of the same shape as r,
        the minimum image of r
    """
    return r - L*np.round(r / L)

def cubic_lattice(tiling, L):
    """
    required for: initialization

    args:
        tiling (int): determines number of coordinates,
        by tiling^3
        L (float): side length of simulation box
    returns:
        array of shape (tiling**3, 3): coordinates on a cubic lattice,
        all between -0.5L and 0.5L
    """
    coord = []
    for x in range(tiling):
        for y in range(tiling):
            for z in range(tiling):
                coord.append([x,y,z])
    coord = np.array(coord)/float(tiling)
    coord -= 0.5
    return coord*L

def get_temperature(mass, velocities):
    """
    calculates the instantaneous temperature
    required for: initial_velocities()

    args:
        mass (float): mass of particles;
        it is assumed all particles have the same mass
        velocities (array): velocities of particles,
        assumed to have shape (N, 3)
    returns:
        float: temperature according to equipartition
    """
    N = len(velocities)
    deg_of_freedom = 3*N
    total_vsq = np.einsum('ij,ij', velocities, velocities)
    return mass*total_vsq / deg_of_freedom

def initial_velocities(N, m, T):
    """
    initialize velocities at a desired temperature
    required for: initialization

    args:
        N (int): number of particles
        m (float): mass of particles
        T (float): desired temperature
    returns:
        array: initial velocities, with shape (N, 3)
    """
    velocities = np.random.rand(N, 3)
    #center velocities
    new_v = velocities - 0.5
    #zero the total velocity
    total_v = np.sum(new_v, axis=0)
    new_v -= total_v / N
    #get the right temperature
    current_temp = get_temperature(m, new_v)
    factor  = np.sqrt(T / current_temp)
    new_v *= factor
    return new_v

def displacement_table(coordinates, L):
    """
    required for: force(), advance()

    args:
        coordinates (array): coordinates of particles,
        assumed to have shape (N, 3)
        e.g. coordinates[3,0] should give the x component
        of particle 3
        L (float): side length of cubic box,
        must be known in order to compute minimum image
    returns:
        array: table of displacements r
        such that r[i,j] is the minimum image of
        coordinates[i] - coordinates[j]
    """
    table = coordinates[:,np.newaxis,:] - coordinates[np.newaxis,:,:]
    return minimum_image(table, L)

def kinetic(m, v):
    """
    required for measurement

    args:
        m (float): mass of particles
        v (array): velocities of particles,
        assumed to be a 2D array of shape (N, 3)
    returns:
        float: total kinetic energy
    """
    total_vsq = np.einsum('ij,ij', v, v)
    return 0.5*m*total_vsq

def potential(dist, rc):
    """
    required for measurement

    args:
        dist (array): distance table with shape (N, N)
        i.e. dist[i,j] is the distance
        between particle i and particle j
        in the minimum image convention
        note that the diagonal of dist can be zero
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, the pair potential between
        i and j will be 0
    returns:
        float: total potential energy
    """
    r = np.copy(dist)
    r[np.diag_indices(len(r))] = np.inf
    v = 4*np.power(r, -6)*(np.power(r, -6) - 1)
    vc = 4*np.power(rc, -6)*(np.power(rc, -6) - 1)
    v[r < rc] -= vc #shift
    v[r >= rc] = 0 #cut
    return 0.5*np.sum(v)

def force(disp, dist, rc):
    """
    required for: advance()

    args:
        disp (array): displacement table,
        with shape (N, N, 3)
        dist (array): distance table, with shape (N, N)
        can be calculated from displacement table,
        but since there is a separate copy available
        it is just passed in here
        rc (float): cutoff distance for interaction
        i.e. if dist[i,j] > rc, particle i will feel no force
        from particle j
    returns:
        array: forces f on all particles, with shape (N, 3)
        i.e. f[3,0] gives the force on particle i
        in the x direction
    """
    r = np.copy(dist)
    r[np.diag_indices(len(r))] = np.inf
    magnitude = 1./r**8
    magnitude *= 2./r**6 - 1
    magnitude *= 24
    magnitude[r >= rc] = 0
    val = magnitude[:,:,np.newaxis]*disp
    val = np.sum(val, axis=1)
    return val

def advance(pos, vel, mass, dt, disp, dist, rc, L):
    """
    advance system according to velocity verlet

    args:
        pos (array): coordinates of particles
        val (array): velocities of particles
        mass (float): mass of particles
        dt (float): timestep by which to advance
        disp (array): displacement table
        dist (array): distance table
        rc (float): cutoff
        L (float): length of cubic box
    returns:
        array, array, array, array:
        new positions, new velocities, new displacement table,
        and new distance table
    """
    accel = force(disp, dist, rc) / mass
    #move
    vel_half = vel + 0.5*dt*accel
    pos_new = pos + dt*vel_half
    pos_new = minimum_image(pos_new, L)
    disp_new = displacement_table(pos_new, L)
    dist_new = np.linalg.norm(disp_new, axis=-1)
    #repeat force calculation for new pos
    accel = force(disp_new, dist_new, rc) / mass
    #finish move
    vel_new = vel_half + 0.5*dt*accel
    return pos_new, vel_new, disp_new, dist_new

def advance2(pos, vel, mass, dt, disp, dist, rc, L):
    """
    advance system according to velocity verlet - Taylor expansion

    args:
        pos (array): coordinates of particles
        val (array): velocities of particles
        mass (float): mass of particles
        dt (float): timestep by which to advance
        disp (array): displacement table
        dist (array): distance table
        rc (float): cutoff
        L (float): length of cubic box
    returns:
        array, array, array, array:
        new positions, new velocities, new displacement table,
        and new distance table
    """
    accel = force(disp, dist, rc) / mass
    #move
    vel_new = vel + dt*accel
    pos_new = pos + dt*vel_new + 0.5*accel*dt**2
    pos_new = minimum_image(pos_new, L)
    disp_new = displacement_table(pos_new, L)
    dist_new = np.linalg.norm(disp_new, axis=-1)

    return pos_new, vel_new, disp_new, dist_new

def thermostat(v, m, T, prob):
    """
    args:
        v (array): velocities of particles
        m (float): mass of particles
        T (float): target temperature
        prob (float): collision probability,
            a number of between 0 and 1
    returns:
        nothing, but v is modified
    """


def run_lj_solid(
        num_particles = 64,
    temperature = 0.728,
    length = 4.2323167,
    cutoff = 2.08,
    mass = 48,
    timestep = 0.01,
    T = 1000, # total time
    method='verlet'
):
    nsteps = int(T/timestep) 

    # system
    coordinates = cubic_lattice(4, length)

    velocities = initial_velocities(num_particles, mass, temperature)

    # tables required to compute quantities like forces, energies
    displacements = displacement_table(coordinates, length)
    distances = np.linalg.norm(displacements, axis=-1)
    KE = []
    PE = []
    for _ in range(nsteps):
        if method=='verlet':
            coordinates, velocities, displacements, distances = advance(coordinates,\
                    velocities, mass, timestep, displacements, distances, cutoff,\
                    length)
        elif method=='euler':
            coordinates, velocities, displacements, distances = advance2(coordinates,\
                    velocities, mass, timestep, displacements, distances, cutoff,\
                    length)

        PE.append(potential(distances, cutoff))
        KE.append(kinetic(mass, velocities))

    return {'PE':np.asarray(PE), 'KE':np.asarray(KE) } 



def average(data):
    """
    Average with error bars and correlations
    args: 
      data: timeseries data with warmup removed

    """
    Neq = len(data)
    v = np.var(data, ddof=1)
    a = np.mean(data)
    cor = [np.mean((data[i:]-a)*(data[:-i]-a))/v for i in range(1,100)]
    kappa = 1 + 2.0*np.sum(cor)
    err = np.sqrt(v*kappa/Neq)
    return a, err, kappa, v