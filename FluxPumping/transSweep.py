import numpy as np
import qutip as qt
# %%
# Parameters
Nmax = 5
omegaR = 8e9*2*np.pi
omegaQ = 7.6e9*2*np.pi
g = 129e6*2*np.pi
kappa = 0.37e6
gamma = 1/10e-6
# Operators
a = qt.tensor(qt.destroy(Nmax), qt.qeye(2))
adag = a.dag()
sx = qt.tensor(qt.qeye(Nmax), qt.sigmax())
sy = qt.tensor(qt.qeye(Nmax), qt.sigmay())
sz = qt.tensor(qt.qeye(Nmax), qt.sigmaz())
sp = qt.tensor(qt.qeye(Nmax), qt.sigmap())
sm = qt.tensor(qt.qeye(Nmax), qt.sigmam())
# Hamiltonian
H0 = (omegaR-omegaQ)*adag*a + g*(adag*sm + a*sp)
# operators for dressed qubit and cavity
eigvals, eigvecs = H0.eigenstates()


def transmission(omega, ampD, ampP):
    # omega[0]: cavity drive frequency
    # omega[1]: flux pump frequency
    # ampD: cavity drive amplitude
    # ampP: flux pump frequency
    # Returns: expectation value of adag*a of the steady state
    omegaD = omega[0]
    omegaP = omega[1]

    Ht = [(omegaR-omegaD)*adag*a + (omegaQ-omegaD)/2*sz +
          g*(adag*sm + a*sp) + ampD*(a+adag),
          [ampP*sz, lambda t, args: np.sin(args['omegaP']*t)]]
    # Calculate propagator for one pump period T = 2pi/omegaP
    U = qt.propagator(H=Ht, t=2*np.pi/omegaP,
                      c_op_list=[np.sqrt(kappa)*a, np.sqrt(gamma)*sm],
                      args={'omegaP': omegaP})
    # Calucate steady state density matrix
    rhoss = qt.propagator_steadystate(U)
    # Return photon number for steady state
    return np.real(qt.expect(adag*a, rhoss))
# %%
omegaQD = (omegaQ/2+eigvals[0]) - (-omegaQ/2+eigvals[1])
omegaRD = (omegaQ/2+eigvals[3]) - (-omegaQ/2+eigvals[1])
deltaD = omegaRD-omegaQD

omegaDlist = np.linspace(omegaRD-30e6*2*np.pi, omegaRD+30e6*2*np.pi, 301)
omegaPlist = np.linspace(deltaD-40e6*2*np.pi, deltaD+40e6*2*np.pi, 81)
ampD = 0.5*kappa
ampP = 50e6*2*np.pi

# Construct all combinations of omegaD and omegaP for parallelization
omegazip = []
for omegaP in omegaPlist:
    omegazip.extend(zip(omegaDlist, np.repeat(omegaP, len(omegaDlist))))
# Divide into batches
bsize = 4*len(omegaDlist)
nbatch = int(np.ceil(len(omegazip)/bsize))
filename = 'transSweep.dat'
result = []
# Run simulation and save data batch by batch
for ii in range(0, nbatch):
    print('Calculating batch %d of %d ...' % (ii+1, nbatch))
    result.extend(qt.parallel_map(transmission,
                                  omegazip[ii*bsize:(ii+1)*bsize],
                                  task_args=(ampD, ampP),
                                  num_cpus=40, progress_bar=True))
    qt.file_data_store(filename, np.array(result).reshape((-1, 1)),
                       numtype='real')
    print('Saved to %s\n' % filename)

# Save data
# data is an numpy array with M rows and N columns
# M = len(omegaPlist)+1, N = len(omegaDlist)+1
# data[0, 1:] = omegaDlist, data[1:, 0] = omegaPlist
# data[1:, 1:] = transmission data
result = np.array(result).reshape((len(omegaPlist), len(omegaDlist)))
data = np.vstack((omegaDlist, result))
data = np.hstack((np.append([0], omegaPlist).reshape(-1, 1), data))
qt.file_data_store(filename, data, numtype='real')
print('Finished.')
