import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# %%
sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
sy1 = qt.tensor(qt.sigmay(), qt.qeye(2))
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
sp1 = qt.tensor(qt.sigmap(), qt.qeye(2))
sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))

sx2 = qt.tensor(qt.qeye(2), qt.sigmax())
sy2 = qt.tensor(qt.qeye(2), qt.sigmay())
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
sp2 = qt.tensor(qt.qeye(2), qt.sigmap())
sm2 = qt.tensor(qt.qeye(2), qt.sigmam())

e_op_list = []
e_op_list.extend((sx1, sy1, sz1))
e_op_list.extend((sx2, sy2, sz2))

# Detuning between qubit 1 and 2
Delta12 = 100*2*np.pi
# Static coupling between qubit 1 and 2
J12 = 7*2*np.pi
# m2 sets the strength of spurious drive on qubit 2
m1 = 1.0
m2 = 0.2
# Drive strength on qubit 1
Omega = 20*2*np.pi

# (Time indenpendent) Hamiltonian in frame rotating at omega 2
Hp = Delta12/2*sz1 + J12*(sp1*sm2 + sm1*sp2) + m1*Omega*sx1 + m2*Omega*sx2
Hm = Delta12/2*sz1 + J12*(sp1*sm2 + sm1*sp2) - m1*Omega*sx1 - m2*Omega*sx2

# (Time dependent) Hamiltonian in frame rotating at omega 1
H0 = -Delta12/2*sz2 + J12*(sp1*sm2 + sm1*sp2)
Htp = [H0, [Omega*(m1*sp1+m2*sp2), lambda t, args: np.exp(1j*Delta12*t)],
       [Omega*(m1*sm1+m2*sm2), lambda t, args: np.exp(-1j*Delta12*t)]]
Htm = [H0, [-Omega*(m1*sp1+m2*sp2), lambda t, args: np.exp(1j*Delta12*t)],
       [-Omega*(m1*sm1+m2*sm2), lambda t, args: np.exp(-1j*Delta12*t)]]

# Time axis for simulation
dt = 0.00025
endtime = 1
tlist_simple = np.linspace(0.0, endtime, 1001)
tlist_echo = np.linspace(0.0, endtime, 101)

# Initial states
# |00>
psidown = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
# |10>
psiup = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
# |+0>
psiplus = qt.tensor((qt.basis(2, 1) + qt.basis(2, 0))/np.sqrt(2),
                    qt.basis(2, 1))


def CR(tlist, psi0):
    # Simple CR gate
    output = qt.mesolve(Hp, psi0, tlist, [], e_op_list)
    return(output.expect)


def CR_echo(tend, psi0):
    # Echoed CR gate
    if tend == 0:
        return qt.expect(e_op_list, psi0)

    tlist = np.arange(0, tend, dt)

    output1 = qt.mesolve(Hp, psi0, tlist[0:int(len(tlist)/2)], [], [])
    psiEcho = qt.tensor(qt.rx(np.pi), qt.identity(2)) * output1.states[-1]
    output2 = qt.mesolve(Hm, psiEcho, tlist[int(len(tlist)/2):], [], e_op_list)
    return np.array(output2.expect)[:, -1]


def CRt(tlist, psi0):
    # Simple CR gate in qubit 1 frame
    output = qt.mesolve(Htp, psi0, tlist, [], e_op_list, {})
    return(output.expect)


def CRt_echo(tend, psi0):
    # Echoed CR in qubit 1 frame
    if tend == 0:
        return qt.expect(e_op_list, psi0)

    tlist = np.arange(0, tend, dt)
    output1 = qt.mesolve(Htp, psi0, tlist[0:int(len(tlist)/2)], [], [], {})
    psiEcho = qt.tensor(qt.rx(np.pi), qt.identity(2)) * output1.states[-1]
    output2 = qt.mesolve(Htm, psiEcho, tlist[int(len(tlist)/2):], [], e_op_list, {})
    return np.array(output2.expect)[:, -1]
# %%
if __name__ == '__main__':
    # Simple CR, in qubit 2 frame
    print('Calculate simple CR gate')
    result_down = CR(tlist_simple, psidown)
    result_up = CR(tlist_simple, psiup)
    # Save result to file
    print('Save data')
    data = np.vstack((tlist_simple, result_down, result_up))
    qt.file_data_store('CR0.dat', data, numtype='real')
    print('Finished')
    # Echoed CR, in qubit 2 frame
    print('Calculate echoed CR gate')
    print('|psi0> = |00>')
    result_down = \
        np.array(qt.parallel_map(CR_echo, tlist_echo, task_args=(psidown,),
                                 progress_bar=True)).T
    print('|psi0> = |10>')
    result_up = \
        np.array(qt.parallel_map(CR_echo, tlist_echo, task_args=(psiup,),
                                 progress_bar=True)).T
    print('Finished')
    # Save result to file
    print('Save data')
    data = np.vstack((tlist_echo, result_down, result_up))
    qt.file_data_store('CR2.dat', data, numtype='real')
# %% Plot results
    data = qt.file_data_read('CR0.dat')
    tlist = data[0]
    result_down = data[1:7]
    result_up = data[7:13]

    plt.close('all')
    plt.figure(1)
    plt.subplot(211)
    plt.plot(tlist, np.real(result_down[4]), tlist, np.real(result_up[4]))
    plt.ylabel(r'$\langle\sigma_2^y\rangle$')
    plt.legend([r'$|\psi_0\rangle=|00\rangle$',
                r'$|\psi_0\rangle=|10\rangle$'])
    plt.title('Simple CR gate')
    plt.subplot(212)
    plt.plot(tlist, np.real(result_down[5]), tlist, np.real(result_up[5]))
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'$\langle\sigma_2^z\rangle$')
    plt.tight_layout()

    data = qt.file_data_read('CR2.dat')
    tlist = data[0]
    result_down = data[1:7]
    result_up = data[7:13]

    plt.figure(2)
    plt.subplot(211)
    plt.plot(tlist, np.real(result_down[4]), tlist, np.real(result_up[4]))
    plt.ylabel(r'$\langle\sigma_2^y\rangle$')
    plt.title('Echoed CR gate')
    plt.legend([r'$|\psi_0\rangle=|00\rangle$',
                r'$|\psi_0\rangle=|10\rangle$'])
    plt.subplot(212)
    plt.plot(tlist, np.real(result_down[5]), tlist, np.real(result_up[5]))
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'$\langle\sigma_2^z\rangle$')
    plt.tight_layout()

# %% Repeat calculation in qubit 1 frame
# Takes a lot more time because Hamiltonian is time dependent
#    print('Calculate simple CR gate dephasing')
#    result_simple = CRt(tlist_simple, psiplus)
#    print('Finished')
#    print('Save data')
#    data = np.vstack((tlist_simple, result_simple))
#    qt.file_data_store('CR0_PhaseShift.dat', data, numtype='real')
#    print('Calculate echoed CR gate dephasing')
#    result_echo = np.array(qt.parallel_map(CRt_echo, tlist_echo,
#                                           task_args=(psiplus,),
#                                           progress_bar=True)).T
#    print('Finished')
#    print('Save data')
#    data = np.vstack((tlist_echo, result_echo))
#    qt.file_data_store('CR2_PhaseShift.dat', data, numtype='real')
# %%
#    plt.figure(3)
#    plt.subplot(211)
#    data = qt.file_data_read('CR0_PhaseShift.dat')
#    tlist = data[0]
#    result_simple = data[1:7]
#    plt.plot(tlist, np.real(result_simple[0]))
#    plt.ylabel(r'$\langle\sigma_1^x\rangle$')
#    plt.title('Simple CR gate')

#    data = qt.file_data_read('CR2_PhaseShift.dat')
#    tlist = data[0]
#    result_echo = data[1:7]
#    plt.subplot(212)
#    plt.plot(tlist, np.real(result_echo[0]))
#    plt.ylim(-1, 1)
#    plt.xlabel(r'Time ($\mu$s)')
#    plt.ylabel(r'$\langle\sigma_1^x\rangle$')
#    plt.title('Echoed CR gate')
