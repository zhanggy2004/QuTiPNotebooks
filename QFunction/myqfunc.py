# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
#
# Modified Q function calculation
# Gengyan Zhang, Dec 4 2015

from scipy import (zeros, arange, exp, real, pi,
                   sqrt, meshgrid, size, polyval, fliplr, conjugate)
import numpy as np
import scipy.linalg as la

from qutip.qobj import Qobj, isket, isoper
from scipy.misc import factorial

# -----------------------------------------------------------------------------
# Q FUNCTION
#
def qfunc(state, xvec, yvec, g=sqrt(2)):
    """Q-function of a given state vector or density matrix
    at points `xvec + i * yvec`.

    Parameters
    ----------
    state : qobj
        A state vector or density matrix.

    xvec : array_like
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like
        y-coordinates at which to calculate the Wigner function.

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.

    Returns
    --------
    Q : array
        Values representing the Q-function calculated over the specified range
        [xvec,yvec].

    """
    X, Y = meshgrid(xvec, yvec)
    avec = np.reshape(0.5 * g * (X + Y * 1j),[len(xvec)*len(yvec),1])

    if not (isoper(state) or isket(state)):
        raise TypeError('Invalid state operand to qfunc.')

    qvec = zeros(size(avec))

    if isket(state):
        qvec = _qfunc_pure(state, avec)
    elif isoper(state):
        d, v = la.eig(state.full())
        # d[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        qvec = zeros(np.shape(avec))
        for k in arange(0, len(d)):
            qvec1 = _qfunc_pure(v[:, k], avec)
            qvec += real(d[k] * qvec1)

    qvec = 0.25 * qvec * g ** 2
    return np.reshape(qvec,X.shape)


#
# Q-function for a pure state: Q = |<alpha|psi>|^2 / pi
#
# |psi>   = the state in fock basis
# |alpha> = the coherent state with amplitude alpha
#
def _qfunc_pure(psi, alpha_vec):
    """
    Calculate the Q-function for a pure state.
    """
    n = np.prod(psi.shape)

# Gengyan: maximun number to use factorial()    
    nmax = 170

    if isinstance(psi, Qobj):
        psi = psi.full().flatten()
    else:
        psi = psi.T

    if n < nmax:
        qvec = abs(polyval(fliplr([psi / sqrt(factorial(arange(n)))])[0],
                           conjugate(alpha_vec)))**2*exp(-abs(alpha_vec)**2)
    else:
# Gengyan: for m < nmax, use factorial()
        qvec = polyval(fliplr([psi[0:nmax] / sqrt(factorial(arange(nmax)))])[0], 
                           conjugate(alpha_vec))*exp(-abs(alpha_vec)**2/2)
# Gengyan: for m >= nmax, use Stirling's approximation
        for m in range(nmax,n):
            qvec += (conjugate(alpha_vec)/sqrt(m))**m*psi[m] * \
                exp((m-abs(alpha_vec)**2)/2)*(2*pi*m)**(-0.25)
        qvec = abs(qvec)**2
            
    return np.real(qvec) / pi
