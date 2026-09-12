"""
Flexible rotor modal analysis + Campbell diagram
Translated from:
    flexible_rotor_modal_DTU_c(3).m

Original model:
    Machinery Dynamics Lectures (41514), DTU
    Ilmar Ferreira Santos, 2021

The formulation preserves the original 4-DOF/node model:
    [v, w, theta_y, theta_z]

The Campbell diagram is obtained by sweeping the rotational speed
and solving the gyroscopic eigenvalue problem at each speed.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


# ============================================================
# MODEL DEFINITION
# ============================================================

NE = 12                  # number of shaft elements
NN = NE + 1              # number of nodes
DOF_PER_NODE = 4
GL = NN * DOF_PER_NODE   # total physical DOFs

ND = 2
NM = 2

# MATLAB node numbering -> Python zero-based node numbering
CD1 = 4 - 1              # disc 1
CD2 = 10 - 1             # disc 2
CMM1 = 1 - 1             # bearing 1
CMM2 = 13 - 1            # bearing 2


# ============================================================
# CONSTANTS
# ============================================================

E = 2.0e11               # Young's modulus [N/m^2]
RHO_STEEL = 7800.0       # steel density [kg/m^3]
RHO_AL = 2770.0          # aluminium density [kg/m^3]


# ============================================================
# GEOMETRY
# ============================================================

# ---- Discs -------------------------------------------------

Rd = 6.0 / 100.0
Ri = (5.0 / 2.0) / 1000.0
espD = 1.1 / 100.0

MasD = (
    np.pi * Rd**2 * espD * RHO_AL
    - np.pi * Ri**2 * espD * RHO_AL
)

# Transverse mass moment of inertia
Id = (
    (0.25 * Rd**2 + espD**2 / 12.0)
    * np.pi * Rd**2 * espD * RHO_AL
    - (0.25 * Ri**2 + espD**2 / 12.0)
    * np.pi * Ri**2 * espD * RHO_AL
)

# Polar mass moment of inertia
Ip = (
    0.5 * Rd**2 * (np.pi * Rd**2 * espD * RHO_AL)
    - 0.5 * Ri**2 * (np.pi * Ri**2 * espD * RHO_AL)
)


# ---- Bearings ----------------------------------------------

MasM = 0.40698

h = 1.0 / 1000.0
b = 28.5 / 1000.0
Area = b * h
I_bearing = b * h**3 / 12.0
lr = 7.5 / 100.0

Kty0 = 2.0 * 12.0 * E * I_bearing / lr**3
Ktz0 = 2.0 * E * Area / lr

# Bearing damping
Dty1 = Dtz1 = Dry1 = Drz1 = 0.0
Dty2 = Dtz2 = Dry2 = Drz2 = 0.0

# Bearing stiffness
Kty1 = Kty0
Ktz1 = Ktz0
Kry1 = Krz1 = 0.0

Kty2 = Kty0
Ktz2 = Ktz0
Kry2 = Krz2 = 0.0


# ---- Shaft -------------------------------------------------

l = np.array([
    0.140 / 3.0,
    0.140 / 3.0,
    0.140 / 3.0,
    0.205 / 6.0,
    0.205 / 6.0,
    0.205 / 6.0,
    0.205 / 6.0,
    0.205 / 6.0,
    0.205 / 6.0,
    0.090 / 3.0,
    0.090 / 3.0,
    0.090 / 3.0,
])

Rext = (5.0 / 2.0) / 1000.0
Rint = 0.0

rx = np.full(NE, Rext)
ri = np.full(NE, Rint)
rho = np.full(NE, RHO_STEEL)

St = np.pi * (rx**2 - ri**2)
II = np.pi * (rx**4 - ri**4) / 4.0


# ============================================================
# LOCAL ELEMENT MATRICES
# ============================================================

def shaft_mass_matrix(length, rho_, area, r_outer, r_inner):
    """8x8 consistent mass matrix of one shaft element."""
    MteAux = np.array([
        [156, 0, 0, 22*length, 54, 0, 0, -13*length],
        [0, 156, -22*length, 0, 0, 54, 13*length, 0],
        [0, -22*length, 4*length**2, 0, 0, -13*length, -3*length**2, 0],
        [22*length, 0, 0, 4*length**2, 13*length, 0, 0, -3*length**2],
        [54, 0, 0, 13*length, 156, 0, 0, -22*length],
        [0, 54, -13*length, 0, 0, 156, 22*length, 0],
        [0, 13*length, -3*length**2, 0, 0, 22*length, 4*length**2, 0],
        [-13*length, 0, 0, -3*length**2, -22*length, 0, 0, 4*length**2],
    ], dtype=float)

    Mte = (rho_ * area * length / 420.0) * MteAux

    MreAux = np.array([
        [36, 0, 0, 3*length, -36, 0, 0, 3*length],
        [0, 36, -3*length, 0, 0, -36, -3*length, 0],
        [0, -3*length, 4*length**2, 0, 0, 3*length, -length**2, 0],
        [3*length, 0, 0, 4*length**2, -3*length, 0, 0, -length**2],
        [-36, 0, 0, -3*length, 36, 0, 0, -3*length],
        [0, -36, 3*length, 0, 0, 36, 3*length, 0],
        [0, -3*length, -length**2, 0, 0, 3*length, 4*length**2, 0],
        [3*length, 0, 0, -length**2, -3*length, 0, 0, 4*length**2],
    ], dtype=float)

    Mre = (
        rho_ * area * (r_outer**2 - r_inner**2)
        / (120.0 * length)
    ) * MreAux

    return Mte + Mre


def shaft_gyro_matrix(length, rho_, area, r_outer, r_inner):
    """8x8 gyroscopic matrix of one shaft element."""
    GeAux = np.array([
        [0, -36, 3*length, 0, 0, 36, 3*length, 0],
        [36, 0, 0, 3*length, -36, 0, 0, 3*length],
        [-3*length, 0, 0, -4*length**2, 3*length, 0, 0, length**2],
        [0, -3*length, 4*length**2, 0, 0, 3*length, -length**2, 0],
        [0, 36, -3*length, 0, 0, -36, -3*length, 0],
        [-36, 0, 0, -3*length, 36, 0, 0, -3*length],
        [-3*length, 0, 0, length**2, 3*length, 0, 0, -4*length**2],
        [0, -3*length, -length**2, 0, 0, 3*length, 4*length**2, 0],
    ], dtype=float)

    Ge = (
        2.0
        * (rho_ * area * (r_outer**2 + r_inner**2)
           / (120.0 * length))
        * GeAux
    )

    return Ge


def shaft_bending_stiffness_matrix(length, E_, I_area):
    """8x8 bending stiffness matrix of one shaft element."""
    KbeAux = np.array([
        [12, 0, 0, 6*length, -12, 0, 0, 6*length],
        [0, 12, -6*length, 0, 0, -12, -6*length, 0],
        [0, -6*length, 4*length**2, 0, 0, 6*length, 2*length**2, 0],
        [6*length, 0, 0, 4*length**2, -6*length, 0, 0, 2*length**2],
        [-12, 0, 0, -6*length, 12, 0, 0, -6*length],
        [0, -12, 6*length, 0, 0, 12, 6*length, 0],
        [0, -6*length, 2*length**2, 0, 0, 6*length, 4*length**2, 0],
        [6*length, 0, 0, 2*length**2, -6*length, 0, 0, 4*length**2],
    ], dtype=float)

    return (E_ * I_area / length**3) * KbeAux


# ============================================================
# GLOBAL MATRICES
# ============================================================

def node_dofs(node):
    """Return the four global DOF indices for a zero-based node."""
    start = node * DOF_PER_NODE
    return np.arange(start, start + DOF_PER_NODE)


def assemble_model():
    """
    Assemble the global M, G and K matrices.

    This corresponds to the MATLAB sections:
        GLOBAL MASS MATRIX
        GLOBAL GYROSCOPIC MATRIX
        GLOBAL STIFFNESS MATRIX
    """
    M = np.zeros((GL, GL))
    G = np.zeros((GL, GL))
    K = np.zeros((GL, GL))

    for n in range(NE):
        # Element connects nodes n and n+1.
        dofs = np.r_[node_dofs(n), node_dofs(n + 1)]

        M_e = shaft_mass_matrix(
            l[n], rho[n], St[n], rx[n], ri[n]
        )
        G_e = shaft_gyro_matrix(
            l[n], rho[n], St[n], rx[n], ri[n]
        )
        K_e = shaft_bending_stiffness_matrix(
            l[n], E, II[n]
        )

        M[np.ix_(dofs, dofs)] += M_e
        G[np.ix_(dofs, dofs)] += G_e
        K[np.ix_(dofs, dofs)] += K_e

    # ---- Disc 1 --------------------------------------------
    d = node_dofs(CD1)
    M[d[0], d[0]] += MasD
    M[d[1], d[1]] += MasD
    M[d[2], d[2]] += Id
    M[d[3], d[3]] += Id

    G[d[2], d[3]] -= Ip
    G[d[3], d[2]] += Ip

    # ---- Disc 2 --------------------------------------------
    d = node_dofs(CD2)
    M[d[0], d[0]] += MasD
    M[d[1], d[1]] += MasD
    M[d[2], d[2]] += Id
    M[d[3], d[3]] += Id

    G[d[2], d[3]] -= Ip
    G[d[3], d[2]] += Ip

    # ---- Bearings ------------------------------------------
    d = node_dofs(CMM1)
    M[d[0], d[0]] += MasM
    M[d[1], d[1]] += MasM
    K[d[0], d[0]] += Ktz1
    K[d[1], d[1]] += Kty1

    d = node_dofs(CMM2)
    M[d[0], d[0]] += MasM
    M[d[1], d[1]] += MasM
    K[d[0], d[0]] += Ktz2
    K[d[1], d[1]] += Kty2

    return M, G, K


# ============================================================
# MODAL ANALYSIS
# ============================================================

def modal_analysis(Omega, M, G, K):
    """
    Solve the same first-order gyroscopic eigenvalue problem
    used in the MATLAB code.

    MATLAB:
        Mglob = [M  0]
                [0  K]

        Kglob = [-Omega*G  K]
                [-K         0]

        [U,lambda] = eig(-Kglob,Mglob)

    Here scipy.linalg.eig is used for the generalized
    eigenvalue problem.
    """
    Z = np.zeros_like(M)

    Mglob = np.block([
        [M, Z],
        [Z, K],
    ])

    Kglob = np.block([
        [-Omega * G, K],
        [-K,         Z],
    ])

    eigenvalues, eigenvectors = la.eig(-Kglob, Mglob)

    # Sort primarily by imaginary part, matching the spirit
    # of the original MATLAB sorting operation.
    order = np.argsort(np.imag(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def natural_frequencies(Omega, M, G, K, positive_only=True):
    """
    Return natural frequencies in Hz.

    For an undamped gyroscopic system the eigenvalues occur
    approximately as +/- i*omega. Both forward and backward
    branches are therefore retained when positive_only=False.
    """
    lam, U = modal_analysis(Omega, M, G, K)

    wn = np.abs(np.imag(lam)) / (2.0 * np.pi)

    if positive_only:
        # Keep one copy of each +/- pair.
        mask = np.imag(lam) > 1e-8
        wn = wn[mask]
        U = U[:, mask]
        lam = lam[mask]

    order = np.argsort(wn)

    return wn[order], U[:, order], lam[order]


# ============================================================
# CAMPBELL DIAGRAM
# ============================================================

def calculate_campbell(
    M,
    G,
    K,
    speed_hz=None,
    max_speed_hz=160.0,
    n_speed=161,
    n_modes=12,
):
    """
    Calculate the Campbell diagram including gyroscopic effects.

    Parameters
    ----------
    M, G, K : ndarray
        Global mass, gyroscopic and stiffness matrices.
    speed_hz : array-like, optional
        Rotational speed points [Hz].
        If None, points are generated from 0 to max_speed_hz.
    max_speed_hz : float
        Maximum rotational speed [Hz].
    n_speed : int
        Number of speed points.
    n_modes : int
        Number of positive-frequency branches to retain.

    Returns
    -------
    speed_hz : ndarray
        Rotor rotational speed [Hz].
    frequencies_hz : ndarray
        Natural frequencies [Hz], shape (n_speed, n_modes).

    Notes
    -----
    The gyroscopic matrix is explicitly included through

        Kglob = [-Omega*G, K]
                [-K,       0]

    with

        Omega = 2*pi*speed_hz.

    Therefore this is a true gyroscopic Campbell calculation,
    rather than a speed-independent modal calculation.
    """
    if speed_hz is None:
        speed_hz = np.linspace(0.0, max_speed_hz, n_speed)
    else:
        speed_hz = np.asarray(speed_hz, dtype=float)

    frequencies_hz = np.full(
        (len(speed_hz), n_modes),
        np.nan,
        dtype=float,
    )

    for i, speed in enumerate(speed_hz):
        Omega = 2.0 * np.pi * speed

        wn, _, _ = natural_frequencies(
            Omega, M, G, K, positive_only=True
        )

        n = min(n_modes, len(wn))
        frequencies_hz[i, :n] = wn[:n]

    return speed_hz, frequencies_hz


def plot_campbell(
    speed_hz,
    frequencies_hz,
    max_speed_hz=160.0,
    max_frequency_hz=160.0,
    synchronous=True,
    n_modes=None,
):
    """
    Plot the Campbell diagram.

    x-axis: rotor speed [Hz]
    y-axis: natural frequency [Hz]

    The 1X synchronous line is plotted as f = speed.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    if n_modes is None:
        n_modes = frequencies_hz.shape[1]

    for mode in range(n_modes):
        ax.plot(
            speed_hz,
            frequencies_hz[:, mode],
            linewidth=1.5,
        )

    if synchronous:
        ax.plot(
            speed_hz,
            speed_hz,
            "k--",
            linewidth=1.2,
            label="1X synchronous",
        )

    ax.set_xlim(0.0, max_speed_hz)
    ax.set_ylim(0.0, max_frequency_hz)

    ax.set_xlabel("Rotational speed [Hz]")
    ax.set_ylabel("Natural frequency [Hz]")
    ax.set_title("Campbell Diagram")
    ax.grid(True, alpha=0.3)

    if synchronous:
        ax.legend()

    fig.tight_layout()
    return fig, ax


# ============================================================
# MODE SHAPE PLOT
# ============================================================

def plot_mode_shape(
    mode_number,
    Omega,
    M,
    G,
    K,
    n_points=99,
):
    """
    Reproduce the 3D mode-shape visualization from MATLAB.

    mode_number is one-based, as in the original MATLAB input.
    """
    wn, U, lam = natural_frequencies(
        Omega, M, G, K, positive_only=True
    )

    if mode_number < 1 or mode_number > len(wn):
        raise ValueError(
            f"mode_number must be between 1 and {len(wn)}."
        )

    # Positive-frequency eigenvector.
    # U contains [q; qdot-like auxiliary variables].
    mode = U[:GL, mode_number - 1]

    v = np.real(mode[0::4])
    w = np.real(mode[1::4])

    # Use the complex eigenvector phase, following the
    # MATLAB reconstruction:
    #
    # v(t) = Re(U)*cos(wn*t) + Im(U)*sin(wn*t)
    #
    # Here wn is angular frequency.
    lam_mode = lam[mode_number - 1]
    omega_n = abs(np.imag(lam_mode))

    if omega_n <= 0:
        raise ValueError("Selected mode does not have a positive imaginary eigenvalue.")

    ttotal = 8.0 / omega_n
    t = np.linspace(0.0, ttotal, n_points)

    vr = np.real(mode[0::4])
    vi = np.imag(mode[0::4])
    wr = np.real(mode[1::4])
    wi = np.imag(mode[1::4])

    v_t = np.outer(vr, np.cos(omega_n * t)) + np.outer(vi, np.sin(omega_n * t))
    w_t = np.outer(wr, np.cos(omega_n * t)) + np.outer(wi, np.sin(omega_n * t))

    # Normalize only for visualization.
    scale = np.max(np.abs(np.r_[v_t.ravel(), w_t.ravel()]))
    if scale > 0:
        v_t /= scale
        w_t /= scale

    x = np.linspace(0.0, np.sum(l), NN)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(NN):
        ax.plot(
            np.full_like(t, x[i]),
            w_t[i, :],
            v_t[i, :],
            linewidth=2.0,
        )

    ax.set_xlabel("Axial position [m]")
    ax.set_ylabel("w")
    ax.set_zlabel("v")

    speed_hz = Omega / (2.0 * np.pi)
    ax.set_title(
        f"Speed: {speed_hz:.2f} Hz | "
        f"Mode: {mode_number} | "
        f"Natural frequency: {wn[mode_number - 1]:.3f} Hz"
    )

    ax.view_init(elev=20, azim=-25)
    plt.tight_layout()

    return fig, ax


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Assembling global matrices...")
    M, G, K = assemble_model()

    print(f"Number of nodes: {NN}")
    print(f"Number of physical DOFs: {GL}")
    print(f"Disc mass: {MasD:.6g} kg")
    print(f"Disc transverse inertia: {Id:.6g} kg.m^2")
    print(f"Disc polar inertia: {Ip:.6g} kg.m^2")
    print()

    # --------------------------------------------------------
    # Modal analysis at a selected speed
    # --------------------------------------------------------

    speed_hz = 0.0
    Omega = 2.0 * np.pi * speed_hz

    wn, U, lam = natural_frequencies(
        Omega, M, G, K, positive_only=True
    )

    print(f"Natural frequencies at {speed_hz:.2f} Hz:")
    for i, freq in enumerate(wn[:12], start=1):
        print(f"  Mode {i:2d}: {freq:.6f} Hz")

    # --------------------------------------------------------
    # Campbell diagram
    # --------------------------------------------------------

    speed_hz, frequencies_hz = calculate_campbell(
        M,
        G,
        K,
        max_speed_hz=160.0,
        n_speed=161,
        n_modes=12,
    )

    plot_campbell(
        speed_hz,
        frequencies_hz,
        max_speed_hz=160.0,
        max_frequency_hz=160.0,
        synchronous=True,
    )

    plt.show()

    # To plot a mode shape, for example:
    #
    # plot_mode_shape(
    #     mode_number=1,
    #     Omega=2*np.pi*0.0,
    #     M=M,
    #     G=G,
    #     K=K,
    # )
    # plt.show()
