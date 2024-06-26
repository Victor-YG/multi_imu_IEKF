import numpy as np
import matplotlib.pyplot as plt


def get_lifted_form(da):
    '''convert change in angle to rotation matrix'''

    C = np.zeros([3, 3])
    da = da.flatten()
    C[0, 1] = -da[2]
    C[0, 2] =  da[1]
    C[1, 0] =  da[2]
    C[1, 2] = -da[0]
    C[2, 0] = -da[1]
    C[2, 1] =  da[0]
    return C


def get_circle_dot_for_SE_3(p):
    M = np.zeros([4, 6])
    p = p.flatten()
    M[0:3, 0:3] = get_lifted_form(p) * (-1)
    M[0:3, 3: ] = np.eye(3) * p[3]
    return M


def get_circle_dot_for_SE2_3(p):
    M = np.zeros([4, 9])
    p = p.flatten()
    M[0:3, 0:3] = get_lifted_form(p) * (-1)
    M[0:3, 6: ] = np.eye(3) * p[3]
    return M


def Cr2T(C_vi, r_i_vi):
    '''given C_vi and r_i_vi return T_vi'''

    T_vi = np.eye(4)
    r_v_iv = -np.matmul(C_vi, r_i_vi.flatten())
    T_vi[0:3, 0:3] = C_vi.copy()
    T_vi[0:3,   3] = r_v_iv.copy()
    return T_vi


def T2Cr(T_vi):
    '''given T_vi return C_vi and r_i_vi'''

    r_v_iv = T_vi[0:3, 3].copy()
    C_vi   = T_vi[0:3, 0:3].copy()
    C_iv   = np.linalg.inv(C_vi)
    r_i_vi = -np.matmul(C_iv, r_v_iv)

    return C_vi, r_i_vi


def plot_trajectory(trajectory_tru, trajectory_est):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory_tru[:, 0], trajectory_tru[:, 1], trajectory_tru[:, 2], c='g', label="trajectory_tru")
    ax.plot(trajectory_est[:, 0], trajectory_est[:, 1], trajectory_est[:, 2], c='r', label="Trajectory_est")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def plot_error_distribution(X, P):
    '''plot estimated pose error against ground truth'''

    K = X.shape[0]
    X_var = np.zeros([K, 9])

    for k in range(K):
        var = np.diag(P[0: 9, 0: 9, k])
        X_var[k, :] = var

    fig, axes = plt.subplots(9, 1)

    # axes[0].set_ylim([-0.5, 0.5])
    # axes[1].set_ylim([-0.5, 0.5])
    # axes[2].set_ylim([-0.5, 0.5])
    # axes[3].set_ylim([-0.5, 0.5])
    # axes[4].set_ylim([-0.5, 0.5])
    # axes[5].set_ylim([-0.5, 0.5])
    # axes[6].set_ylim([-0.5, 0.5])
    # axes[7].set_ylim([-0.5, 0.5])
    # axes[8].set_ylim([-0.5, 0.5])

    t = np.linspace(start=0, stop=K, num=K, dtype=int)

    axes[0].plot(t, X[:, 0])
    axes[0].plot(t,  3 * np.sqrt(X_var)[:, 0], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[0].plot(t, -3 * np.sqrt(X_var)[:, 0], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[0].fill_between(t, 3 * np.sqrt(X_var)[:, 0], -3 * np.sqrt(X_var)[:, 0],
                         color='green', alpha=0.1, label='Filled Area')

    axes[1].plot(t, X[:, 1])
    axes[1].plot(t,  3 * np.sqrt(X_var)[:, 1], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[1].plot(t, -3 * np.sqrt(X_var)[:, 1], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[1].fill_between(t, 3 * np.sqrt(X_var)[:, 1], -3 * np.sqrt(X_var)[:, 1],
                         color='green', alpha=0.1, label='Filled Area')

    axes[2].plot(t, X[:, 2])
    axes[2].plot(t,  3 * np.sqrt(X_var)[:, 2], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[2].plot(t, -3 * np.sqrt(X_var)[:, 2], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[2].fill_between(t, 3 * np.sqrt(X_var)[:, 2], -3 * np.sqrt(X_var)[:, 2],
                         color='green', alpha=0.1, label='Filled Area')

    axes[3].plot(t, X[:, 3])
    axes[3].plot(t,  3 * np.sqrt(X_var)[:, 3], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[3].plot(t, -3 * np.sqrt(X_var)[:, 3], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[3].fill_between(t, 3 * np.sqrt(X_var)[:, 3], -3 * np.sqrt(X_var)[:, 3],
                         color='green', alpha=0.1, label='Filled Area')

    axes[4].plot(t, X[:, 4])
    axes[4].plot(t,  3 * np.sqrt(X_var)[:, 4], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[4].plot(t, -3 * np.sqrt(X_var)[:, 4], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[4].fill_between(t, 3 * np.sqrt(X_var)[:, 4], -3 * np.sqrt(X_var)[:, 4],
                         color='green', alpha=0.1, label='Filled Area')

    axes[5].plot(t, X[:, 5])
    axes[5].plot(t,  3 * np.sqrt(X_var)[:, 5], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[5].plot(t, -3 * np.sqrt(X_var)[:, 5], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[5].fill_between(t, 3 * np.sqrt(X_var)[:, 5], -3 * np.sqrt(X_var)[:, 5],
                         color='green', alpha=0.1, label='Filled Area')

    axes[6].plot(t, X[:, 6])
    axes[6].plot(t,  3 * np.sqrt(X_var)[:, 6], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[6].plot(t, -3 * np.sqrt(X_var)[:, 6], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[6].fill_between(t, 3 * np.sqrt(X_var)[:, 6], -3 * np.sqrt(X_var)[:, 6],
                         color='green', alpha=0.1, label='Filled Area')

    axes[7].plot(t, X[:, 7])
    axes[7].plot(t,  3 * np.sqrt(X_var)[:, 7], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[7].plot(t, -3 * np.sqrt(X_var)[:, 7], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[7].fill_between(t, 3 * np.sqrt(X_var)[:, 7], -3 * np.sqrt(X_var)[:, 7],
                         color='green', alpha=0.1, label='Filled Area')

    axes[8].plot(t, X[:, 8])
    axes[8].plot(t,  3 * np.sqrt(X_var)[:, 8], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[8].plot(t, -3 * np.sqrt(X_var)[:, 8], color="orange",
                 linewidth=1, linestyle='dashed')
    axes[8].fill_between(t, 3 * np.sqrt(X_var)[:, 8], -3 * np.sqrt(X_var)[:, 8],
                         color='green', alpha=0.1, label='Filled Area')

    plt.show()