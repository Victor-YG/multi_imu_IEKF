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


def Cr2T(C_vi, r_vi_i):
    '''given C_vi and r_vi_i return T_vi'''

    T_vi = np.eye(4)
    r_v_iv = -np.matmul(C_vi, r_vi_i.flatten())
    T_vi[0:3, 0:3] = C_vi.copy()
    T_vi[0:3,   3] = r_v_iv.copy()
    return T_vi


def T2Cr(T_vi):
    '''given T_vi return C_vi and r_vi_i'''

    r_v_iv = T_vi[0:3, 3].copy()
    C_vi   = T_vi[0:3, 0:3].copy()
    C_iv   = np.linalg.inv(C_vi)
    r_vi_i = -np.matmul(C_iv, r_v_iv)

    return C_vi, r_vi_i


def pose_to_rotation_and_translation(T_iv):
    '''given T_iv return C_iv and r_vi_i'''

    C_iv   = T_iv[0:3, 0:3].copy()
    r_vi_i = T_iv[0:3,   3].copy()
    return C_iv, r_vi_i


def rotation_and_translation_to_pose(C_iv, r_vi_i):
    '''given C_iv and r_vi_i return T_iv'''

    T_iv = np.eye(4)
    T_iv[0:3, 0:3] = C_iv.copy()
    T_iv[0:3,   3] = r_vi_i.copy()
    return T_iv


def plot_trajectory_and_initial_velocity(trajectory, v0):
    x0 = trajectory[0, 0]
    y0 = trajectory[0, 1]
    z0 = trajectory[0, 2]
    x1 = x0 + 1 * v0[0]
    y1 = y0 + 1 * v0[1]
    z1 = z0 + 1 * v0[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='g', label="trajectory")
    ax.plot([x0, x1], [y0, y1], [z0, z1])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def plot_trajectory(trajectory_tru, trajectory_est):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trajectory_tru[:, 0], trajectory_tru[:, 1], trajectory_tru[:, 2], c='g', label="trajectory_tru")
    ax.plot(trajectory_est[:, 0], trajectory_est[:, 1], trajectory_est[:, 2], c='r', label="Trajectory_est")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def plot_state_error_and_uncertainty(e, P, titles):
    '''plot error and standard deviation'''

    K = e.shape[0]
    D = e.shape[1]
    stddev = np.zeros([K, D])

    for k in range(K):
        var = np.diag(P[:, :, k])
        stddev[k, :] = np.sqrt(var)

    fig, axes = plt.subplots(D, 1)
    t = np.linspace(start=0, stop=K, num=K, dtype=int)

    for d in range(D):
        axes[d].plot(t,  3 * stddev[:, d], color="orange", linewidth=1, linestyle='dashed')
        axes[d].plot(t, -3 * stddev[:, d], color="orange", linewidth=1, linestyle='dashed')
        axes[d].fill_between(t, 3 * stddev[:, d], -3 * stddev[:, d], color='green', alpha=0.1, label='Filled Area')
        axes[d].plot(t, e[:, d])
        axes[d].set_title(titles[d])

    plt.show()