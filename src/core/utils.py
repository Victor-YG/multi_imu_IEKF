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


def compute_process_model_jocobian(omega, accel):
    A = np.zeros([15, 15])
    omega_vee = get_lifted_form(omega)
    accel_vee = get_lifted_form(accel)

    A[0:3, 0:3] = -omega_vee
    A[3:6, 3:6] = -omega_vee
    A[6:9, 6:9] = -omega_vee
    A[3:6, 0:3] = -accel_vee
    A[6:9, 3:6] = np.eye(3)

    A[0:6, 9:15] = -np.eye(6)
    return A


def compute_measurement_model_jocobian(T, fx, fy, cx, cy, x, y, z, u, v):
    e_y = np.zeros(2)
    p_i = np.array([x, y, z, 1.0])
    p_c = np.matmul(T, p_i) # landmarks in camera frame

    y_c  = np.array([u, v])
    y_op = np.zeros(2)
    y_op[0] = fx *  p_c[0] / p_c[2] + cx # u
    y_op[1] = fy *  p_c[1] / p_c[2] + cy # v
    e_y = y_c - y_op

    Z_jk = get_circle_dot_for_SE2_3(p_c)

    S_jk = np.zeros([2, 3])
    S_jk[0, 0] =  fx /  p_c[2]
    S_jk[0, 2] = -fx *  p_c[0] / p_c[2]**2
    S_jk[1, 1] =  fy /  p_c[2]
    S_jk[1, 2] = -fy *  p_c[1] / p_c[2]**2
    # print(f"S_jk = {S_jk}")

    D_T = np.zeros([3, 4])
    D_T[0: 3, 0: 3] = np.eye(3)
    G = -S_jk @ D_T @ Z_jk

    return e_y, G


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
        axes[d].plot(t, e[:, d])
        axes[d].plot(t,  3 * stddev[:, d], color="orange", linewidth=1, linestyle='dashed')
        axes[d].plot(t, -3 * stddev[:, d], color="orange", linewidth=1, linestyle='dashed')
        axes[d].fill_between(t, 3 * stddev[:, d], -3 * stddev[:, d], color='green', alpha=0.1, label='Filled Area')
        axes[d].set_title(titles[d])

    plt.show()