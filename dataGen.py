'''FD solver for 2d Buergers equation'''
# spatial diff: 4th order laplacian
# temporal diff: O(dt^5) due to RK4

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def apply_laplacian(mat, dx=1.0):
    # dx is inversely proportional to N
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -5 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (4 / 3, (-1, 0)),
        (4 / 3, (0, -1)),
        (4 / 3, (0, 1)),
        (4 / 3, (1, 0)),
        (-1 / 12, (-2, 0)),
        (-1 / 12, (0, -2)),
        (-1 / 12, (0, 2)),
        (-1 / 12, (2, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx ** 2


def apply_dx(mat, dx=1.0):
    ''' central diff for dx'''

    # np.roll, axis=0 -> row
    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (2, 0)),
        (-8.0 / 12, (1, 0)),
        (8.0 / 12, (-1, 0)),
        (-1.0 / 12, (-2, 0))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dx


def apply_dy(mat, dy=1.0):
    ''' central diff for dx'''

    # the total difference
    neigh_mat = -0 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0 / 12, (0, 2)),
        (-8.0 / 12, (0, 1)),
        (8.0 / 12, (0, -1)),
        (-1.0 / 12, (0, -2))
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat / dy


def get_temporal_diff(U, V, R, dx):
    # u and v in (h, w)

    laplace_u = apply_laplacian(U, dx)
    laplace_v = apply_laplacian(V, dx)

    u_x = apply_dx(U, dx)
    v_x = apply_dx(V, dx)

    u_y = apply_dy(U, dx)
    v_y = apply_dy(V, dx)

    # governing equation
    u_t = (1.0 / R) * laplace_u - U * u_x - V * u_y
    v_t = (1.0 / R) * laplace_v - U * v_x - V * v_y

    return u_t, v_t


def update(U0, V0, R=100.0, dt=0.05, dx=1.0):
    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    U = U0 + dt * u_t
    V = V0 + dt * v_t
    return U, V


def update_rk4(U0, V0, R=100.0, dt=0.05, dx=1.0):
    """Update with Runge-kutta-4 method
       See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """
    ############# Stage 1 ##############
    # compute the diffusion part of the update

    u_t, v_t = get_temporal_diff(U0, V0, R, dx)

    K1_u = u_t
    K1_v = v_t

    ############# Stage 1 ##############
    U1 = U0 + K1_u * dt / 2.0
    V1 = V0 + K1_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U1, V1, R, dx)

    K2_u = u_t
    K2_v = v_t

    ############# Stage 2 ##############
    U2 = U0 + K2_u * dt / 2.0
    V2 = V0 + K2_v * dt / 2.0

    u_t, v_t = get_temporal_diff(U2, V2, R, dx)

    K3_u = u_t
    K3_v = v_t

    ############# Stage 3 ##############
    U3 = U0 + K3_u * dt
    V3 = V0 + K3_v * dt

    u_t, v_t = get_temporal_diff(U3, V3, R, dx)

    K4_u = u_t
    K4_v = v_t

    # Final solution
    U = U0 + dt * (K1_u + 2 * K2_u + 2 * K3_u + K4_u) / 6.0
    V = V0 + dt * (K1_v + 2 * K2_v + 2 * K3_v + K4_v) / 6.0

    return U, V


def postProcess(output, reso, xmin, xmax, ymin, ymax, num, fig_save_dir):
    ''' num: Number of time step
    '''

    x = np.linspace(0, reso, reso + 1)
    y = np.linspace(0, reso, reso + 1)
    x_star, y_star = np.meshgrid(x, y)
    x_star, y_star = x_star[:-1, :-1], y_star[:-1, :-1]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('u-FDM')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=0.95, edgecolors='none', cmap='RdYlBu',
                       marker='s', s=3, vmin=-1, vmax=1)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('black')
    # cf.cmap.set_over('whitesmoke')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('v-FDM')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    # plt.draw()
    plt.savefig(fig_save_dir + 'uv_%s.png' % str(num).zfill(4))
    plt.close('all')


def rand_gaussian_ic(Num_a, Num_b, Nx, Ny, seed,plot=True):
    assert Nx == Ny
    x, y = [np.linspace(0, 1, Nx + 1)] * 2
    xx, yy = np.meshgrid(x[:-1], y[:-1])
    # print("xx",xx)
    # print("yy",yy)
    Wx, Wy = [0] * 2
    np.random.seed(seed)#1
    # np.random.seed(2)#2
    # np.random.seed(3)#3
    # np.random.seed(4)#4
    Ax = np.random.normal(0, 1, size=(Num_a, Num_b))
    Bx = np.random.normal(0, 1, size=(Num_a, Num_b))
    Ay = np.random.normal(0, 1, size=(Num_a, Num_b))
    By = np.random.normal(0, 1, size=(Num_a, Num_b))
    cxy = np.random.normal(-1, 1, size=2)
    for i in range(Num_a):
        for j in range(Num_b):
            Wx = Wx + Ax[i, j] * np.sin(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy)) + Bx[
                i, j] * np.cos(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy))
            Wy = Wy + Ay[i, j] * np.sin(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy)) + By[
                i, j] * np.cos(2 * np.pi * ((i - Num_a // 2) * xx + (j - Num_b // 2) * yy))
    # print(cxy)
    # print(cxy[0])
    # print(cxy[1])
    Ux = 2 * Wx / Wx.max() + cxy[0]
    Uy = 2 * Wy / Wy.max() + cxy[1]

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        #
        ax[0].axis('square')
        cf = ax[0].contourf(xx, yy, Ux, levels=101, cmap='jet')
        ax[0].set_xlim([0, 1])
        ax[0].set_ylim([0, 1])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(r'$u_0$', )
        fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
        #
        ax[1].axis('square')
        cf = ax[1].contourf(xx, yy, Uy, levels=101, cmap='jet')
        ax[1].set_xlim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(r'$v_0$')
        fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)
        #
        plt.show()

    return Ux, Uy
def createdata(seed):
    M, N = 100, 100
    n_simu_steps = 2101
    # dt = 0.00025  # 0.00025 still not converge for FWE, 0.001, 0.002 works for RK4
    dt = 0.001
    dx = 1.0 / M
    # R = 120.0
    R = 200

    print(seed)
    U, V = rand_gaussian_ic(Num_a=10, Num_b=10, Nx=100, Ny=100,seed=seed, plot=False)  # 2101_1 and 2
    # U, V = rand_gaussian_ic(Num_a=20, Num_b=20, Nx=100, Ny=100, plot=False)#4

    U, V = U / 3.0, V / 3.0

    # U_record = U.copy()[None, ...]
    # V_record = V.copy()[None, ...]

    U_list = []
    V_list = []

    for step in range(n_simu_steps):

        # U, V = update(U, V, R, dt, dx)  # [h,w]
        U, V = update_rk4(U, V, R, dt, dx)  # [h,w]

        if (step + 1) % 1 == 0:
            print(step, '\n')
            U_list.append(U[None, ...])
            V_list.append(V[None, ...])

    U_record = np.concatenate(U_list, axis=0)  # [t,h,w]
    V_record = np.concatenate(V_list, axis=0)

    UV = np.concatenate((U_record[None, ...], V_record[None, ...]), axis=0)  # (c,t,h,w)
    UV = np.transpose(UV, [1, 0, 2, 3])  # (t,c,h,w) (751,2,128,128)
    #np.ascontiguousarray(UV, dtype=np.float64)

    # fig_save_dir = './figures/'
    # for i in range(0, 2000, 50):  # 1500
    #     postProcess(UV, M, 0, M, 0, M, i, fig_save_dir)

    # Output result if you want
    data_save_dir = '../data/'


    scipy.io.savemat(data_save_dir + 'Burgers_2101x2x104x104_[RK4,R=200,dt=0_001,#seed'+str(seed)+'].mat', {'uv': UV})

    print("seed"+str(seed)+"create over")

if __name__ == '__main__':
    # grid size
    for seed in range(1,2):
        createdata(seed)
    
