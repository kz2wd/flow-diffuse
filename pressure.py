import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from pathlib import Path

from visualize_ret180 import read_binary_field, read_params, read_y
param = read_params(Path("param.txt"))
yp = read_y(Path("yp.dat"), param)

def get_velocity(t):
    ux = read_binary_field(Path(f"Ret180/vel/ux{t}"), param)
    uy = read_binary_field(Path(f"Ret180/vel/uy{t}"), param)
    uz = read_binary_field(Path(f"Ret180/vel/uz{t}"), param)
    return np.stack([ux, uy, uz], axis=-1)

u = np.stack([get_velocity("200"), get_velocity("300")])

def get_slow(t):
    return read_binary_field(Path(f"Ret180/pres/p_slow{t}"), param)
source_slow_data = np.stack([get_slow("200"), get_slow("300")])

def get_rapid(t):
    return read_binary_field(Path(f"Ret180/pres/p_rapid{t}"), param)
source_rapid_data = np.stack([get_rapid("200"), get_rapid("300")])

def get_stokes(t):
    return read_binary_field(Path(f"Ret180/pres/p_stokes{t}"), param)
stokes_data = np.stack([get_stokes("200"), get_stokes("300")])

combined_pressure_data = source_slow_data + source_rapid_data + stokes_data

def L1(X, Y):
    return np.mean(np.abs(X - Y))
def L2(X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))
def rel(X, Y):
    return L2(X, Y) / np.sqrt(np.mean(Y**2))

delta_yp = np.gradient(yp, param.Ly / (param.Ny - 1))
y_stretching = 1.0 / delta_yp

T, X, Y, Z, C = u.shape

def broadcast_to(source, target, index):
    d = source.shape
    target_shape = list(target.shape)
    for i in range(len(target_shape)):
        if i != index:
            target_shape[i] = 1
    return source.reshape(tuple(target_shape))


def derive_fft(u, direction, h):
    n = u.shape[direction]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    k_broad = broadcast_to(k, u, direction)
    u_hat = np.fft.fftn(u, axes=(direction,))
    u_hat_prime = 1.0j * k_broad * u_hat
    return np.fft.ifftn(u_hat_prime, axes=(direction,)).real


def derive_compact(u, direction, h):
    rhs = np.zeros_like(u)
    rhs = np.roll(u, -1, axis=direction) - np.roll(u, 1, axis=direction)
    rhs *= 3.0 / h
    length = u.shape[direction]
    ab = np.zeros((3, length))
    ab[0, 2:] = 1
    ab[1, :] = 4
    ab[2, :-2] = 1
    # BC
    ab[0, 1] = 0
    ab[2, -2] = 0
    rhs_reordered = np.moveaxis(rhs, direction, 0)
    reordered_shape = rhs_reordered.shape
    rhs_reshaped = rhs_reordered.reshape(length, -1)
    u_prime_reshaped = solve_banded((1, 1), ab, rhs_reshaped)
    u_prime = np.moveaxis(u_prime_reshaped.reshape(reordered_shape), 0, direction)
    return u_prime

def grad(field):
    grad = np.zeros((T, X, Y, Z, 3, C))
    grad[:, :, :, :, 0, :] = derive_fft(field, 1, param.Lx / param.Nx)
    grad[:, :, :, :, 1, :] = derive_compact(field, 2, param.Ly / (param.Ny - 1)) * y_stretching[None, None, :, None, None]
    grad[:, :, :, :, 2, :] = derive_fft(field, 3, param.Lz / param.Nz)
    return grad



def compute_pressure(rf):

    rf_hat = np.fft.fftn(rf, axes=(1, 3))
    freqx = 2.0 * np.pi * np.fft.fftfreq(param.Nx, param.Lx /param.Nx)
    freqz = 2.0 * np.pi * np.fft.fftfreq(param.Nz, param.Lz /param.Nz)


    p_prime_hat = np.zeros_like(rf_hat)

    for t in range(T):
        for x in range(param.Nx):
            for z in range(param.Nz):
                kx = freqx[x]
                kz = freqz[z]
                k2 = kx * kx + kz * kz

                A = np.zeros((3, param.Ny), dtype=np.complex128)
                A[0, 2:] = 1 - (k2 * 1.0 / 10.0)
                A[1, :] = -2 - (k2 * 1.0)
                A[2, :-2] = 1 - (k2 * 1.0 / 10.0)
                A[0, 1] = 2 - (k2 * 2.0 / 10.0)
                A[2, -2] = 2 - (k2 * 2.0 / 10.0)

                rf_slice = rf_hat[t, x, :, z]
                rhs = np.zeros_like(rf_slice)
                rhs[1:-1] = (1.0 / 10.0) * rf_slice[:-2] + 1.0 * rf_slice[1:-1] + (1.0 / 10.0) * rf_slice[2:]
                rhs[0] = (2.0 / 10.0) * rf_slice[1] + 1.0 * rf_slice[0]
                rhs[-1] = 1.0 * rf_slice[-1] + (2.0 / 10.0) * rf_slice[-2]

                # Pin mode 0,0
                if x == 0 and z == 0:
                    # Skip mode 0,0 as it represents average and we dont want it
                    continue


                p_prime_slice = solve_banded((1, 1), A, rhs)
                p_prime_hat[t, x, :, z] = p_prime_slice

    return np.fft.ifftn(p_prime_hat, axes=(1, 3)).real

# Average over time, x, z
u_mean = u.mean(axis=(0, 1, 3), keepdims=True)
u_fluc = u - u_mean

u_mean_grad = grad(u_mean)
u_fluc_grad = grad(u_fluc)


slow_u = -np.einsum("...ij, ...ji -> ...", u_fluc_grad, u_fluc_grad)
rapid_u = -2 * np.einsum("...ij, ...ji -> ...", u_mean_grad, u_fluc_grad)

p_prime = compute_pressure(slow_u + rapid_u)
