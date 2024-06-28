import lbmSolver as lbm
import convertor as con
import taichi as ti
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
matplotlib.use('TkAgg')

ti.init(arch=ti.gpu)

def regularized_pinv(A, lambda_reg=1e-8):
    identity_matrix = np.eye(A.shape[1]) 
    regularized_A = A.T @ A + lambda_reg * identity_matrix  
    inverse_regularized_A = np.linalg.inv(regularized_A)  
    pinv_A = inverse_regularized_A @ A.T  
    return pinv_A

# physical parameter
length = 5 # m
height = 0.5
ventilation_width = 0.2
inlet_velocity = 0.015 # m/s
source_term_velocity = 0.1
# source_term_diameter = 0.05
air_density = 1.21
methane_density = 0.71
kinematic_viscosity = 1.5 * 10 ** (-5) # m2/s
diffusion_coefficient = 2.2 * 10 ** (-5)
# convertor
convertor = con.convertor(0.51,20,0.1, inlet_velocity, kinematic_viscosity,
                           diffusion_coefficient,air_density,[length,height,source_term_velocity])

# simulation parameters
nx = round(convertor.lattice_length) # mesh number
ny = round(convertor.lattice_height)
mesh_number = nx * ny
niu = convertor.lattice_kinematic_viscosity
D = convertor.lattice_diffusion_coefficient
total_iteration = 15000 # total_iteration = da_interval * da_iteration
ST_lb = 200 # source term left boundary
ST_rb = 210
ST_bb = 1
ST_ub = 5
Y_display = ti.field(ti.i32, (nx, ny)) # for final visualization

# DA parameters
np.random.seed(0)
setDA = True
#setDA = False
ensemble_size = 20
da_interval = 500
da_iterations = 30 # da number
I = 0
inflation = 1.05
observation_size = 10
state_matrix_forcast = ti.field(float, shape=(mesh_number, ensemble_size))
ensemble_mean_forcast = ti.field(float, shape=(mesh_number, 1))
source_term = np.zeros((da_iterations + 1, ensemble_size))
record_source_term = np.zeros((da_iterations + 1, ensemble_size))
objects = np.empty(ensemble_size, dtype=object) # lbm ensembles
Yf = np.zeros((observation_size, ensemble_size))
#mean = convertor.lattice_methane_velocity # average velocity
mean = 0.05
std = 0.002

probe = np.zeros((10, 30))

# kernel
def error_initialize():
    for i in range(ensemble_size):
        source_term[0, i] = mean + std * (np.random.randn()-0.5)

field_1d = ti.field(float, shape=(mesh_number))
@ti.kernel
def reshape_2d_to_1d(object: ti.template(), field_1d: ti.template()):
    for i, j in object.Y:
        idx = i * object.ny + j
        field_1d[idx] = object.Y[i, j]

@ti.kernel
def fill_column_from_vector(mat: ti.template(), vec: ti.template(), col_index: ti.i32):
    for i in vec:
        mat[i, col_index] = vec[i]

# initialization
error_initialize()
with open('source_term_prior.txt', 'w') as f:
    np.savetxt('source_term_prior.txt', source_term[0, :], fmt='%.7f', delimiter=' ')

for i in range(ensemble_size):
    objects[i] = lbm.lbm_solver(nx, ny, niu, D, [0, 0, 1, 0],
                                [[convertor.lattice_inlet_velocity, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, source_term[I, i]]])
# time loop
for i in range(total_iteration):
    # print('time loop: ', i)
    for j in range(ensemble_size):
        #print('ensemble: ', j)
        # LBM solver
        if i == 0:
            objects[j].init()
            objects[j].apply_ST(ST_lb, ST_rb, ST_bb, ST_ub)
        objects[j].apply_buoyancy()
        # LES smagorinsky model
        objects[j].pi.fill(0)
        objects[j].compute_stress()
        objects[j].compute_PiNeqNormSqr()
        objects[j].compute_tauEff()
        objects[j].collide_and_stream()
        objects[j].update_macro_var()
        objects[j].collide_and_stream_scalar()
        objects[j].update_macro_var_scalar()
        objects[j].apply_bc()
        objects[j].apply_ST(ST_lb, ST_rb, ST_bb, ST_ub)
        reshape_2d_to_1d(objects[j], field_1d)
        fill_column_from_vector(state_matrix_forcast, field_1d, j)
    ensemble_mean_forcast = np.mean(state_matrix_forcast.to_numpy(), axis=1)
    Y_display = ensemble_mean_forcast.reshape((nx, ny))
    # plot
    # if i % 200 == 0:
    #    clear_output(wait=True)
    #    plt.imshow(Y_display.T, origin='lower', cmap='viridis')
    #    display(plt.gcf())
    #    plt.pause(0.05)

    if(setDA):
        if (i+1) % da_interval == 0:
            # Y inflation
            state_matrix_concentration = state_matrix_forcast.to_numpy()
            repeat_mean_concentration = np.tile(ensemble_mean_forcast[:, np.newaxis], (1, ensemble_size))
            state_matrix_concentration = repeat_mean_concentration + inflation * (state_matrix_concentration - repeat_mean_concentration)
            # source_term inflation
            source_term_mean = np.mean(source_term, axis=1)
            repeat_mean_ST = np.tile(source_term_mean[:, np.newaxis], (1, ensemble_size))
            source_term = repeat_mean_ST + inflation * (source_term - repeat_mean_ST)
            # DA calculation
            Xf = np.vstack((state_matrix_concentration, source_term))
            Xf_mean = np.mean(Xf, axis=1)
            repeat_mean_Xf = np.tile(Xf_mean[:, np.newaxis], (1, ensemble_size))
            Xf_deviation = Xf - repeat_mean_Xf
            # observation operator
            for i in range(ensemble_size):
                Yf[0, i] = state_matrix_concentration[200 * ny + 40, i]
                Yf[1, i] = state_matrix_concentration[250 * ny + 40, i]
                Yf[2, i] = state_matrix_concentration[300 * ny + 40, i]
                Yf[3, i] = state_matrix_concentration[350 * ny + 40, i]
                Yf[4, i] = state_matrix_concentration[400 * ny + 40, i]
                Yf[5, i] = state_matrix_concentration[450 * ny + 40, i]
                Yf[6, i] = state_matrix_concentration[500 * ny + 40, i]
                Yf[7, i] = state_matrix_concentration[550 * ny + 40, i]
                Yf[8, i] = state_matrix_concentration[600 * ny + 40, i]
                Yf[9, i] = state_matrix_concentration[650 * ny + 40, i]
            Yf_mean = np.mean(Yf, axis=1)
            repeat_mean_Yf = np.tile(Yf_mean[:, np.newaxis], (1, ensemble_size))
            Yf_deviation = Yf - repeat_mean_Yf
            # observation
            observations = np.loadtxt('matrix.txt', delimiter=' ')
            y0 = observations[:,I]
            repeat_mean_y0 = np.tile(y0[:, np.newaxis], (1, ensemble_size))
            ov = np.sum(y0)/observation_size * 0.002
            y0_error = np.sqrt(ov) * np.random.rand(observation_size,ensemble_size)
            y = repeat_mean_y0 + y0_error
            # DA update
            YfYfT_plus_error = Yf_deviation @ Yf_deviation.T + y0_error @ y0_error.T
            Xa = Xf + Xf_deviation @ Yf_deviation.T @ regularized_pinv(YfYfT_plus_error) @ (y - Yf)
            source_term = Xa[mesh_number:,:]
            Xa = Xa[:mesh_number,:]

            # when elements less than zero
            for n_temp in range(ensemble_size):
                idd = np.where(source_term[:, n_temp] < 0)[0]
                rate_temp_negative = -1 * np.sum(source_term[idd, n_temp])
                # source_term[idd, n_temp] = np.finfo(float).eps
                source_term[idd, n_temp] = 0.00000000000000001
                idd = np.where(Xa[:, n_temp] < 0)[0]
                negative_tmp = (-1 * np.sum(Xa[idd, n_temp]) * 1 * 1 / 1 /     # 1 * 1 / 1: dx * dy / dt, dt_interval，physical units?
                                (500) - rate_temp_negative)
                source_term[:, n_temp] += (source_term[:, n_temp] /
                                           np.sum(source_term[:, n_temp])) * negative_tmp
                Xa[idd, n_temp] = 0

            # source term update
            source_term_mean = np.mean(source_term[I, :])
            #rate_std = max(np.std(source_term[I, :]), 0.1 * source_term_mean)
            rate_std = 0
            for n_temp in range(ensemble_size):
                rate_std += (source_term[I, n_temp] - source_term_mean) * (source_term[I, n_temp] - source_term_mean)
            rate_std = np.sqrt(rate_std/ensemble_size)
            rate_std = max(rate_std, 0.0)
            alpha = 0.8
            source_term[I + 1,:] =  (source_term_mean + alpha * (source_term[I, :] - source_term_mean) +
                                     np.sqrt(1 - alpha ** 2) * np.random.randn(1, ensemble_size) * rate_std)
            for n_temp in range(ensemble_size):
                objects[n_temp].Y = Xa[:,n_temp].reshape((nx, ny))
                ST_values = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, source_term[I + 1, n_temp]]], dtype=np.float32)
                taichi_array = ti.Vector.field(2, float, shape=4)
                taichi_array.from_numpy(ST_values)
                objects[n_temp].update_bc(taichi_array)
                objects[n_temp].apply_bc()
                objects[n_temp].apply_ST(ST_lb, ST_rb, ST_bb, ST_ub)
            print('da analysis：', I)
            I = I + 1

            # for final visualization
            ensemble_mean_forcast = np.mean(Xa, axis=1)
            Y_display = ensemble_mean_forcast.reshape((nx, ny))
            # plt.imshow(Y_display.T, origin='lower')
            # plt.colorbar()
            # plt.show()

            # filename = f'result_{I}.npy'
            # np.save(filename, Y_display)
            probe[0, I-1] = Y_display[200, 40]
            probe[1, I-1] = Y_display[250, 40]
            probe[2, I-1] = Y_display[300, 40]
            probe[3, I-1] = Y_display[350, 40]
            probe[4, I-1] = Y_display[400, 40]
            probe[5, I-1] = Y_display[450, 40]
            probe[6, I-1] = Y_display[500, 40]
            probe[7, I-1] = Y_display[550, 40]
            probe[8, I-1] = Y_display[600, 40]
            probe[9, I-1] = Y_display[650, 40]

with open('source_term.txt', 'w') as f:
    np.savetxt('source_term.txt', source_term, fmt='%.7f', delimiter=' ')
with open('probe.txt', 'w') as f:
    np.savetxt('probe.txt', probe, fmt='%.7f', delimiter=' ')
