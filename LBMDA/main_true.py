import lbmSolver as lbm
import convertor as con
import taichi as ti
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
matplotlib.use('TkAgg')

ti.init(arch=ti.gpu)

# physical parameter
length = 5 # m
height = 0.5
ventilation_width = 0.2
inlet_velocity = 0.02 # m/s
source_term_velocity = 0.05
# source_term_diameter = 0.05
air_density = 1.21
methane_density = 0.71
kinematic_viscosity = 1.5 * 10 ** (-5) # m2/s
diffusion_coefficient = 2.2 * 10 ** (-5)
# convertor
convertor = con.convertor(0.51,20,0.1, inlet_velocity, kinematic_viscosity,
                           diffusion_coefficient,air_density,[length,height,source_term_velocity])

# simulation parameters
nx = round(convertor.lattice_length) # meh number
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
Y_display.fill(0.0)
Y_display1 = Y_display.to_numpy()

solve = lbm.lbm_solver(nx, ny, niu, D, [0, 0, 1, 0],
                                [[convertor.lattice_inlet_velocity, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, convertor.lattice_methane_velocity]])
# time loop
single_observations = np.zeros((10, 1))
all_observations = np.zeros((10, 30))

# 创建图形和轴
fig, ax = plt.subplots()
# 创建图像对象，并设定显示的初始数据和颜色映射
im = ax.imshow(Y_display1.T, origin='lower',cmap='viridis', vmin=0, vmax=0.1)
# 添加 colorbar
cbar = fig.colorbar(im)
cbar.set_label('Value Scale')

for i in range(total_iteration):
    print("Iteration: ", i)
    if i == 0:
        solve.init()
        solve.apply_ST(ST_lb, ST_rb, ST_bb, ST_ub)
    solve.apply_buoyancy()
    # LES smagorinsky model
    # solve.pi.fill(0)
    # solve.compute_stress()
    # solve.compute_PiNeqNormSqr()
    # solve.compute_tauEff()
    solve.collide_and_stream()
    solve.update_macro_var()
    solve.collide_and_stream_scalar()
    solve.update_macro_var_scalar()
    solve.apply_bc()
    solve.apply_ST(ST_lb, ST_rb, ST_bb, ST_ub)
    Y_display1 = solve.Y.to_numpy()

    #plot
    if i % 200 == 0:
        im.set_data(Y_display1.T)

        # 更新图形，不需要重新创建图形和 colorbar
        ax.draw_artist(ax.patch)  # 绘制背景
        ax.draw_artist(im)  # 绘制图像

        # 清除旧的显示并显示当前帧
        clear_output(wait=True)  # 清除之前的输出
        display(fig)  # 显示当前图形

        plt.pause(0.05)  # 短暂暂停，以便观察更新

    if (i + 1) % 500 == 0:
        I = (i + 1) // 500
        # single_observations[0, 0] = Y_display1[200, 40]
        # single_observations[1, 0] = Y_display1[250, 40]
        # single_observations[2, 0] = Y_display1[300, 40]
        # single_observations[3, 0] = Y_display1[350, 40]
        # single_observations[4, 0] = Y_display1[400, 40]
        # single_observations[5, 0] = Y_display1[450, 40]
        # single_observations[6, 0] = Y_display1[500, 40]
        # single_observations[7, 0] = Y_display1[550, 40]
        # single_observations[8, 0] = Y_display1[600, 40]
        # single_observations[9, 0] = Y_display1[650, 40]
        # all_observations[:, I - 1] = single_observations[:, 0]
        # filename = f'result_no_truth_{I}.npy'
        # np.save(filename, Y_display1)
with open('matrix.txt', 'w') as f:
    np.savetxt('matrix.txt', all_observations, fmt='%.7f', delimiter=' ')


