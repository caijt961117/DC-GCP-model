# A LBM-DA coupling solver for simulating pollutant dispersion in real time
# Author : Jitao (jitao.cai@outlook.com)

import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu)

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        nx,  # domain size
        ny,
        niu,  # viscosity of fluid
        D, # diffusion coefficient
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        ST_value, # source term
        ):
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.d = 2 # dimension 2D
        self.q = 9

        # -----scalar-----
        self.D = D
        self.tau_scalar = 3.0 * D + 0.5
        self.inv_tau_scalar = 1.0 / self.tau_scalar
        self.g = ti.Vector.field(2, float, shape=(nx, ny)) # gravity
        self.F = ti.Vector.field(9, float, shape=(nx, ny))
        # ----------------

        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.Y = ti.field(float, shape=(nx, ny)) # mass fraction
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])

        # -----scalar-----
        self.f_old_scalar = ti.Vector.field(5, float, shape=(nx, ny))
        self.f_new_scalar = ti.Vector.field(5, float, shape=(nx, ny))
        self.w_scalar = ti.types.vector(5, float)(2, 1, 1, 1, 1) / 6.0
        self.e_scalar = ti.types.matrix(5, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1])
        # ----------------

        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.ST_value = ti.Vector.field(2, float, shape=4)
        self.ST_value.from_numpy(np.array(ST_value, dtype=np.float32))

        # turbulent parameters
        self.inv_tauEFF = ti.field(float, shape=(nx, ny))
        self.inv_tauEFF_scalar = ti.field(float, shape=(nx, ny))
        self.pi = ti.Vector.field(self.d * (self.d + 1) // 2, float, shape=(nx, ny)) # 对称性 只求3个
        self.piNeqNormSqr = ti.field(float, shape=(nx, ny))
        self.viscosity_t = ti.field(float, shape=(nx, ny))
    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func  # compute equilibrium distribution function for scalar
    def f_eq_scalar(self, i, j):
        eu = self.e_scalar @ self.vel[i, j]
        return self.w_scalar * self.Y[i, j] * (1 + 3 * eu)

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.Y.fill(0)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            self.f_old_scalar[i, j] = self.f_new_scalar[i, j] = self.f_eq_scalar(i, j)
            self.g[i, j] = [0, -0.001]
            # if no turbulence
            self.inv_tauEFF[i, j] = self.inv_tau
            self.inv_tauEFF_scalar[i, j] = self.inv_tau_scalar

    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0] # ip，jp for previous location，i，j for location after streaming (current)
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.inv_tauEFF[i, j]) * self.f_old[ip, jp][k] + feq[k] * self.inv_tauEFF[i, j] + self.F[ip, jp][k]

    @ti.kernel
    def collide_and_stream_scalar(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(5)):
                ip = i - self.e_scalar[k, 0]
                jp = j - self.e_scalar[k, 1]
                feq = self.f_eq_scalar(ip, jp)
                self.f_new_scalar[i, j][k] = (1 - self.inv_tauEFF_scalar[i, j]) * self.f_old_scalar[ip, jp][k] + feq[k] * self.inv_tauEFF_scalar[i, j]
    @ti.kernel
    def apply_buoyancy(self):  # compute buoyancy
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                # 访问 e 的第 k 行，得到一个二维向量
                e_k = tm.vec2([self.e[k, 0], self.e[k, 1]])
                # 计算 e_k 与 vel[i, j] 的逐元素乘积
                product = e_k * self.vel[i, j]
                # force = (1 - 0.5 * self.inv_tauEFF[i, j]) * self.w[k] * (3 * (e_k - self.vel[i, j]) + 9 * product * e_k) * (1 * (-1) * (self.Y[i, j] * 0.71 - 1))
                force = (1 - 0.5 * self.inv_tauEFF[i, j]) * self.w[k] * (
                            3 * (e_k - self.vel[i, j]) + 9 * product * e_k) * ((self.Y[i, j] * 0.71 + (1-self.Y[i, j]) * 1) - 1)* self.g[i, j]
                force_scalar =  tm.dot(force, self.g[i, j])
                self.F[i, j][k] = force_scalar

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            # self.vel[i, j] += 0.5 * (1 * (-1) * (self.Y[i, j] * 0.71 - 1)) * self.g[i, j]
            self.vel[i, j] += 0.5 * ((self.Y[i, j] * 0.71 + (1-self.Y[i, j]) * 1) - 1) * self.g[i, j]
            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def update_macro_var_scalar(self):  # compute Y
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.Y[i, j] = 0
            for k in ti.static(range(5)):
                self.f_old_scalar[i, j][k] = self.f_new_scalar[i, j][k]
                self.Y[i, j] += self.f_new_scalar[i, j][k]

    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

    @ti.kernel
    def apply_ST(self, left: int, right: int, bottom: int, top: int):  # impose source term
        for i in range(left, right + 1):
            # bottom (local): dr = 3; ibc = 90 to 100; jbc = 0; inb = 90 to 100; jnb = 1
            self.vel[i, 0] = self.ST_value[3]
        # source term for scalar
        for i, j in ti.ndrange((left, right + 1), (bottom, top + 1)):
            self.Y[i, j] = 1

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:  # handle outer boundary
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
                self.Y[ibc, jbc] = 0

            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]
                self.Y[ibc, jbc] = self.Y[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        # self.f_old_scalar[ibc, jbc] = self.f_eq_scalar(ibc, jbc) - self.f_eq_scalar(inb, jnb) + self.f_old_scalar[inb, jnb]

    @ti.kernel
    def update_bc(self, ST_new_value: ti.template()):
        for i in self.ST_value:
            self.ST_value[i] = ST_new_value[i]

    # turbulent model LES
    @ti.kernel
    def compute_stress(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # ipi = 0, ialpha = 0, ibeta = 0
            for ipop in ti.static(range(9)):
                self.pi[i, j][0] += self.e[ipop, 0] * self.e[ipop, 0] * self.f_old[i, j][ipop]
            self.pi[i, j][0] -= self.rho[i, j] * self.vel[i, j][0] * self.vel[i, j][0]
            self.pi[i, j][0] -= 1.0 / 3 * (self.rho[i, j] - 1)
            # ipi = 1, ialpha = 0, ibeta = 1
            for ipop in ti.static(range(9)):
                self.pi[i, j][1] += self.e[ipop, 0] * self.e[ipop, 1] * self.f_old[i, j][ipop]
            self.pi[i, j][1] -= self.rho[i, j] * self.vel[i, j][0] * self.vel[i, j][1]
            # ipi = 2, ialpha = 1, ibeta = 1
            for ipop in ti.static(range(9)):
                self.pi[i, j][2] += self.e[ipop, 1] * self.e[ipop, 1] * self.f_old[i, j][ipop]
            self.pi[i, j][2] -= self.rho[i, j] * self.vel[i, j][1] * self.vel[i, j][1]
            self.pi[i, j][2] -= 1.0 / 3 * (self.rho[i, j] - 1)


    @ti.kernel
    def compute_PiNeqNormSqr(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.piNeqNormSqr[i, j] = self.pi[i, j][0] * self.pi[i, j][0] + 2. * self.pi[i, j][1] * self.pi[i, j][1] + self.pi[i, j][2] * self.pi[i, j][2]

    @ti.kernel
    def compute_tauEff(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            piNeqNorm = ti.sqrt(self.piNeqNormSqr[i, j])
            preFactor =  0.16 * 0.16 * 3 * 3 * 2 * 1  # Cs * Cs * cs * cs (lattice velocity) * delta_x * delta_x //* sqrt(2) ?
            tauMol = 1 / self.tau
            self.viscosity_t[i, j] = 1 / 6 * (ti.sqrt(tauMol * tauMol + preFactor / self.rho[i, j] * piNeqNorm) - tauMol)
            tau_t = 3 * self.viscosity_t[i, j]
            self.inv_tauEFF[i, j] = 1 / (tauMol + tau_t)
            self.inv_tauEFF_scalar[i, j] = 1 / (self.tau_scalar + self.viscosity_t[i, j] / 0.1 * 3)
