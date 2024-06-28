import numpy as np
class convertor:
    def __init__(
        self,
        tau,
        resolution , # physical length / convertor_length
        characteristic_length, # physical
        characteristic_velocity, # physical
        kinematic_viscosity, # physical
        diffusion_coefficient,
        air_density, # physical
        physical_parameters
        ):
        self.tau = tau
        self.resolution = resolution
        self.characteristic_length = characteristic_length
        self.characteristic_velocity = characteristic_velocity
        self.viscosity = kinematic_viscosity
        self.diffusion_coefficient = diffusion_coefficient
        self.air_density = air_density
        self.physical_parameters = np.array(physical_parameters, dtype=np.float32) # input [length, height, ST_velocity]

        self.convertor_length = self.characteristic_length / self.resolution
        self.convertor_time =  (self.tau - 0.5) / 3 * (self.convertor_length * self.convertor_length) / self.viscosity
        self.convertor_velocity = self.convertor_length / self.convertor_time
        self.convertor_viscosity = self.convertor_length * self.convertor_length / self.convertor_time
        self.convertor_gravity = self.convertor_length / (self.convertor_time * self.convertor_time)
        #self.convertor_mass = 1.21 * self.convertor_length * self.convertor_length * self.convertor_length
        #self.convertor_force = self.convertor_gravity * self.convertor_mass

        self.lattice_length = self.physical_parameters[0] / self.convertor_length
        self.lattice_height = self.physical_parameters[1] / self.convertor_length
        self.lattice_ventilation_width = 0.2 / self.convertor_length
        self.lattice_inlet_velocity = self.characteristic_velocity / self.convertor_velocity
        self.lattice_methane_velocity = self.physical_parameters[2] / self.convertor_velocity
        self.lattice_air_density = 1
        self.lattice_methane_density = 0.71 / self.lattice_air_density
        self.lattice_kinematic_viscosity = self.viscosity / self.convertor_viscosity
        self.lattice_diffusion_coefficient = self.diffusion_coefficient / self.convertor_viscosity
        self.lattice_gravity = 9.8 / self.convertor_gravity

    def print (self):
        print ("Tau:", self.tau)
        print ("Resolution:", self.resolution)
        print ("Characteristic Length:", self.characteristic_length)
        print ("Characteristic Velocity:", self.characteristic_velocity)
        print ("Viscosity:", self.viscosity)
        print ("Diffusion Coefficient:", self.diffusion_coefficient)
        print ("Air Density:", self.air_density)
        print ("_______________________________")
        print ("Convertor Length:", self.convertor_length)
        print ("Convertor Time:", self.convertor_time)
        print ("Convertor Velocity:", self.convertor_velocity)
        print ("Convertor Viscosity:", self.convertor_viscosity)
        print ("convertor_gravity:", self.convertor_gravity)
        print("_______________________________")
        print ("Lattice Length:", self.lattice_length)
        print ("Lattice Height:", self.lattice_height)
        print ("Lattice Ventilation Width:", self.lattice_ventilation_width)
        print ("Lattice Inlet Velocity:", self.lattice_inlet_velocity)
        print ("Lattice Methane Velocity:", self.lattice_methane_velocity)
        print ("Lattice Air Density:", self.lattice_air_density)
        print ("Lattice_methane_density:", self.lattice_methane_density)
        print ("Lattice_kinematic_viscosity:", self.lattice_kinematic_viscosity)
        print ("lattice_diffusion_coefficient:", self.lattice_diffusion_coefficient)
        print ("lattice_gravity:", self.lattice_gravity)
