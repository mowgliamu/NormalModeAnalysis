# @Author: Prateek Goel
# @Date: 03/05/2017
# @Email: prateik.goel@gmail.com
# @Last modified by: Prateek

from normal_mode_analysis.x_to_q import get_x_to_q_transformation_matrix
from normal_mode_analysis.normal_mode_analysis import driver_process_abinitio_data

# ====================================
# Input parameters for ab initio data
# ====================================

input_data_dir = 'input_data'
minima = 'r_small'
gaussian_version = 'g09'
data_minima_full_path = [input_data_dir + '/' + minima + '.log', input_data_dir + '/' + minima + '.fchk']

# Dimensionless normal coordinates
dimensionless = False

# MAIN

if __name__ == "__main__":
    all_data = driver_process_abinitio_data(gaussian_version, data_minima_full_path)

    # Use get_attr
    number_of_atoms = getattr(all_data, 'natom')
    atomic_masses = getattr(all_data, 'amass')
    nmode = getattr(all_data, 'nmode')
    eq_geom = getattr(all_data, 'eq_geom_cart')
    frequencies = getattr(all_data, 'frequencies')
    normal_modes = getattr(all_data, 'Lmwc')

    # ============================================
    # X->Q and Q->X Transformation (Write to file)
    # ============================================

    Q_to_X, X_to_Q = get_x_to_q_transformation_matrix(number_of_atoms, atomic_masses, nmode, eq_geom,
                                                  frequencies, normal_modes, dimensionless)

