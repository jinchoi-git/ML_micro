import jax
import jax.numpy as np
import numpy as onp
import os
import meshio
from src.integrator import MultiVarSolver
from src.utils import Field
from src.yaml_parse import args
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def set_params(idx):
    '''
    If a certain parameter is not set, a default value will be used according to the YAML file.
    '''
    args['id'] = idx
    args['case'] = f'unet_input_{idx}'
    args['num_grains'] = 4000
    args['domain_x'] = 0.2
    args['domain_y'] = 0.2
    args['domain_z'] = 0.1
    args['r_beam'] = 0.03
    args['power'] = 100
    args['write_sol_interval'] = 400
    
    # args['ad_hoc'] = 0.1


def pre_processing(idx):
    '''
    We use Neper to generate polycrystal structure.
    Neper has two major functions: generate a polycrystal structure, and mesh it.
    See https://neper.info/ for more information.
    '''
    set_params(idx)
    os.system(f'''neper -T -n {args['num_grains']} -id {args['id']} -regularization 0 -domain "cube({args['domain_x']},\
               {args['domain_y']},{args['domain_z']})" \
                -o post-processing/neper/{args['case']}/domain -format tess,obj,ori''')
    os.system(f"neper -T -loadtess post-processing/neper/{args['case']}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
    os.system(f"neper -M -rcl 1 -elttype hex -faset faces post-processing/neper/{args['case']}/domain.tess")

    # Optional, write the Neper files to local for visualization
    polycrystal = Field()
    polycrystal.write_vtu_files()


def run(idx):
    set_params(idx)
    solver = MultiVarSolver()
    solver.solve()


def post_processing(idx):
    set_params(idx)
    polycrystal = Field()
    cell_ori_inds_3D = polycrystal.convert_to_3D_images()


# if __name__ == "__main__":
#     pre_processing()
#     run()
#     post_processing()

if __name__ == "__main__":
    for i in range(1750, 2000):
        # pre_processing(i)
        run(i)
        post_processing(i)
        plt.close('all')
