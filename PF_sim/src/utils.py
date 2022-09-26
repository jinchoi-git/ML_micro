import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import matplotlib.pyplot as plt
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.spatial.transform import Rotation as R
from src.yaml_parse import args


class Field:
    """TODO(Tianju): this Field class is in a mess
    """
    def __init__(self):
        #initialize directories for PF
        os.makedirs(f"post-processing/numpy/{args['case']}/pf/inds", exist_ok=True)
        os.makedirs(f"post-processing/numpy/{args['case']}/pf/info", exist_ok=True)
        os.makedirs(f"post-processing/numpy/{args['case']}/pf/sols", exist_ok=True)
        os.makedirs(f"post-processing/vtk/{args['case']}/mesh", exist_ok=True)
        os.makedirs(f"post-processing/vtk/{args['case']}/pf/sols", exist_ok=True)

        filepath = f"post-processing/neper/{args['case']}/domain.msh"
        mesh = meshio.read(filepath)
        points = mesh.points
        cells =  mesh.cells_dict['hexahedron']
        cell_grain_inds = mesh.cell_data['gmsh:physical'][0] - 1
        onp.save(f"post-processing/numpy/{args['case']}/pf/info/cell_grain_inds.npy", cell_grain_inds)
        assert args['num_grains'] == onp.max(cell_grain_inds) + 1, \
        f"specified number of grains = {args['num_grains']}, actual Neper = {onp.max(cell_grain_inds) + 1}"

        unique_oris_rgb, unique_grain_directions = get_unique_ori_colors()
        grain_oris_inds = onp.random.randint(args['num_oris'], size=args['num_grains'])
        cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

        # TODO: Not robust
        Nx = round(args['domain_x'] / points[1, 0])
        Ny = round(args['domain_y'] / points[Nx + 1, 1])
        Nz = round(args['domain_z'] / points[(Nx + 1)*(Ny + 1), 2])
        args['Nx'] = Nx
        args['Ny'] = Ny
        args['Nz'] = Nz

        print(f"Total num of grains = {args['num_grains']}")
        print(f"Total num of orientations = {args['num_oris']}")
        print(f"Total num of finite difference cells = {len(cells)}")
        assert Nx*Ny*Nz == len(cells)

        edges = []
        for i in range(Nx):
            if i % 100 == 0:
                print(f"i = {i}")
            for j in range(Ny):
                for k in range(Nz):
                    crt_ind = i + j * Nx + k * Nx * Ny
                    if i != Nx - 1:
                        edges.append([crt_ind, (i + 1) + j * Nx + k * Nx * Ny])
                    if j != Ny - 1:
                        edges.append([crt_ind, i + (j + 1) * Nx + k * Nx * Ny])
                    if k != Nz - 1:
                        edges.append([crt_ind, i + j * Nx + (k + 1) * Nx * Ny])

        edges = onp.array(edges)
        cell_points = onp.take(points, cells, axis=0)
        centroids = onp.mean(cell_points, axis=1)
        domain_vol = args['domain_x']*args['domain_y']*args['domain_z']
        volumes = domain_vol / (Nx*Ny*Nz) * onp.ones(len(cells))
        ch_len = (domain_vol / len(cells))**(1./3.) * onp.ones(len(edges))

        face_inds = [[0, 3, 4, 7], [1, 2, 5, 6], [0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
        boundary_face_centroids = onp.transpose(onp.stack([onp.mean(onp.take(cell_points, face_ind, axis=1), axis=1) 
            for face_ind in face_inds]), axes=(1, 0, 2))
        
        boundary_face_areas = []
        domain_measures = [args['domain_x'], args['domain_y'], args['domain_z']]
        face_cell_nums = [Ny*Nz, Nx*Nz, Nx*Ny]
        for i, domain_measure in enumerate(domain_measures):
            cell_area = domain_vol/domain_measure/face_cell_nums[i]
            boundary_face_area1 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i, i], 0., atol=1e-08), cell_area, 0.)
            boundary_face_area2 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i + 1, i], domain_measure, atol=1e-08), cell_area, 0.)
            boundary_face_areas += [boundary_face_area1, boundary_face_area2]

        boundary_face_areas = onp.transpose(onp.stack(boundary_face_areas))

        # TODO: unique_oris_rgb and unique_grain_directions should be a class property, not an instance property

        self.mesh = mesh
        
        self.edges = edges
        self.ch_len = ch_len
        self.centroids = centroids
        self.volumes = volumes
        self.unique_oris_rgb = unique_oris_rgb
        self.unique_grain_directions = unique_grain_directions
        self.cell_ori_inds = cell_ori_inds 
        self.boundary_face_areas = boundary_face_areas
        self.boundary_face_centroids = boundary_face_centroids
        

    def write_info(self):
        '''
        Mostly for post-processing. E.g., compute grain volume, aspect ratios, etc.
        '''
        onp.save(f"post-processing/numpy/{args['case']}/pf/info/edges.npy", self.edges)
        onp.save(f"post-processing/numpy/{args['case']}/pf/info/vols.npy", self.volumes)
        onp.save(f"post-processing/numpy/{args['case']}/pf/info/centroids.npy", self.centroids)


    def write_vtu_files(self):
        '''
        This is just a helper function if you want to visualize the polycrystal or the mesh generated by Neper.
        You may use Paraview to open the output vtu files.
        '''
        filepath = f"post-processing/neper/{args['case']}/domain.msh"
        fd_mesh = meshio.read(filepath)
        fd_mesh.write(f"post-processing/vtk/{args['case']}/mesh/fd_mesh.vtu")
        poly_mesh = self.obj_to_vtu()
        poly_mesh.write(f"post-processing/vtk/{args['case']}/mesh/poly_mesh.vtu")


    def obj_to_vtu(self):
        '''
        Convert the Neper .obj file to .vtu file.
        '''
        filepath=f"post-processing/neper/{args['case']}/domain.obj"
        file = open(filepath, 'r')
        lines = file.readlines()
        points = []
        cells_inds = []

        for i, line in enumerate(lines):
            l = line.split()
            if l[0] == 'v':
                points.append([float(l[1]), float(l[2]), float(l[3])])
            if l[0] == 'g':
                cells_inds.append([])
            if l[0] == 'f':
                cells_inds[-1].append([int(pt_ind) - 1 for pt_ind in l[1:]])

        cells = [('polyhedron', cells_inds)]
        mesh = meshio.Mesh(points, cells)
        return mesh


    def convert_to_3D_images(self):
        step = 0
        for step in range(11):
            filepath = f"post-processing/vtk/{args['case']}/pf/sols/u{step:03d}.vtu"     
            mesh_w_data = meshio.read(filepath)
            cell_ori_inds = mesh_w_data.cell_data['ori_inds'][0] 
    
            # By default, numpy uses order='C'
            cell_ori_inds_3D = np.reshape(cell_ori_inds, (args['Nz'], args['Ny'], args['Nx']))
    
            # This should also work
            # cell_ori_inds_3D = np.reshape(cell_ori_inds, (args['Nx'], args['Ny'], args['Nz']), order='F')
    
            print(cell_ori_inds_3D.shape)
            onp.save(f"post-processing/numpy/{args['case']}/pf/inds/cell_ori_inds_3d_{step:03d}.npy", cell_ori_inds)
        return cell_ori_inds_3D
       

def get_unique_ori_colors():
    '''
    Get colors.
    '''
    onp.random.seed(1)

    ori2 = Orientation.random(args['num_oris'])        

    vx = Vector3d((1, 0, 0))
    vy = Vector3d((0, 1, 0))
    vz = Vector3d((0, 0, 1))
    ipfkey_x = plot.IPFColorKeyTSL(symmetry.Oh, vx)
    rgb_x = ipfkey_x.orientation2color(ori2)
    ipfkey_y = plot.IPFColorKeyTSL(symmetry.Oh, vy)
    rgb_y = ipfkey_y.orientation2color(ori2)
    ipfkey_z = plot.IPFColorKeyTSL(symmetry.Oh, vz)
    rgb_z = ipfkey_z.orientation2color(ori2)
    rgb = onp.stack((rgb_x, rgb_y, rgb_z))

    onp.save(f"post-processing/numpy/quat_{args['case']}.npy", ori2.data)
    dx = onp.array([1., 0., 0.])
    dy = onp.array([0., 1., 0.])
    dz = onp.array([0., 0., 1.])
    scipy_quat = onp.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
    r = R.from_quat(scipy_quat)
    grain_directions = onp.stack((r.apply(dx), r.apply(dy), r.apply(dz)))

    save_ipf = True
    if save_ipf:
        # Plot IPF for those orientations
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 20,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        ori2.symmetry = symmetry.Oh
        ori2.scatter("ipf", c=rgb_x, direction=ipfkey_x.direction)
        # plt.savefig(f'post-processing/pdf/ipf_x.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_y, direction=ipfkey_y.direction)
        # plt.savefig(f'post-processing/pdf/ipf_y.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_z, direction=ipfkey_z.direction)
        # plt.savefig(f'post-processing/pdf/ipf_z.pdf', bbox_inches='tight')

    return rgb, grain_directions


def ipf_logo():
    new_params = {
        "figure.facecolor": "w",
        "figure.figsize": (6, 3),
        "lines.markersize": 10,
        "font.size": 25,
        "axes.grid": True,
    }
    plt.rcParams.update(new_params)
    plot.IPFColorKeyTSL(symmetry.Oh).plot()
    plt.savefig(f'data/pdf/ipf_legend.pdf', bbox_inches='tight')


def walltime(func):
    def wrapper(*list_args, **keyword_args):
        start_time = time.time()
        return_values = func(*list_args, **keyword_args)
        end_time = time.time()
        time_elapsed = end_time - start_time
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"Time elapsed {time_elapsed} on platform {platform}") 
        with open(f"post-processing/txt/walltime_{platform}_{args['case']}.txt", 'w') as f:
            f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
        return return_values
    return wrapper


def read_path():
    traveled_time = args['laser_path']['time']
    x_corners = args['laser_path']['x_pos']
    y_corners = args['laser_path']['y_pos']
    power_control = args['laser_path']['switch'][:-1]

    ts, xs, ys, ps = [], [], [], []
    for i in range(len(traveled_time) - 1):
        ts_seg = onp.arange(traveled_time[i], traveled_time[i + 1], args['dt'])
        xs_seg = onp.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = onp.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        ps_seg = onp.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        ps.append(ps_seg)

    ts, xs, ys, ps = onp.hstack(ts), onp.hstack(xs), onp.hstack(ys), onp.hstack(ps)  
    print(f"Total number of time steps = {len(ts)}")
    return ts, xs, ys, ps
