# This example implements macroscopic homogenized model of Biot-Darcy-Brinkman model of flow in deformable
# double porous media.
# The mathematical model is described in:
#
#ROHAN E., TURJANICOVÁ J., LUKEŠ V.
#The Biot-Darcy-Brinkman model of flow in deformable double porous media; homogenization and numerical modelling.
# Computers and Mathematics with applications, 78(9):3044-3066, 2019,
# https://doi.org/10.1016/j.camwa.2019.04.004
#
# Run simulation:
#
#   ./simple.py example_perfusion_BDB/perf_BDB_mac.py
#
# The results are stored in `example_perfusion_BDB/results/macro` directory.
#

import numpy as nm
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.homogenization.utils import define_box_regions
from sfepy.discrete.fem.mesh import Mesh
import os.path as osp


material_cache = {}
data_dir = 'example_perfusion_BDB'


def coefs2qp(coefs, nqp):
    out = {}
    for k, v in coefs.iteritems():
        if type(v) not in [nm.ndarray, float]:
            continue

        if type(v) is nm.ndarray:
            if len(v.shape) >= 3:
                out[k] = v

        out[k] = nm.tile(v, (nqp, 1, 1))

    return out

# Get raw homogenized coefficients, recalculate them if necessary
def get_raw_coefs(problem):
    if 'raw_coefs' not in material_cache:
        micro_filename = material_cache['meso_filename']
        coefs_filename = 'coefs_meso'
        coefs_filename = osp.join(problem.conf.options.get('output_dir', '.'),
                                  coefs_filename) + '.h5'
        coefs = get_homog_coefs_linear(0, 0, None,
                                       micro_filename=micro_filename, coefs_filename=coefs_filename)
        coefs['B'] = coefs['B'][:, nm.newaxis]
        material_cache['raw_coefs'] = coefs
    return material_cache['raw_coefs']

#Get homogenized coefficients in quadrature points
def get_homog(coors,pb, mode,  **kwargs):
    if not (mode == 'qp'):
        return
    nqp = coors.shape[0]

    coefs=get_raw_coefs(pb)
    for k in coefs.keys():
        v = coefs[k]
        if type(v) is nm.ndarray:
            if len(v.shape) == 0:
                coefs[k] = v.reshape((1, 1))
            elif len(v.shape) == 1:
                coefs[k] = v[:, nm.newaxis]
        elif isinstance(v, float):
            coefs[k] = nm.array([[v]])

    out = coefs2qp(coefs, nqp)

    return out

#Definition of dirichlet boundary conditions
def get_ebc( coors, amplitude,  cg1, cg2,const=False):
    """
    Define the essential boundary conditions as a function of coordinates
    `coors` of region nodes.
    """
    y = coors[:, 1] - cg1
    z = coors[:, 2] - cg2

    val = amplitude*((cg1**2 - (abs(y)**2))+(cg2**2 - (abs(z)**2)))
    if const:
        val=nm.ones_like(y) *amplitude

    return val

#Returns value of \phi_c\bar{w}^{mes} as a material function
def get_ebc_mat(  coors,pb, mode, amplitude,  cg1, cg2,konst=False):
    if mode == 'qp':
        val = get_ebc(  coors, amplitude,  cg1, cg2,konst)
        phic = get_raw_coefs(pb)['vol']["fraction_Zc"]
        v_w1 = val[:, nm.newaxis, nm.newaxis]
        return {'val': v_w1*phic}

#Definition of boundary conditions for numerical example at http://sfepy.org/sfepy_examples/example_perfusion_BDB/
def define_bc(cg1,cg2, val_in=1e2, val_out=1e2):

    funs = {
        'w_in': (lambda  ts, coor, bc, problem, **kwargs:
                 get_ebc( coor, val_in, cg1, cg2),),
        'w_out': (lambda  ts, coor, bc, problem, **kwargs:
                  get_ebc(  coor, val_out,  cg1, cg2),),
        'w_in_mat': (lambda  ts,coor, problem, mode=None, **kwargs:
                     get_ebc_mat( coor, problem, mode, val_in,
                                  cg1, cg2),),
        'w_out_mat': (lambda  ts,coor, problem, mode=None, **kwargs:
                      get_ebc_mat(  coor, problem, mode, val_out,
                                    cg1, cg2),),
    }
    mats = {
        'w_in': 'w_in_mat',
        'w_out': 'w_out_mat',
    }

    ebcs = {
        'fix_u_in': ('In', {'u.all': 0.0}),
        'fix_u_out': ('Out', {'u.all': 0.0}),
        'w_in': ('In', {'w.0': 'w_in','w.[1,2]': 0.0}),
        'w_out': ('Out', {'w.0': 'w_out','w.[1,2]': 0.0}),
        'wB_dirichlet':('Bottom',{'w.2' :0.0,'u.2':0.0}),
        'WT_dirichlet':('Top',{'w.2' :0.0,'u.2':0.0}),
        'wN_dirichlet':('Near',{'w.1' :0.0,'u.1':0.0}),
        'wF_dirichlet':('Far',{'w.1' :0.0,'u.1':0.0}),
    }
    lcbcs = {
               'imv': ('Omega', {'ls.all' : None}, None, 'integral_mean_value')
            }
    return ebcs, funs, mats, lcbcs


#Definition of macroscopic problem
def define(filename_mesh=None,cg1=None, cg2=None):

    if filename_mesh is None:
        filename_mesh = osp.join(data_dir, 'macro_perf.vtk')
        cg1, cg2 = 0.0015, 0.0015  # y and z coordinates of center of gravity

    mesh = Mesh.from_file(filename_mesh)
    poroela_mezo_file = osp.join(data_dir,'perf_BDB_mes.py')
    material_cache['meso_filename']=poroela_mezo_file

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(mesh.dim, bbox[0], bbox[1], eps=1e-6)

    regions.update({
        'Omega': 'all',
        'Wall': ('r.Top +v r.Bottom +v r.Far +v r.Near', 'facet'),
        'In': ('r.Left -v r.Wall', 'facet'),
        'Out': ('r.Right -v r.Wall', 'facet'),

    })
    ebcs, bc_funs, mats, lcbcs = define_bc(cg1,cg2,val_in=1.e4,val_out=1.e4)

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
        'velocity': ('real', 'vector', 'Omega', 2),
    }

    variables = {
        #Displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        #Pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'ls': ('unknown field', 'pressure'),
        'lv': ('test field', 'pressure', 'ls'),
        #Velocity
        'w': ('unknown field', 'velocity'),
        'z': ('test field', 'velocity', 'w'),
    }

    functions = {
        'get_homog': (lambda ts, coors, problem, mode=None, **kwargs: \
                          get_homog(coors,problem, mode, **kwargs),),    }
    functions.update(bc_funs)

    materials = {
        'hom': 'get_homog',
    }
    materials.update(mats)

    integrals = {
        'i': 4,
        "is": ("s", 4),
    }
    #Definition of solvers
    solvers = {
        'ls': ('ls.mumps', {}),
        'newton': ('nls.newton',
                   {'i_max': 2,
                    'eps_a': 1e-12,
                    'eps_r': 1e-3,
                    'problem': 'nonlinear',
                    })
    }
    #Definition of macroscopic equations, see (43)
    equations = {
        'eq1': """
            dw_lin_elastic.i.Omega(hom.A, v, u)
          - dw_biot.i.Omega(hom.B, v, p)
          - dw_v_dot_grad_s.i.Omega(hom.PT, v, p)
          - dw_volume_dot.i.Omega(hom.H, v, w)
          = 0""",

        'eq2': """
            dw_diffusion.i.Omega(hom.K, q, p)
          - dw_v_dot_grad_s.i.Omega(hom.P, w, q)+ dw_volume_dot.i.Omega( q,ls )
        = + dw_surface_integrate.is.In(w_in.val, q) - dw_surface_integrate.is.Out(w_out.val, q)
          """,

        'eq3': """
            dw_lin_elastic.i.Omega(hom.S, z, w)
          + dw_volume_dot.i.Omega(hom.H, z, w)
          + dw_v_dot_grad_s.i.Omega(hom.PT, z, p)
          = 0""",
        'eq_imv': 'dw_volume_dot.i.Omega( lv, p ) = 0',
    }

    options = {
        'output_dir': data_dir + '/results/macro',
        'ls': 'ls',
        'nls': 'newton',
        'micro_filename' : poroela_mezo_file,
        'absolute_mesh_path': True,
    }
    return locals()