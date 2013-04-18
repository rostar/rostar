from stats.spirrid import make_ogrid as orthogonalize
import numpy as np
from etsproxy.mayavi import mlab
import mayavi

#x,y data generation
CoV_tau_range = [1e-7, 0.5]
CoV_r_range = [1e-7, 0.5]
dp_tau = 20
dp_r = 20
CoV_tau_arr = np.linspace( CoV_tau_range[0], CoV_tau_range[1], dp_tau )
#CoV_tau_arr = CoV_tau_arr[::-1]
CoV_r_arr = np.linspace( CoV_r_range[0], CoV_r_range[1], dp_r )
e_arr = orthogonalize( [CoV_tau_arr, CoV_r_arr] )
x_axis1 = e_arr[0]
y_axis1 = e_arr[1]
list( CoV_tau_arr )
#x,y,z plot


# ------------------------------------------- 
def sigma_m( m ):
    dataname = "sigmaOPT20_with_m{}.npy".format( m )
    res = np.load( dataname )
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
    if len( engine.scenes ) == 0:
        engine.new_scene()
    mlab.surf( x_axis1 , y_axis1 , res  , representation = 'wireframe' )
    surface = engine.scenes[0].children[0].children[0].children[0].children[0].children[0]
    surface.actor.mapper.scalar_range = np.array( [ 6.97602671, 8.8533387 ] )
    surface.actor.mapper.scalar_visibility = False
    scene = engine.scenes[0]
    scene.scene.background = ( 1.0, 1.0, 1.0 )
    surface.actor.property.specular_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.diffuse_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.ambient_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.line_width = 1.
    warp_scalar = engine.scenes[0].children[0].children[0]
    module_manager = engine.scenes[0].children[0].children[0].children[0].children[0]
    warp_scalar.filter.normal = np.array( [ 0. , 0. , 0.2] )
    module_manager.scalar_lut_manager.scalar_bar.global_warning_display = 1
    module_manager.scalar_lut_manager.scalar_bar.position2 = np.array( [ 0.8 , 0.17] )
    module_manager.scalar_lut_manager.scalar_bar.position = np.array( [ 0.1 , 0.01] )
    module_manager.scalar_lut_manager.data_range = np.array( [ 6.97602671, 8.8533387 ] )
    module_manager.scalar_lut_manager.default_data_range = np.array( [ 6.97602671, 8.8533387 ] )
    module_manager.vector_lut_manager.scalar_bar.global_warning_display = 1
    module_manager.vector_lut_manager.scalar_bar.position2 = np.array( [ 0.8 , 0.17] )
    module_manager.vector_lut_manager.scalar_bar.position = np.array( [ 0.1 , 0.01] )
    module_manager.vector_lut_manager.data_range = np.array( [ 0., 1.] )
    module_manager.vector_lut_manager.default_data_range = np.array( [ 0., 1.] )
    scene.scene.isometric_view()

    scene.scene.camera.position = [1.2836424071875543, 1.4371492101974028, 4.0390558511994534]
    scene.scene.camera.focal_point = [0.17361105930834225, 0.21417386120592863, 3.4874067491767562]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.21371002618964222, -0.23386371429877953, 0.94849132196367569]
    scene.scene.camera.clipping_range = [0.79163912587841923, 2.7365159886347699]
    scene.scene.camera.compute_view_plane_normal()

    #mlab.surf( x_axis2, y_axis2, res2 )
    #mlab.xlabel( "rand tau" )
    #mlab.ylabel( "rand r" )
    #mlab.zlabel( "z" )
    mlab.show()

def w_m( m ):
    dataname = "wOPT20_with_m{}.npy".format( m )
    res = np.load( dataname )
    #res2 = np.load( "sigmaAN_with_m7.0.npy" )
    
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
    if len( engine.scenes ) == 0:
        engine.new_scene()
    mlab.surf( x_axis1 , y_axis1 , res  , representation = 'wireframe' )
    
    surface = engine.scenes[0].children[0].children[0].children[0].children[0].children[0]
    surface.actor.mapper.scalar_range = np.array( [ 6.97602671, 8.8533387 ] )
    surface.actor.mapper.scalar_visibility = False
    scene = engine.scenes[0]
    scene.scene.background = ( 1.0, 1.0, 1.0 )
    surface.actor.property.specular_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.diffuse_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.ambient_color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.color = ( 0.0, 0.0, 0.0 )
    surface.actor.property.line_width = 1.
    warp_scalar = engine.scenes[0].children[0].children[0]
    module_manager = engine.scenes[0].children[0].children[0].children[0].children[0]
    warp_scalar.filter.normal = np.array( [ 0. , 0. , 15] )
    module_manager.scalar_lut_manager.scalar_bar.global_warning_display = 1
    module_manager.scalar_lut_manager.scalar_bar.position2 = np.array( [ 0.8 , 0.17] )
    module_manager.scalar_lut_manager.scalar_bar.position = np.array( [ 0.1 , 0.01] )
    module_manager.scalar_lut_manager.data_range = np.array( [ 6.97602671, 8.8533387 ] )
    module_manager.scalar_lut_manager.default_data_range = np.array( [ 6.97602671, 8.8533387 ] )
    module_manager.vector_lut_manager.scalar_bar.global_warning_display = 1
    module_manager.vector_lut_manager.scalar_bar.position2 = np.array( [ 0.8 , 0.17] )
    module_manager.vector_lut_manager.scalar_bar.position = np.array( [ 0.1 , 0.01] )
    module_manager.vector_lut_manager.data_range = np.array( [ 0., 1.] )
    module_manager.vector_lut_manager.default_data_range = np.array( [ 0., 1.] )
    scene.scene.isometric_view()

    scene.scene.camera.position = [1.2836424071875543, 1.4371492101974028, 4.0390558511994534]
    scene.scene.camera.focal_point = [0.17361105930834225, 0.21417386120592863, 3.4874067491767562]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.21371002618964222, -0.23386371429877953, 0.94849132196367569]
    scene.scene.camera.clipping_range = [0.79163912587841923, 2.7365159886347699]
    scene.scene.camera.compute_view_plane_normal()

    mlab.show()

m = 7.0
sigma_m( m )
#w_m( m )
