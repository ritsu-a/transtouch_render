# By Jet <i@jetd.me>
#
import sapien.core as sapien
import numpy as np
import transforms3d.euler
from sapien.core import Pose
import PIL.Image as im
import transforms3d as t3d
import os
import json
import math
from render_utils import *
from transforms3d.euler import euler2mat, mat2euler

OBJECT_DIR=f"/share/pengyang/render/models_sim_real/models/"

def load_obj(scene, obj_name, renderer, pose=Pose(), is_kinematic=False, material_name="kuafu_material"):
    builder = scene.create_actor_builder()
    kuafu_material_path = os.path.join(OBJECT_DIR, obj_name, f"{material_name}.json")
    mesh_list_file = os.path.join(OBJECT_DIR, obj_name, "visual_mesh_list.txt")
    if os.path.exists(mesh_list_file):
        load_mesh_list(builder, mesh_list_file, renderer, material_name)
    elif os.path.exists(kuafu_material_path):
        obj_material = load_kuafu_material(kuafu_material_path, renderer)
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"), material=obj_material)
    else:
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "collision_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)

    return obj




def cv2ex2pose(ex):
    ros2opencv = np.array([[0., -1., 0., 0.],
                           [0., 0., -1., 0.],
                           [1., 0., 0., 0.],
                           [0., 0., 0., 1.]], dtype=np.float32)

    pose = np.linalg.inv(ex) @ ros2opencv

    from transforms3d.quaternions import mat2quat

    return sapien.Pose(pose[:3, 3], mat2quat(pose[:3, :3]))

def main():
    #use_kuafu = False
    use_kuafu = True

    sim = sapien.Engine()
    sim.set_log_level('warning')
    sapien.KuafuRenderer.set_log_level('warning')

    if use_kuafu:
        render_config = sapien.KuafuConfig()
        render_config.use_viewer = False
        render_config.spp = 32
        render_config.max_bounces = 8
        render_config.use_denoiser = False
        renderer = sapien.KuafuRenderer(render_config)
    else:
        renderer = sapien.VulkanRenderer(offscreen_only=False)

    sim.set_renderer(renderer)

    scene_config = sapien.SceneConfig()

    scene = sim.create_scene(scene_config)

    camera_mount = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(30),
        near=0.1,
        far=100,
    )
    
    # camera.set_perspective_parameters(0, 1000,
    #                                 9.0416251695769188e+02,
    #                                 9.0576564751998546e+02,
    #                                 6.5475305927534237e+02,
    #                                 3.7820226912221301e+02,
    #                                 0)

    camera.set_perspective_parameters(0.1, 100,
                                9.0449392394010329e+02,
                                9.0705079051427947e+02,
                                6.3950607626314161e+02,
                                3.5955400321720902e+02,
                                0)
    
    tablecam_pose_mat =   [[-0.36484535, 0.9310666 , 0.00169146,-0.08190167],
                        [ 0.76365924, 0.30028384,-0.57153669, 0.06015198],
                        [-0.53264664,-0.2072308 ,-0.82057477, 0.89746678],
                        [ 0.        , 0.        , 0.        , 1.        ]]
    camera_mount.set_pose( cv2ex2pose(tablecam_pose_mat))


    # ground_material = renderer.create_material()
    # ground_material.base_color = np.array([0, 1., 0, 1.])
    # ground_material.specular = 0.5
    # scene.add_ground(0, render_material=ground_material)
    # scene.set_timestep(1 / 240)

    
    ior = 1.0

    # TODO : return back to 0.1
    ambient = 1

    scene.set_ambient_light([ambient, ambient, ambient])
    
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.5
    material.transmission = 0.0
    material.metallic = 0.2
    material.specular = 0.2
    material.set_diffuse_texture_from_file("optical_table/table.png")
    builder.add_visual_from_file("optical_table/optical_table.obj", material=material)
    table = builder.build_kinematic(name="table_kuafu")
    # table.set_pose(sapien.Pose([0.405808, 0.022201, -0.043524,], 
    #                             [0.999921, -0.000290915, -0.00932814, 0.00842011]))                    


    with open ("/share/sim2real_tactile/real/real_data_v10/0-300163-2/meta.pkl", "rb") as file:
        meta = pickle.load(file)


    # # for idx in range(len(meta["object_names"])):
    # #     object_name = meta["object_names"][idx]
    # #     object_pose = meta["poses_world"][meta["object_ids"][idx]]
    # #     #object_T1 = sapien.Pose.to_transformation_matrix(object_pose)
    # #     with open(f"/share/pengyang/render/scene_json/paper_{object_name}/input.json","r") as file:
    # #         js = json.load(file)

    # #     print(object_name)

    # #     for name, object_T in js.items():
    # #         load_obj(
    # #             scene,
    # #             object_name,
    # #             renderer=renderer,
    # #             pose=sapien.Pose.from_transformation_matrix(np.linalg.inv(table_world_mat) @ object_pose),
    # #             is_kinematic=True,
    # #             material_name="kuafu_material_new2",
    # #         )

    scene.step()
    scene.update_render()
    camera.take_picture()

    p_l = camera.get_color_rgba()
    p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
    p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2BGR)

    cv2.imwrite('/share/pengyang/render/kuafu_table_results/sim_photo.png', p_l)

    p_gt = cv2.imread("/share/pengyang/render/kuafu_table_results/ir_photo0_l.png")

    p_combine = (p_l + p_gt) // 2
    cv2.imwrite('/share/pengyang/render/kuafu_table_results/combine.png', p_combine)


    mtx = camera.get_camera_matrix()[:3, :3]
    dist = np.array([ 2.2417902208261261e-03, 2.6654767216023081e-02,
       -4.9094837920522500e-04, 7.2984677314572055e-04,
       -6.3162091851993743e-02 ])
    p_undistort = cv2.undistort(p_gt, mtx, dist, None, mtx)
    cv2.imwrite('/share/pengyang/render/kuafu_table_results/undistort.png', p_undistort)

    p_combine1 = (p_l + p_undistort) // 2
    cv2.imwrite('/share/pengyang/render/kuafu_table_results/combine_undistort.png', p_combine1)

    




main()

