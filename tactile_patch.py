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
import argparse

OBJECT_DIR=f"/share/pengyang/render/models_v2/models/"


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

def load_obj_vk(scene, obj_name, pose=Pose(), is_kinematic=False):
    builder = scene.create_actor_builder()
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



def parse_args():
    parser = argparse.ArgumentParser(description="Generate tactile mask from patch list")
    parser.add_argument(
        "-n",
        dest="save_dir",
        default="tune",
        help="where to save the mask",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    TAR_DIR = f"/share/pengyang/render/{args.save_dir}/"
    sim = sapien.Engine()
    sim.set_log_level('warning')
    sapien.KuafuRenderer.set_log_level('warning')
    
    renderer = sapien.VulkanRenderer(offscreen_only=False)
    sim.set_renderer(renderer)

    
    with open("chosen_patch.txt", "r") as file:
        lines = file.readlines()

    ### loading masks

    masks = {}
    for view_id in range(1, 18):
        rgb_mask = cv2.imread(f"/share/pengyang/sim2real_active_tactile/active_zero2/assets/real_robot_masks/m{view_id}.png")
        rgb_mask = cv2.resize(rgb_mask[:, :, 0], (1280, 720))
        mask = (rgb_mask == rgb_mask.min())
        masks[view_id] = mask


    for line in lines:
 
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
        #camera.set_perspective_parameters(0.1, 100,1382.246,1381.136,943.902,557.413,0)
        
        # intrinsic_l
        #  camera.set_perspective_parameters(0.1, 100,1344.0165,1344.0165,951.879,541.428,0)

        # ground_material = renderer.create_material()
        # ground_material.base_color = np.array([0, 1., 0, 1.])
        # ground_material.specular = 0.5
        # scene.add_ground(0, render_material=ground_material)
        # scene.set_timestep(1 / 240)

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

        SCENE_NAME = line.split(" ")[0]

        coords_3d = []            
        SAVE_DIR = TAR_DIR + SCENE_NAME + "/"
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open (f"/share/datasets/sim2real_tactile/real/dataset/" + SCENE_NAME + "/meta.pkl", "rb") as file:
            meta = pickle.load(file)
        intrinsic_l = meta["intrinsic_l"]

        camera.set_perspective_parameters(0.1, 100,intrinsic_l[0, 0],intrinsic_l[1, 1],intrinsic_l[0, 2],intrinsic_l[1, 2],0)
        camera_mount.set_pose( cv2ex2pose(meta["extrinsic_l"]))

        patch_list = []
        for str_pair in line.strip().split(" ")[1:]:
            x = int(str_pair.strip().strip("(").strip(")").split(",")[0])
            y = int(str_pair.strip().strip("(").strip(")").split(",")[1])
            patch_list.append((x, y))
        print(SCENE_NAME, patch_list)
        for idx in range(len(meta["object_names"])):
            object_name = meta["object_names"][idx]
            object_pose = meta["poses_world"][meta["object_ids"][idx]]
            load_obj_vk(
                scene,
                object_name,
                pose=sapien.Pose.from_transformation_matrix(object_pose),
                is_kinematic=True,
            )
        scene.step()
        scene.update_render()
        camera.take_picture()


        p_l = camera.get_color_rgba()
        p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
        p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2BGR)
        cor = p_l.copy()
        cv2.imwrite(SAVE_DIR + 'rgb.png', p_l)

        normal = camera.get_float_texture("Normal")
        pos = camera.get_float_texture("Position")
        depth = -pos[..., 2]
        mask = np.zeros_like(depth).astype(bool)

        table_mask = ~(depth == 0)
        img_depth_l = (
                    cv2.imread(f"/share/datasets/sim2real_tactile/real/dataset/" + SCENE_NAME + "/depthL.png"\
                        , cv2.IMREAD_UNCHANGED).astype(float) / 1000
                )

        print(img_depth_l.shape)
        print(depth.shape)
        print(img_depth_l[500:510, 500:510])
        print("-" * 50)
        depth1 = np.round(depth - 0.0005, 3)



        print((depth1 - img_depth_l)[table_mask].mean())
        print("-" * 50)

        def get_normal(x, y):
            # return np.ndarray((4)), the last axis is 0
            return camera.get_model_matrix() @ normal[x, y]
        def coord_2d_to_3d(x, y):
            # map a pixel to 3d world frame
            # (x, y) in the range (1080, 1920)
            print(x, y)
            pt = np.array([y, x, 1, 0]) * depth[x, y] + np.array([0, 0, 0, 1])
            camera_frame = np.linalg.inv(camera.get_camera_matrix()) @ pt
            world_frame = np.linalg.inv(meta["extrinsic_l"]) @ camera_frame
            return world_frame

        normal_baseline = []

        for (x, y) in patch_list:
            x = int(x / 960 * 1280)
            y = int(y / 540 * 720)
            coord = coord_2d_to_3d(x, y)
            coords_3d.append(coord)
            normal_baseline.append(get_normal(x, y))
        
        print(normal_baseline)
        print("*" * 50)

        for view in range(1):
            SUB_SCENE_NAME = SCENE_NAME
            view = int(SCENE_NAME.split("-")[1])
            print("+" * 50)
            print(SUB_SCENE_NAME)
            SAVE_DIR = TAR_DIR + SUB_SCENE_NAME + "/"
            os.makedirs(SAVE_DIR, exist_ok=True)
            with open (f"/share/datasets/sim2real_tactile/real/dataset/" + SUB_SCENE_NAME + "/meta.pkl", "rb") as file:
                meta = pickle.load(file)
            intrinsic_l = meta["intrinsic_l"]

            camera_mount.set_pose( cv2ex2pose(meta["extrinsic_l"]))
            scene.step()
            scene.update_render()
            camera.take_picture()

            p_l = camera.get_color_rgba()
            p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2BGR)
            cor = p_l.copy()
            cv2.imwrite(SAVE_DIR + 'rgb.png', p_l)

            normal = camera.get_float_texture("Normal")
            pos = camera.get_float_texture("Position")
            depth = -pos[..., 2]
            mask = np.zeros_like(depth).astype(bool)

            table_mask = ~(depth == 0)
            table_mask = masks[view]
            img_depth_l = (
                        cv2.imread(f"/share/datasets/sim2real_tactile/real/dataset/" + SUB_SCENE_NAME + "/depthL.png"\
                            , cv2.IMREAD_UNCHANGED).astype(float) / 1000
                    )

            # print(img_depth_l.shape)
            # print(img_depth_l[500:510, 500:510])
            print("-" * 50)
            depth1 = np.round(depth - 0.0005, 3)
            print((depth1 - img_depth_l)[table_mask].mean())
            print("-" * 50)


            def get_normal_sub(x, y):
                # return np.ndarray((4)), the last axis is 0
                return camera.get_model_matrix() @ normal[x, y]

            def coord_2d_to_3d_sub(x, y):
                # map a pixel to 3d world frame
                # (x, y) in the range (1080, 1920)
                pt = np.array([y, x, 1, 0]) * depth[x, y] + np.array([0, 0, 0, 1])
                camera_frame = np.linalg.inv(camera.get_camera_matrix()) @ pt
                world_frame = np.linalg.inv(meta["extrinsic_l"]) @ camera_frame
                return world_frame

            def coord_3d_to_2d_sub(world_frame):
                camera_frame = meta["extrinsic_l"] @ world_frame
                pt = camera.get_camera_matrix() @ camera_frame 
                x = int(pt[1] / pt[2])
                y = int(pt[0] / pt[2])
                return x, y

            for coord, nor_baseline in zip(coords_3d, normal_baseline):
                x, y = coord_3d_to_2d_sub(coord)
                nor = get_normal_sub(x, y) 
                print("1" * 50)
                print(nor.T @ nor_baseline)
                if nor.T @ nor_baseline < 0.999:
                    continue
                for i in range(max(0, x - 20), min(720, x + 20)):
                    for j in range(max(0, y - 20), min(1280, y + 20)):
                        coord1 = coord_2d_to_3d_sub(i, j)
                        if np.linalg.norm(coord - coord1) < 0.0025:
                            # and abs(nor.T @ (coord - coord1)) < 0.001:
                            cor[i, j, :] = 255
                            mask[i, j] = True

            cv2.imwrite(SAVE_DIR + 'tactile_photo.png', cor)
            plt.imsave(os.path.join(SAVE_DIR, "mask.png"), mask, vmin=0, vmax=1, cmap="jet")
            


        del scene

        print(meta["scales"])
        print(meta["object_ids"])
        print(meta["extents"])
        print(meta["intrinsic"])
        print(meta["intrinsic_l"])
        print(meta["intrinsic_r"])





    

main()

