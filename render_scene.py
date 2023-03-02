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


OBJECT_DIR=f"/share/pengyang/render/models_v2/models/"

os.makedirs("/share/pengyang/render/results_render_scene", exist_ok=True)



def parse_csv(filename):
    """Parse the CSV file to acquire the information of objects used in TOC.

    Args:
        filename (str): a CSV file containing object information.

    Returns:
        dict: {str: dict}, object_name -> object_info
    """
    object_db = OrderedDict()
    with open(filename, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            object_db[row["object"]] = row
    return object_db

OBJECT_CSV_PATH = "/share/pengyang/render/objects_v2.csv"
OBJECT_DB = parse_csv(OBJECT_CSV_PATH)
OBJECT_NAMES = tuple(OBJECT_DB.keys())
NUM_OBJECTS = len(OBJECT_NAMES)

# For visualization
# Note that COLOR_PALETTE is a [C, 3] np.uint8 array.
# The last color is used for background.
# COLOR_PALETTE = np.array([ImageColor.getrgb(color)
#                           for color in np.unique(list(ImageColor.colormap.values()))], dtype=np.uint8)
# COLOR_PALETTE = COLOR_PALETTE[:NUM_OBJECTS + 3]
cmap = plt.get_cmap("rainbow", NUM_OBJECTS)
COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
COLOR_PALETTE[-3] = [119, 135, 150]
COLOR_PALETTE[-2] = [176, 194, 216]
COLOR_PALETTE[-1] = [255, 255, 225]

def get_seg_id_to_obj_id(seg_id_to_obj_name, max_seg_id=None):
    """Get the mapping from seg_id to obj_id.

    The seg_id (segmentation id) corresponds to an object,
    which might be beyond the objects of interest.

    Args:
        seg_id_to_obj_name (dict[int, str]): seg_id -> object name.
        max_seg_id (int): maximum segmentation id.

    Returns:
        np.ndarray: [max_seg_id + 1], an integer array.
            -1 for not cared; -2 for ground; -3 for table
    """
    if max_seg_id is None:
        max_seg_id = max(list(seg_id_to_obj_name.keys()))
    seg_id_to_obj_id = np.full(max_seg_id + 1, NUM_OBJECTS, dtype=int)
    for seg_id, obj_name in seg_id_to_obj_name.items():
        if obj_name == "ground":
            seg_id_to_obj_id[seg_id] = NUM_OBJECTS + 1
        elif obj_name == "table":
            seg_id_to_obj_id[seg_id] = NUM_OBJECTS + 2
        else:
            try:
                obj_id = OBJECT_NAMES.index(obj_name)
                seg_id_to_obj_id[seg_id] = obj_id
            except ValueError:
                pass
    return seg_id_to_obj_id


def load_obj_vk(scene, obj_name, pose=Pose(), is_kinematic=False):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)
    return obj


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
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
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

def render_scene(args):

    SAVE_DIR = f"/share/pengyang/render/results_render_scene/{args.scene_name}/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    seed = int(np.random.rand() * 100000)
    print("-" * 80)
    print("seed = ", seed)
    print("-" * 80)
    random.seed(seed)
    np.random.seed(seed)



    sim = sapien.Engine()
    sim.set_log_level('warning')
    # sapien.KuafuRenderer.set_log_level('warning')

    renderer = sapien.VulkanRenderer(offscreen_only=False)
    sapien.VulkanRenderer.set_log_level('warning')

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
    
    camera.set_perspective_parameters(0.1, 100,
                            8.9687159287313307e+02,
                            8.9685095756783505e+02,
                            6.4037813319745328e+02,
                            3.7131751630652178e+02,
                            0)
                            
    tablecam_pose_mat =    [[-0.99998931,-0.00324778,-0.00328973,-0.08225272],
                            [-0.00332778, 0.99969158, 0.02461047, 0.01168893],
                            [ 0.00320878, 0.02462115,-0.9996917 , 0.65327796],
                            [ 0.        , 0.        , 0.        , 1.        ]]
    camera_mount.set_pose( cv2ex2pose(tablecam_pose_mat))


    ground_material = renderer.create_material()
    ground_material.base_color = np.array([0, 1., 0, 1.])
    ground_material.specular = 0.5
    scene.add_ground(0.04, render_material=ground_material)
    scene.set_timestep(1 / 240)


    # TODO : return back to 0.1
    ambient = 1

    scene.set_ambient_light([ambient, ambient, ambient])
    
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.5
    material.metallic = 0.2
    material.specular = 0.2
    material.set_diffuse_texture_from_file("optical_table/table.png")
    builder.add_visual_from_file("optical_table/optical_table.obj", material=material)
    # builder.add_multiple_collisions_from_file("optical_table/optical_table.obj")
    table = builder.build_kinematic(name="table_kuafu")


    objects_raw = [ 
        "beaker_small",
        "flask1",
        "flask4",
        "spellegrino",
        "beer_can",
        "camera",
        "cellphone",
        "champagne",
        "coca_cola",
        "coffee_cup",
        "coke_bottle",
        "gold_ball",
        "hammer",
        "jack_daniels",
        "pepsi_bottle",
        "rubik",
        "sharpener",
        "steel_ball",
        "tennis_ball",
        "voss",
    ]
    
    random.shuffle(objects_raw)

    UNROTATABLE_OBJECTS = [ 
        "beaker_small",
        "flask1",
        "flask4",
        "spellegrino",
        "beer_can",
        "camera",
        "cellphone",
        "champagne",
        "coca_cola",
        "coffee_cup",
        "coke_bottle",
        "gold_ball",
        "hammer",
        "jack_daniels",
        "pepsi_bottle",
        "rubik",
        "sharpener",
        "steel_ball",
        "tennis_ball",
        "voss",
    ]


                       

    UNROTATABLE_OBJECTS_MAP = {
        k : False for k in UNROTATABLE_OBJECTS
    }



    for idx in range(5):
        object_name = objects_raw[idx]
        if object_name in UNROTATABLE_OBJECTS:
            UNROTATABLE_OBJECTS_MAP[object_name] = True

    objects = []

    for object_name in UNROTATABLE_OBJECTS:
        if UNROTATABLE_OBJECTS_MAP[object_name]:
            objects.append(object_name)



    # keep rendering until no collision
    num = 0
    for idx in range(len(objects)):
        num += 1
        if num > 100:
            break
        object_name = objects[idx]

        finished = False
        while finished == False:


            # raw = pi / 2 correspond to y_dim
            # pitch = pi / 2 correspond to x_dim


            x = np.random.rand() * 0.5 - 0.3375
            y = np.random.rand() * 0.375 - 0.2125
            z = 0.08

            if object_name == "funnel":
                z = 0.01

            scale = 0.5

            x = (x + 0.0875) * scale - 0.0875
            y = (y + 0.025) * scale - 0.025 

            print(x, y)

            raw = 0
            pitch = 0
            roll = np.random.rand() * np.pi * 2
            # roll = 0
            # roll = np.pi / 2

            z += float(OBJECT_DB[object_name]["z_dim"]) / 2

            object_pose = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z],
                                    [0, 0, 0, 1],])

            object_pose[:3, :3] = euler2mat(raw, pitch, roll)

            
           
            actor = load_obj_vk(
                    scene,
                    object_name,
                    pose=sapien.Pose.from_transformation_matrix(object_pose),
                    is_kinematic=False,
                )
            for i in range(1000):
                scene.step()
            partly_finished = True
            for contact in scene.get_contacts():
                if contact.actor0.name != "ground" and contact.actor1.name != "ground" \
                    and contact.actor0.name != "table_kuafu" and contact.actor1.name != "table_kuafu":
                    scene.remove_actor(actor)
                    renderer.clear_cached_resources()
                    partly_finished = False
                    break

            if partly_finished:
                object_pose = actor.get_pose() 
                pose_mat = object_pose.to_transformation_matrix()

                scene.remove_actor(actor)

                renderer.clear_cached_resources()
                x = pose_mat[0, 3] + 0.0875
                y = pose_mat[1, 3] + 0.025
                if abs(x) > 0.3 or abs(y) > 0.25:
                    continue
                load_obj_vk(
                    scene,
                    object_name,
                    pose=object_pose,
                    is_kinematic=True,
                )
                finished = True

    scene.step()
    scene.update_render()
    camera.take_picture()

    for actor in scene.get_all_actors():
        print(actor.get_name(), mat2euler((actor.get_pose().to_transformation_matrix()[:3, :3])))
        print(actor.get_pose().to_transformation_matrix()[0, 3])
        print(actor.get_pose().to_transformation_matrix()[1, 3])
        print(actor.get_pose().to_transformation_matrix()[2, 3])

    # p_rgb = camera.get_color_rgba()
    # p_rgb = (p_rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)
    # p_rgb = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(SAVE_DIR + 'sim_photo.png', p_rgb)


    # obj_segmentation = camera.get_uint32_texture("Segmentation")[..., 1]


    poses_world = [None for _ in range(NUM_OBJECTS)]
    extents = [None for _ in range(NUM_OBJECTS)]
    scales = [None for _ in range(NUM_OBJECTS)]
    obj_ids = []
    object_names = []

    print(scene.get_all_actors())

    seg_id_to_obj_name = {}

    for actor in scene.get_all_actors()[3:]:
        obj_name = actor.get_name()
        #print(OBJECT_DB.keys())
        if object_name in OBJECT_DB.keys():
            #print(obj_name)
            pose = actor.get_pose().to_transformation_matrix()
            obj_id = OBJECT_NAMES.index(obj_name)
            obj_ids.append(obj_id)
            object_names.append(obj_name)
            poses_world[obj_id] = pose
            extents[obj_id] = np.array(
                [
                    float(OBJECT_DB[obj_name]["x_dim"]),
                    float(OBJECT_DB[obj_name]["y_dim"]),
                    float(OBJECT_DB[obj_name]["z_dim"]),
                ],
                dtype=np.float32,
            )
            scales[obj_id] = np.ones(3)
            seg_id_to_obj_name[actor.get_id()] = object_name
    seg_id_to_sapien_name = {actor.get_id(): actor.get_name() for actor in scene.get_all_actors()}
    seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
    seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
    
    
    meta = {
        "scene_name": args.scene_name,
        "seed": seed,
        "poses_world": poses_world,
        "extents": extents,
        "scales": scales,
        "object_ids": obj_ids,
        "object_names": object_names,
        # this l actually denotes rgb
        "extrinsic_l" : tablecam_pose_mat,
        "intrinsic_l" : camera.get_camera_matrix(),
    }

    print(meta)
    with open(SAVE_DIR + "raw_meta.pkl", "wb") as file:
        pickle.dump(meta, file)


    # assert not np.any(obj_segmentation > 255)
    # # Mapping seg_id to obj_id, as a semantic label image.
    # seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
    # seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
    # obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])
    # # Semantic labels
    # sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
    # sem_image = im.fromarray(sem_labels)
    # sem_image.save(os.path.join(SAVE_DIR, "labelL.png"))
    # sem_labels_with_color = COLOR_PALETTE[sem_labels]
    # sem_image_with_color = im.fromarray(sem_labels_with_color.astype("uint8"))
    # sem_image_with_color.save(os.path.join(SAVE_DIR, "labelL2.png"))

    return 
    
def recover_with_kuafu(args):
    # render with kuafu
    SAVE_DIR = f"/share/pengyang/render/results_render_scene/{args.scene_name}/"
    sim = sapien.Engine()
    sim.set_log_level("warning")

    render_config = sapien.KuafuConfig()
    render_config.use_viewer = False
    render_config.use_denoiser = False
    render_config.spp = 128
    render_config.max_bounces = 8

    renderer = sapien.KuafuRenderer(render_config)
    sim.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    
    scene = sim.create_scene(scene_config)


    camera_mount_base = scene.create_actor_builder().build_kinematic()
    camera_base = scene.add_mounted_camera(
        name="camera_base",
        actor=camera_mount_base,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(30),
        near=0.1,
        far=100,
    )
    camera_base.set_perspective_parameters(0.1, 100,
                            9.1020077777522272e+02,
                            9.1038247935099866e+02,
                            6.5907945440859100e+02,
                            3.7481157428344886e+02,
                            0)
    basecam_pose_mat =       [[-0.37124701, 0.92851306,-0.00625727,-0.0551394 ],
                                [ 0.76209405, 0.30084409,-0.57332844, 0.06398138],
                                [-0.53046048,-0.2176151 ,-0.81930174, 0.89167432],
                                [ 0.        , 0.        , 0.        , 1.        ]]
    camera_mount_base.set_pose( cv2ex2pose(basecam_pose_mat))
 

    camera_mount_hand = scene.create_actor_builder().build_kinematic()
    camera_hand = scene.add_mounted_camera(
        name="camera_hand",
        actor=camera_mount_hand,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(30),
        near=0.1,
        far=100,
    )
    camera_hand.set_perspective_parameters(0.1, 100,
                            8.9687159287313307e+02,
                            8.9685095756783505e+02,
                            6.4037813319745328e+02,
                            3.7131751630652178e+02,
                            0)

    tablecam_pose_mat =    [[-0.99998931,-0.00324778,-0.00328973,-0.08225272],
                            [-0.00332778, 0.99969158, 0.02461047, 0.01168893],
                            [ 0.00320878, 0.02462115,-0.9996917 , 0.65327796],
                            [ 0.        , 0.        , 0.        , 1.        ]]



    camera_mount_hand.set_pose( cv2ex2pose(tablecam_pose_mat))

    ground_material = renderer.create_material()
    ground_material.base_color = np.array([0, 1., 0, 1.])
    ground_material.specular = 0.5
    scene.add_ground(0, render_material=ground_material)
    scene.set_timestep(1 / 240)

    ambient = 1.0
    scene.set_ambient_light([ambient, ambient, ambient])
    
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.5
    material.metallic = 0.2
    material.specular = 0.2
    material.set_diffuse_texture_from_file("optical_table/table.png")
    builder.add_visual_from_file("optical_table/optical_table.obj", material=material)
    builder.add_multiple_collisions_from_file("optical_table/optical_table.obj")
    table = builder.build_kinematic(name="table_kuafu")


    with open("/share/pengyang/render/results_render_scene/" + str(args.scene_name) + "/raw_meta.pkl", "rb") as file:
        meta = pickle.load(file)
        print(meta["seed"])

    for idx in range(len(meta["object_names"])):
        object_name = meta["object_names"][idx]
        object_pose = meta["poses_world"][meta["object_ids"][idx]]

        print(object_name, object_pose)
        print(mat2euler(object_pose))
        load_obj(
            scene,
            object_name,
            renderer=renderer,
            pose=sapien.Pose.from_transformation_matrix(object_pose),
            is_kinematic=True,
            material_name="kuafu_material",
        )     
    
    scene.step()
    scene.update_render()
    camera_hand.take_picture()
    camera_base.take_picture()

    p_rgb = camera_hand.get_color_rgba()
    p_rgb = (p_rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)
    p_rgb = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(SAVE_DIR + 'kuafu_hand.png', p_rgb)

    p_rgb_base = camera_base.get_color_rgba()
    p_rgb_base = (p_rgb_base[..., :3] * 255).clip(0, 255).astype(np.uint8)
    p_rgb_base = cv2.cvtColor(p_rgb_base, cv2.COLOR_RGB2BGR)

    cv2.imwrite(SAVE_DIR + 'kuafu_base.png', p_rgb_base)

def recover_with_vulkan(args):
    # render with vulkan
    SAVE_DIR = f"/share/pengyang/render/results_render_scene/{args.scene_name}/"
    sim = sapien.Engine()
    sim.set_log_level("warning")

    renderer = sapien.VulkanRenderer(offscreen_only=False)
    sapien.VulkanRenderer.set_log_level('warning')
    sim.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    
    scene = sim.create_scene(scene_config)


    camera_mount_base = scene.create_actor_builder().build_kinematic()
    camera_base = scene.add_mounted_camera(
        name="camera_base",
        actor=camera_mount_base,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(30),
        near=0.1,
        far=100,
    )
    camera_base.set_perspective_parameters(0.1, 100,
                            9.1020077777522272e+02,
                            9.1038247935099866e+02,
                            6.5907945440859100e+02,
                            3.7481157428344886e+02,
                            0)
    basecam_pose_mat =       [[-0.37124701, 0.92851306,-0.00625727,-0.0551394 ],
                                [ 0.76209405, 0.30084409,-0.57332844, 0.06398138],
                                [-0.53046048,-0.2176151 ,-0.81930174, 0.89167432],
                                [ 0.        , 0.        , 0.        , 1.        ]]
    camera_mount_base.set_pose( cv2ex2pose(basecam_pose_mat))
 

    camera_mount_hand = scene.create_actor_builder().build_kinematic()
    camera_hand = scene.add_mounted_camera(
        name="camera_hand",
        actor=camera_mount_hand,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=1280,
        height=720,
        fovy=np.deg2rad(30),
        near=0.1,
        far=100,
    )
    camera_hand.set_perspective_parameters(0.1, 100,
                            8.9687159287313307e+02,
                            8.9685095756783505e+02,
                            6.4037813319745328e+02,
                            3.7131751630652178e+02,
                            0)

    tablecam_pose_mat =    [[-0.99998931,-0.00324778,-0.00328973,-0.08225272],
                            [-0.00332778, 0.99969158, 0.02461047, 0.01168893],
                            [ 0.00320878, 0.02462115,-0.9996917 , 0.65327796],
                            [ 0.        , 0.        , 0.        , 1.        ]]



    camera_mount_hand.set_pose( cv2ex2pose(tablecam_pose_mat))

    ground_material = renderer.create_material()
    ground_material.base_color = np.array([0, 1., 0, 1.])
    ground_material.specular = 0.5
    scene.add_ground(0, render_material=ground_material)
    scene.set_timestep(1 / 240)

    ambient = 1.0
    scene.set_ambient_light([ambient, ambient, ambient])
    
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.5
    material.metallic = 0.2
    material.specular = 0.2
    material.set_diffuse_texture_from_file("optical_table/table.png")
    builder.add_visual_from_file("optical_table/optical_table.obj", material=material)
    builder.add_multiple_collisions_from_file("optical_table/optical_table.obj")
    table = builder.build_kinematic(name="table_kuafu")


    with open("/share/pengyang/render/results_render_scene/" + str(args.scene_name) + "/raw_meta.pkl", "rb") as file:
        meta = pickle.load(file)

    seg_id_to_obj_name = {}

    for idx in range(len(meta["object_names"])):
        object_name = meta["object_names"][idx]
        object_pose = meta["poses_world"][meta["object_ids"][idx]]

        actor = load_obj_vk(
            scene,
            object_name,
            pose=sapien.Pose.from_transformation_matrix(object_pose),
            is_kinematic=True,
        )         
        seg_id_to_obj_name[actor.get_id()] = object_name
    
    scene.step()
    scene.update_render()
    camera_hand.take_picture()
    camera_base.take_picture()

    seg_id_to_sapien_name = {actor.get_id(): actor.get_name() for actor in scene.get_all_actors()}

    obj_segmentation = camera_hand.get_uint32_texture("Segmentation")[..., 1]

    assert not np.any(obj_segmentation > 255)
    # Mapping seg_id to obj_id, as a semantic label image.
    seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
    seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
    obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

    # Semantic labels
    sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
    sem_labels_with_color = COLOR_PALETTE[sem_labels]
    sem_image_with_color = im.fromarray(sem_labels_with_color.astype("uint8"))
    sem_image_with_color.save(os.path.join(SAVE_DIR, "label_hand.png"))


    obj_segmentation = camera_base.get_uint32_texture("Segmentation")[..., 1]

    assert not np.any(obj_segmentation > 255)
    # Mapping seg_id to obj_id, as a semantic label image.
    seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
    seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
    obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

    # Semantic labels
    sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
    sem_labels_with_color = COLOR_PALETTE[sem_labels]
    sem_image_with_color = im.fromarray(sem_labels_with_color.astype("uint8"))
    sem_image_with_color.save(os.path.join(SAVE_DIR, "label_base.png"))


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Random placing of a table")
    parser.add_argument(
        "-n",
        dest="scene_name",
        default="0",
        help="id of the scene",
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

args = parse_args()


render_scene(args)
recover_with_kuafu(args)
recover_with_vulkan(args)