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
from PIL import Image


OBJECT_DIR=f"/share/pengyang/render/models_v2/models/"
TAR_DIR = f"/share/pengyang/render/results_recover_scene/"
os.makedirs(TAR_DIR, exist_ok=True)


mats = [ 
    np.array(
         [[-0.37124701, 0.92851306,-0.00625727,-0.0551394 ],
        [ 0.76209405, 0.30084409,-0.57332844, 0.06398138],
        [-0.53046048,-0.2176151 ,-0.81930174, 0.89167432],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
         [[-0.99998931,-0.00324778,-0.00328973,-0.08225272],
        [-0.00332778, 0.99969158, 0.02461047, 0.01168893],
        [ 0.00320878, 0.02462115,-0.9996917 , 0.65327796],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.96552673, 0.0016328 ,-0.26029879,-0.0992079 ],
        [ 0.01084106, 0.99936494,-0.03394392, 0.02497603],
        [ 0.26007806,-0.03559568,-0.96493127, 0.73610967],
        [ 0.        , 0.        , 0.        , 1.        ]] 
    ),
    np.array(
        [[-0.91047621,-0.0039816 ,-0.41354229,-0.11267854],
        [ 0.00614651, 0.99971293,-0.02315776, 0.0123658 ],
        [ 0.41351578,-0.02362643,-0.91019036, 0.83408164],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.9966869 ,-0.07076897, 0.04008722, 0.15545513],
        [-0.04062716,-0.860163  ,-0.50839852, 0.01128133],
        [ 0.07046038, 0.50508551,-0.86018833, 0.68552918],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.99921762, 0.01865731, 0.03487182, 0.16389475],
        [ 0.0385495 ,-0.65646058,-0.7533747 , 0.09506896],
        [ 0.00883603, 0.75412957,-0.65666622, 0.5718026 ],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.97388937,-0.11635256, 0.19493993, 0.11657524],
        [-0.00650754,-0.87263339,-0.48833249,-0.02666083],
        [ 0.22692982, 0.47431325,-0.85060555, 0.78473557],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.92409815,-0.34492929, 0.16451866, 0.1593473 ],
        [-0.21346939,-0.82298878,-0.52642215,-0.03297668],
        [ 0.31697543, 0.45134604,-0.83415426, 0.79409655],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.83635877,-0.14034378, 0.52991285,-0.1200275 ],
        [ 0.01831881,-0.95897711,-0.282891  ,-0.063676  ],
        [ 0.54787628, 0.24630574,-0.79947799, 0.95186297],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.94577329,-0.30461135, 0.11280429, 0.28816027],
        [-0.13192649,-0.6775559 ,-0.72354226,-0.00720545],
        [ 0.2968304 , 0.66942507,-0.68100058, 0.77878103],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.22010547,-0.94497881, 0.24200956,-0.15338285],
        [-0.58499709,-0.32640144,-0.74245572,-0.0382464 ],
        [ 0.7805972 , 0.02184368,-0.6246526 , 0.76865528],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[ 0.69836402,-0.62343468, 0.35159193, 0.22936132],
        [-0.61053406,-0.77525263,-0.16196149,-0.02932941],
        [ 0.37354498,-0.10155077,-0.92203665, 0.84731143],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.52480605,-0.85042995, 0.03670835,-0.12787397],
        [-0.61962715, 0.35209659,-0.7014914 ,-0.03587023],
        [ 0.58364441,-0.39089242,-0.71173192, 0.83185788],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.80162832,-0.58808873,-0.10744149,-0.09833329],
        [-0.48614853, 0.74586405,-0.45535308, 0.01291785],
        [ 0.34792476,-0.31279141,-0.88380422, 0.71794198],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.96780038, 0.08482933,-0.23699452, 0.03684804],
        [ 0.18603171, 0.87528872,-0.44638757, 0.08566153],
        [ 0.16957188,-0.47610256,-0.8628857 , 0.68990929],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.9745704 ,-0.03632878,-0.22111707,-0.11708072],
        [ 0.08320692, 0.85755033,-0.50762589,-0.00598749],
        [ 0.20806045,-0.51311563,-0.83272036, 0.7850985 ],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.95981746,-0.07011203,-0.27172549,-0.10886933],
        [ 0.09713041, 0.82543539,-0.55607742, 0.0757852 ],
        [ 0.26327955,-0.56012563,-0.78545729, 0.61216671],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
    np.array(
        [[-0.94717368,-0.10010322,-0.30469881, 0.00815399],
        [ 0.10551362, 0.79989841,-0.59078711, 0.07236462],
        [ 0.30286778,-0.59172788,-0.74708047, 0.56630189],
        [ 0.        , 0.        , 0.        , 1.        ]]
    ),
]

rgb_base_intrinsic = np.array(
    [[ 9.1020077777522272e+02, 0., 6.5907945440859100e+02,],
    [ 0.,9.1038247935099866e+02, 3.7481157428344886e+02,],
    [ 0., 0., 1. ]]
)

rgb_hand_intrinsic = np.array(
    [   [ 8.9687159287313307e+02, 0., 6.4037813319745328e+02,],
        [0.,8.9685095756783505e+02, 3.7131751630652178e+02,],
        [ 0., 0., 1. ]]
    )

irL_intrinsic = np.array(
    [ [9.0039466537178293e+02, 0., 6.3968492136132227e+02,],
        [0.,9.0123293413383715e+02, 3.6041824118552393e+02,],
        [ 0., 0., 1.,], ]
    )
irR_intrinsic = np.array(
    [ [8.9619520685157840e+02, 0., 6.3966703424574928e+02,],
        [0.,8.9728163687761082e+02, 3.6040539833080373e+02,],
        [ 0., 0., 1. ]]
    )



irL_pose = np.array(
        [[ 9.99997528e-01,-2.14602136e-03, 5.81672868e-04,-1.46269827e-02],
        [ 2.14563291e-03, 9.99997482e-01, 6.54951402e-04, 4.26302826e-03],
        [-5.83082095e-04,-6.53695987e-04, 9.99999620e-01, 1.79415897e-03],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
      )

irR_pose = np.array(
    [[ 9.99995886e-01,-1.99966652e-03, 2.05743511e-03,-7.04094714e-02],
    [ 1.99799643e-03, 9.99997672e-01, 8.12560354e-04, 4.31426008e-03],
    [-2.05905955e-03,-8.08436561e-04, 9.99997557e-01,-6.54027528e-04],
    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)
# irR_pose = np.linalg.inv(np.array(
#         [[ 0.99999706, -0.00185561,  0.00155533, -0.01436345,],
#         [ 0.00185144,  0.99999469,  0.00268054,  0.00372648,],
#         [-0.00156029, -0.00267764,  0.99999519, -0.00466651,],
#         [ 0.,          0.,          0.,          1.,        ]]
#       )
# )

def parse_args():
    parser = argparse.ArgumentParser(description="Render depth of all views")
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

def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth



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


def recover_scene(args):
    

    sim = sapien.Engine()
    sim.set_log_level('warning')
    sapien.KuafuRenderer.set_log_level('warning')
    
    # render_config = sapien.KuafuConfig()
    # render_config.use_viewer = False
    # render_config.use_denoiser = False
    # render_config.spp = 128
    # render_config.max_bounces = 8
    # renderer = sapien.KuafuRenderer(render_config)

    renderer = sapien.VulkanRenderer(offscreen_only=False)
    sim.set_renderer(renderer)

    
    with open("/share/pengyang/render/results_render_scene/" + str(args.scene_name) + "/raw_meta.pkl", "rb") as file:
        meta = pickle.load(file)

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
    table = builder.build_kinematic(name="table")
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
    
    seg_id_to_sapien_name = {actor.get_id(): actor.get_name() for actor in scene.get_all_actors()}


    for view in range(len(mats)):

        camera_mount = scene.create_actor_builder().build_kinematic()
        camera_mount_l = scene.create_actor_builder().build_kinematic()
        camera_mount_r = scene.create_actor_builder().build_kinematic()
        camera_rgb = scene.add_mounted_camera(
            name=f"camera_rgb_{view}",
            actor=camera_mount,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=1280,
            height=720,
            fovy=np.deg2rad(30),
            near=0.1,
            far=100,
        )
        camera_irL = scene.add_mounted_camera(
            name=f"camera_irL_{view}",
            actor=camera_mount_l,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=1280,
            height=720,
            fovy=np.deg2rad(30),
            near=0.1,
            far=100,
        )
        camera_irR = scene.add_mounted_camera(
            name=f"camera_irR_{view}",
            actor=camera_mount_r,
            pose=sapien.Pose(),  # relative to the mounted actor
            width=1280,
            height=720,
            fovy=np.deg2rad(30),
            near=0.1,
            far=100,
        )


        
        

        SCENE_NAME = meta["scene_name"]
        SAVE_DIR = TAR_DIR + str(SCENE_NAME) +  "-" + str(view) + "/"
        # SAVE_DIR = "/share/sim2real_tactile/real/shareset/" + str(SCENE_NAME) +  "-" + str(view + 1) + "/"
        os.makedirs(SAVE_DIR, exist_ok=True)
        

        camera_mount.set_pose(cv2ex2pose(mats[view]))
        camera_mount_l.set_pose(cv2ex2pose(irL_pose @ mats[view]))
        camera_mount_r.set_pose(cv2ex2pose(irR_pose @ mats[view]))

        if view == 0:
            camera_rgb.set_perspective_parameters(0.1, 100,rgb_base_intrinsic[0, 0],rgb_base_intrinsic[1, 1],rgb_base_intrinsic[0, 2],rgb_base_intrinsic[1, 2],0)
        else:
            camera_rgb.set_perspective_parameters(0.1, 100,rgb_hand_intrinsic[0, 0],rgb_hand_intrinsic[1, 1],rgb_hand_intrinsic[0, 2],rgb_hand_intrinsic[1, 2],0)
        camera_irL.set_perspective_parameters(0.1, 100,irL_intrinsic[0, 0],irL_intrinsic[1, 1],irL_intrinsic[0, 2],irL_intrinsic[1, 2],0)
        camera_irR.set_perspective_parameters(0.1, 100,irR_intrinsic[0, 0],irR_intrinsic[1, 1],irR_intrinsic[0, 2],irR_intrinsic[1, 2],0)

        
        # scene.step()
        scene.update_render()
        camera_rgb.take_picture()
        camera_irL.take_picture()
        camera_irR.take_picture()


        # p_irL = camera_irL.get_color_rgba()
        # p_irL = (p_irL[..., :3] * 255).clip(0, 255).astype(np.uint8)
        # p_irL = cv2.cvtColor(p_irL, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(SAVE_DIR + 'irL.png', p_irL)
        # p_irR = camera_irR.get_color_rgba()
        # p_irR = (p_irR[..., :3] * 255).clip(0, 255).astype(np.uint8)
        # p_irR = cv2.cvtColor(p_irR, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(SAVE_DIR + 'irR.png', p_irR)
        p_rgb = camera_rgb.get_color_rgba()
        p_rgb = (p_rgb[..., :3] * 255).clip(0, 255).astype(np.uint8)
        p_rgb = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(SAVE_DIR + 'vulkan_rgb.png', p_rgb)

        obj_segmentation = camera_rgb.get_uint32_texture("Segmentation")[..., 1]

        assert not np.any(obj_segmentation > 255)

        # Mapping seg_id to obj_id, as a semantic label image.
        seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
        seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
        obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

        # Semantic labels
        sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
        sem_labels_with_color = COLOR_PALETTE[sem_labels]
        sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
        sem_image_with_color.save(os.path.join(SAVE_DIR, "label.png"))

        








        obj_segmentation = camera_irL.get_uint32_texture("Segmentation")[..., 1]

        assert not np.any(obj_segmentation > 255)

        # Mapping seg_id to obj_id, as a semantic label image.
        seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
        seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
        obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

        # Semantic labels
        sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
        sem_labels_with_color = COLOR_PALETTE[sem_labels]
        sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
        sem_image_with_color.save(os.path.join(SAVE_DIR, "label_irL.png"))
        
        obj_segmentation = camera_irR.get_uint32_texture("Segmentation")[..., 1]

        assert not np.any(obj_segmentation > 255)

        # Mapping seg_id to obj_id, as a semantic label image.
        seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
        seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
        obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

        # Semantic labels
        sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
        sem_labels_with_color = COLOR_PALETTE[sem_labels]
        sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
        sem_image_with_color.save(os.path.join(SAVE_DIR, "label_irR.png"))


        obj_segmentation = camera_irL.get_uint32_texture("Segmentation")[..., 1]

        assert not np.any(obj_segmentation > 255)

        # Mapping seg_id to obj_id, as a semantic label image.
        seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
        seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
        obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

        # Semantic labels
        sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
        sem_image = Image.fromarray(sem_labels)
        sem_image.save(os.path.join(SAVE_DIR, "labelL.png"))

        sem_labels_with_color = COLOR_PALETTE[sem_labels]
        sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
        sem_image_with_color.save(os.path.join(SAVE_DIR, "labelL2.png"))





        normal = camera_irL.get_float_texture("Normal")
        pos = camera_irL.get_float_texture("Position")
        depth = -pos[..., 2]
        mask = np.zeros_like(depth).astype(bool)

        table_mask = ~(depth == 0)
        
        depth1 = np.round(depth - 0.0005, 3)

        depth = ((depth + 0.0005) * 1000).astype(np.uint16)

        cv2.imwrite(os.path.join(SAVE_DIR, f"depthL.png"), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(SAVE_DIR, f"depthL_colored.png"), vis_depth)

        


        meta["extrinsic_l"] = irL_pose @ mats[view]
        meta["extrinsic"] = mats[view]
        meta["extrinsic_r"] = irR_pose @ mats[view]
        meta["intrinsic_l"] = camera_irL.get_camera_matrix()
        meta["intrinsic"] = camera_rgb.get_camera_matrix()
        meta["intrinsic_r"] = camera_irR.get_camera_matrix()


        with open(SAVE_DIR + "meta.pkl", "wb") as file:
            pickle.dump(meta, file)



args = parse_args()
recover_scene(args)
