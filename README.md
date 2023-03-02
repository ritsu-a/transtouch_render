to render a random scene with sapien:
    python render_scene.py -n $i;
raw_meta.pkl and render images are saved in results_render_scene/ 

for a scene that has been rendered, run
    python recover_scene.py -n $i;
to generate depth image of the scene
