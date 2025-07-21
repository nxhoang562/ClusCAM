from utils_folder import load_image, basic_visualize, list_image_paths
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
    AblationCAM, ShapleyCAM
)

from cam.Cluscam import ClusterScoreCAM
from cam.randomcam import RandomCam

from cam.polycam import PCAMp, PCAMm, PCAMpm
from cam.recipro_cam import ReciproCam
from cam.opticam import Basic_OptCAM


CAM_FACTORY = {
    "gradcam": lambda md, **kw: GradCAM(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    "gradcamplusplus": lambda md, **kw: GradCAMPlusPlus(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    "layercam": lambda md, **kw: LayerCAM(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    "scorecam": lambda md, **kw: ScoreCAM(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    "ablationcam": lambda md, **kw: AblationCAM(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    "shapleycam": lambda md, **kw: ShapleyCAM(
        model=md["arch"],
        target_layers=[md["target_layer"]],
        **kw
    ),
    
    "cluster": lambda md, num_clusters=None: ClusterScoreCAM(
        md,
        num_clusters=num_clusters,
        zero_ratio=md.get("zero_ratio", 0.5),
        temperature=md.get("temperature", 0.5)
    ),
     "polyp": lambda md, **kw: PCAMp(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    
    "polym": lambda md, **kw: PCAMm(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    
    "polypm": lambda md, **kw: PCAMpm(
        model=md["arch"],
        target_layer_list=md.get("target_layer_list"),
        batch_size=md.get("batch_size", 32),
        intermediate_maps=md.get("intermediate_maps", False),
        lnorm=md.get("lnorm", True)
    ),
    
     "opticam": lambda md, **kw: Basic_OptCAM(
        model=md["arch"],
        device=kw.get("device"),
        target_layer=md["target_layer"],
        max_iter=md.get("max_iter", 50),
        learning_rate=md.get("learning_rate", 0.1),
        name_f=md.get("name_f", "logit_predict"),
        name_loss=md.get("name_loss", "norm"),
        name_norm=md.get("name_norm", "max_min"),
        name_mode=md.get("name_mode", "resnet")
    ),
     
     "reciprocam": lambda md, **kw: ReciproCam(
        model=md["arch"],
        device=kw.get("device"),
        target_layer_name=md.get("target_layer_name", None)
    )
    
}