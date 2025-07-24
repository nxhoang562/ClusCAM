import os
import torch
import pandas as pd
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import ClassifierOutputReST
from utils_folder import load_image, basic_visualize, list_image_paths
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
    AblationCAM, ShapleyCAM
)
from cam.Cluscam import ClusterScoreCAM

from pytorch_grad_cam.ablation_layer import AblationLayerVit


from cam.polycam import PCAMp, PCAMm, PCAMpm
from cam.recipro_cam import ReciproCam
from cam.opticam import Basic_OptCAM

from torchvision.models import VisionTransformer

from metrics import (
    AverageDrop, 
    AverageIncrease, 
    Coherency, 
    Complexity,
    deletion_curve,
    DeletionCurveAUC,
    Infidelity,
    insertion_curve,
    InsertionCurveAUC,
    Sensitivity,  
    AverageConfidence,
)



from torchvision import transforms
from torchvision.models import VGG
import torch.nn as nn
import torch.nn.functional as F


rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def get_transform_for_model(model):
    """
    Tự động chọn pipeline transform:
    - Nếu conv đầu tiên chỉ nhận 1 channel (grayscale), dùng gray_transform
    - Ngược lại dùng rgb_transform
    """
    if hasattr(model, "conv1"):
        in_ch = model.conv1.in_channels
    elif isinstance(model, VGG):
        first_conv = next(
            (m for m in model.features if isinstance(m, nn.Conv2d)),
            None
        )
        in_ch = first_conv.in_channels if first_conv is not None else 3
    else:
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        in_ch = conv_layers[0].in_channels if conv_layers else 3
    return gray_transform if in_ch == 1 else rgb_transform


def predict_top1_indices(image_paths, model, device):
    model = model.to(device).eval()
    transform = get_transform_for_model(model)
    top1 = []
    with torch.no_grad():
        for path in image_paths:
            img = load_image(path)
            inp = transform(img).unsqueeze(0).to(device)
            logits = model(inp)
            top1.append(logits.argmax(dim=1).item())
    return top1

# Mapping giữa tên method và hàm khởi tạo tương ứng
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
        target_layer=[md["target_layer"]],
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

def batch_test(
    model,
    model_dict,
    dataset,
    excel_path,
    k_values,
    cam_method="cluster",
    top_n=None,
    start_idx=None,
    end_idx=None,
    model_name=None
):
    if model_name is None:
        raise ValueError("batch_test: cần truyền model_name (chuỗi)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    transform = get_transform_for_model(model)

    all_paths = list_image_paths(dataset)
    if not all_paths:
        raise RuntimeError(f"No images found in {dataset}")
    
 # Lấy ảnh theo start/end nếu có, ngược lại dùng top_n hoặc toàn bộ
    if start_idx is not None and end_idx is not None:
        image_paths = all_paths[start_idx:end_idx]
    else:
        image_paths = all_paths if top_n is None else all_paths[:top_n]
    top1_idxs = predict_top1_indices(image_paths, model, device)
    if os.path.isdir(excel_path):
        excel_dir = excel_path
        excel_filename = "results.xlsx"
    else:
        excel_dir = os.path.dirname(excel_path) or '.'
        excel_filename = os.path.basename(excel_path)
    model_dir = os.path.join(excel_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, excel_filename)
    ks = k_values if cam_method == "cluster" else [None]
    for c in ks:
        info = f"method={cam_method}" + (f", K={c}" if c else "")
        print(f"\n=== Testing {info} ===")
        drops, incs, del_aucs, curves_records, infids = [], [], [], [], []
        ins_aucs = []
        ins_curves_records = []
        cfds = []
        # senss = []
    
        if cam_method in ("polyp","polym","polypm") \
        and model_dict.get("target_layer_list") is None \
        and isinstance(model, VisionTransformer):
            # hook sau self-attention của block cuối
            model_dict["target_layer_list"] = ["conv_proj"]
        
        if cam_method in ("reciprocam") \
        and model_dict.get("target_layer_list") is None \
        and isinstance(model, VisionTransformer):
            model_dict["target_layer_list"] = model.encoder.layers[-1].ln_2
            
        if cam_method == "cluster":
            cam = CAM_FACTORY["cluster"](model_dict, num_clusters=c)
        elif cam_method == "reciprocam":
            cam = CAM_FACTORY["reciprocam"](model_dict)
            cam.model    = cam.model.to(device)       # weights lên GPU nếu device là 'cuda'
            cam.gaussian = cam.gaussian.to(device)    # filter lên GPU
            cam.device   = device    
        # elif cam_method == "ablationcam":
        #     cam = CAM_FACTORY["ablationcam"](
        #         model_dict, 
        #         ablation_layer = AblationLayerVit(),
        #     )
        else:
            cam = CAM_FACTORY[cam_method](
                model_dict,
            )

        
        for idx, (path, cls) in enumerate(zip(image_paths, top1_idxs), 1):
            print(f"[{idx}/{len(image_paths)}] {os.path.basename(path)} -> class {cls}")
            img = load_image(path)
            img_tensor = transform(img).unsqueeze(0).to(device)
            if cam_method == "cluster":
                sal_map = cam(img_tensor, class_idx=cls).cpu().squeeze(0)
            elif cam_method in ["polyp", "polym", "polypm"]:
                out = cam(img_tensor, class_idx=cls)   # out là list hoặc tensor
                # nếu là list thì lấy phần tử cuối
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                # bây giờ out có thể là numpy array hoặc torch.Tensor
                if isinstance(out, np.ndarray):
                    sal_map = torch.from_numpy(out)
                else:
                    sal_map = out
                sal_map = sal_map.cpu().squeeze(0)
                
            elif cam_method == "opticam":
                label_tensor = torch.tensor([cls], device=device)
                norm_map, _ = cam(img_tensor, label_tensor)
                sal_map = norm_map.cpu().squeeze(0)
            
            elif cam_method == "reciprocam":
                out_cam, _ = cam(img_tensor, index=cls)
                # Nếu out_cam là numpy array thì convert, rồi đưa lên CPU
                sal_map = (
                    torch.from_numpy(out_cam)
                    if isinstance(out_cam, np.ndarray)
                    else out_cam
                ).cpu().squeeze(0)
            elif cam_method == "shapleycam":
                # ShapleyCAM trả về numpy array, cần chuyển sang tensor
                saliency_np = cam(input_tensor=img_tensor, targets=[ClassifierOutputReST(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)
            else:
                saliency_np = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(cls)])
                sal_map = torch.from_numpy(saliency_np).cpu().squeeze(0)
            sal3 = sal_map.unsqueeze(0).repeat(1, img_tensor.size(1), 1, 1).to(device)
            
            # Nếu sal3 không có kích thước (C, H, W) thì cần reshape (check cho reciprocam)
            if sal3.shape[-2:] != img_tensor.shape[-2:]:
                sal3 = F.interpolate(
                    sal3,
                    size=img_tensor.shape[-2:],      # (224, 224)
                    mode='bilinear',
                    align_corners=False
                )
            
            
            drop = AverageDrop()(model, img_tensor, sal3, cls, device, True)
            inc = AverageIncrease()(model, img_tensor, sal3, cls, device, True)
            cfd = AverageConfidence()(model, img_tensor, sal3, cls, device, True)
               
            
        # # Với các phương pháp polyCAM, cần adapter để Coherency gọi đúng signature
        #     if cam_method in ("polyp", "polym", "polypm"):
        #         def cam_adapter(input_tensor, targets=None):
        #             out = cam(input_tensor, class_idx=cls)
        #             # nếu out là list/tuple thì lấy phần tử cuối (bước refinement cuối cùng)
        #             if isinstance(out, (list, tuple)):
        #                 o = out[-1]
        #             else:
        #                 o = out
        #             # nếu là numpy array thì convert sang tensor
        #             if isinstance(o, np.ndarray):
        #                 o = torch.from_numpy(o)
        #             # đảm bảo đúng device và shape [1,C,H,W] hoặc [C,H,W]
        #             return o.to(device)
        #         attr_fn = cam_adapter
        #     elif cam_method == "reciprocam":
        #         def cam_adapter(input_tensor, targets=None):
        #             # Bỏ qua `targets`, dùng trực tiếp cls
        #             out_cam, _ = cam(input_tensor, index=cls)
        #             if isinstance(out_cam, np.ndarray):
        #                 out_cam = torch.from_numpy(out_cam)
        #             # Nếu out_cam chỉ là [Hf, Wf], chuyển thành [1, Hf, Wf]
        #             if out_cam.ndim == 2:
        #                 out_cam = out_cam.unsqueeze(0)
        #             return out_cam.to(device)
        #         attr_fn = cam_adapter
        #     else:
        #         attr_fn = cam

        #     coher = Coherency()(
        #         model=model,
        #         test_images=img_tensor,
        #         saliency_maps=sal3,
        #         class_idx=cls,
        #         attribution_method=attr_fn,
        #         upsample_method=lambda attribution, image, device, model, layer: 
        #             F.interpolate(attribution, size=image.shape[-2:], mode='bilinear', align_corners=False),
        #         return_mean=True,
        #         device=device
        #     )
            # coher = Coherency()(
            #     model=model,
            #     test_images=img_tensor,
            #     saliency_maps=sal3,
            #     class_idx=cls,
            #     attribution_method=cam,
            #     upsample_method=lambda attribution, image, device, model, layer: 
            #         F.interpolate(attribution, size=image.shape[-2:], mode='bilinear', align_corners=False),
            #     return_mean=True,
            #     device=device
            # )

            # comp = Complexity()(sal3, return_mean=True)
            
            # adcc = 3 / ((1/coher) + 1/(1-comp) + 1/(1 - drop/100))
            drops.append(drop)
            incs.append(inc)
            cfds.append(cfd)
            # cohers.append(coher*100)  # Chuyển sang phần trăm
            # comps.append(comp*100)  # Chuyển sang phần trăm
            # adccs.append(adcc*100)  # Chuyển sang phần trăm
        
        ##========================================================##
            # sal_single = sal3.mean(dim=1, keepdim=True)
            # infid = Infidelity()(
            # model=model,
            # test_images=img_tensor,          # [1, C, H, W]
            # saliency_maps=sal_single,        # [1, 1, H, W]
            # class_idx=torch.tensor([cls], device=device),
            # device=device,
            # return_mean=True
            # )
            # infids.append(infid)
            
        ##=======Tính Sensitivity=========================================================================#
        
        # # --- adapter cho PolyCAM ---
        #     if cam_method in ("polyp", "polym", "polypm"):
        #         def polycam_attr_fn(input_tensor, targets=None, **kwargs):
        #             # targets có thể là None, list[int], list[Tensor], list[ClassifierOutputTarget]
        #             if targets is None:
        #                 idx = cls
        #             else:
        #                 first = targets[0]
        #                 if isinstance(first, ClassifierOutputTarget):
        #                     idx = first.category
        #                 elif isinstance(first, torch.Tensor):
        #                     idx = first.item()
        #                 else:
        #                     idx = int(first)
        #             out = cam(input_tensor, class_idx=idx)
        #             # nếu trả về list/tuple thì lấy phần tử cuối cùng
        #             if isinstance(out, (list, tuple)):
        #                 out = out[-1]
        #             # nếu là numpy array thì convert sang tensor
        #             if isinstance(out, np.ndarray):
        #                 out = torch.from_numpy(out)
        #             # đảm bảo shape [1, C, H, W]
        #             if out.ndim == 3:
        #                 out = out.unsqueeze(0)
        #             return out.to(device)

        #         attribution_fn = polycam_attr_fn
                
        #     elif cam_method == "opticam":
        #         def opticam_attr_fn(input_tensor, targets=None, **kwargs):
        #             # Lấy chỉ số class từ targets
        #             if targets is None:
        #                 idx = cls
        #             else:
        #                 first = targets[0]
        #                 if isinstance(first, ClassifierOutputTarget):
        #                     idx = first.category
        #                 elif isinstance(first, torch.Tensor):
        #                     idx = first.item()
        #                 else:
        #                     idx = int(first)
        #             # Tạo label_tensor đúng shape [1]
        #             label_tensor = torch.tensor([idx], device=device)
        #             # Gọi OptiCAM: trả về (norm_map, loss) hoặc tương tự
        #             norm_map, _ = cam(input_tensor, label_tensor)
        #             # Đưa về CPU, thành tensor, giữ shape [1, C, H, W]
        #             sal = norm_map.cpu()
        #             if sal.ndim == 3:
        #                 sal = sal.unsqueeze(0)
        #             return sal.to(device)

        #         attribution_fn = opticam_attr_fn
                
        #     elif cam_method == "reciprocam":
        #         def reciprocam_attr_fn(input_tensor, targets=None, **kwargs):
        #             # 1) Lấy class index từ targets (ClassifierOutputTarget, Tensor, hay int)
        #             if targets is None:
        #                 idx = cls
        #             else:
        #                 first = targets[0]
        #                 if isinstance(first, ClassifierOutputTarget):
        #                     idx = first.category
        #                 elif isinstance(first, torch.Tensor):
        #                     idx = first.item()
        #                 else:
        #                     idx = int(first)
        #             # 2) Gọi ReciproCam với đúng tham số index
        #             out_cam, _ = cam(input_tensor, index=idx)
        #             # 3) Chuyển sang tensor nếu cần
        #             if isinstance(out_cam, np.ndarray):
        #                 out_cam = torch.from_numpy(out_cam)
        #             # 4) Đảm bảo shape [1, C, H, W]; giả sử out_cam là [H, W]
        #             if out_cam.ndim == 2:
        #                 out_cam = out_cam.unsqueeze(0).unsqueeze(0)  # -> [1,1,H,W]
        #             elif out_cam.ndim == 3:
        #                 out_cam = out_cam.unsqueeze(0)               # -> [1,C,H,W]
        #             return out_cam.to(device)

        #         attribution_fn = reciprocam_attr_fn
        #     else:
        #         # wrapper chung cho các CAM khác
        #         def generic_attr_fn(input_tensor, targets=None, **kw):
        #             out = cam(input_tensor, targets=targets)
        #             # torch.as_tensor sẽ phụ hợp nếu out là numpy array hay tensor
        #             return torch.as_tensor(out).to(device)

        #         attribution_fn = generic_attr_fn

        #     if cam_method == "shapleycam" or cam_method == "cluster":
        #         senss.append(0)
        #     else:
        #         sens_val = Sensitivity()(
        #                                 model=model,
        #                                 test_images=img_tensor,
        #                                 class_idx=torch.tensor([cls], device=device),
        #                                 attribution_method=attribution_fn,
        #                                 device=device,
        #                                 return_mean=True,
        #                                 layer=model_dict["target_layer"],
        #                                 baseline_dist=torch.distributions.Normal(0, 0.1)
        #                             )
        #         senss.append(sens_val)   
        
        ##=======Tính deletion_curve==============================================================================##
            # Lấy deletion_curve chi tiết
        #     with torch.no_grad():
        #         pixel_removed_perc, confidences = deletion_curve(
        #             model,
        #             img_tensor,
        #             sal_single,                   # shape [1,1,H,W]
        #             torch.tensor([cls], device=device),
        #             device=device,
        #             apply_softmax=True,
        #             num_points=30
        #         )
        #     pr = pixel_removed_perc[0].tolist()
        #     cf = confidences[0].tolist()
        #     for point_idx, (p, c_val) in enumerate(zip(pr, cf)):
        #         curves_records.append({
        #             "image_path": os.path.basename(path),
        #             "point_idx": point_idx,
        #             "pixel_removed_perc": p,
        #             "confidence": c_val
        #         })
            
        #     # Tính deletion AUC để summary
        #     del_auc = DeletionCurveAUC()(
        #         model=model,
        #         test_images=img_tensor,
        #         saliency_maps=sal_single,
        #         class_idx=torch.tensor([cls], device=device),
        #         attribution_method=None,
        #         device=device,
        #         apply_softmax=True, 
        #         return_mean=True # False để trả về tensor AUC cho từng ảnh, nhưng do chạy với từng ảnh nên True vẫn đúng
        #     )

        #     del_aucs.append(del_auc)
        # #=======Tính insertion_curve=======#
        #     with torch.no_grad():
        #         ins_range, ins_vals = insertion_curve(
        #                                             model,
        #                                             img_tensor,
        #                                             sal_single,                   # [1,1,H,W]
        #                                             torch.tensor([cls], device=device),
        #                                             device=device,
        #                                             apply_softmax=True,
        #                                             num_points=30
        #                                         )
        #     pr_ins = ins_range[0].tolist()
        #     cf_ins = ins_vals[0].tolist()
        #     for point_idx, (p, c_val) in enumerate(zip(pr_ins, cf_ins)):
        #         ins_curves_records.append({
        #             "image_path": os.path.basename(path),
        #             "point_idx": point_idx,
        #             "pixel_restored_perc": p,
        #             "confidence": c_val
        #         })
            
        #     ins_auc = InsertionCurveAUC()(    
        #                                 model=model,
        #                                 test_images=img_tensor,
        #                                 saliency_maps=sal_single,
        #                                 class_idx=torch.tensor([cls], device=device),
        #                                 attribution_method=None,
        #                                 device=device,
        #                                 apply_softmax=True,
        #                                 return_mean=True
        #                                 )
        #     ins_aucs.append(ins_auc)
            
       
        df = pd.DataFrame({
            "image_path": image_paths,
            "top1_index": top1_idxs,
            "average_drop": drops,
            "increase_confidence": incs,
            # "deletion_auc": del_aucs,
            # "insertion_auc": ins_aucs,
            # "infidelity": infids,
            "average_confidence": cfds,
            
            
        })
        avg_row = pd.DataFrame([{
            "image_path": "AVERAGE",
            "top1_index": "",
            "average_drop": np.mean(drops),
            "increase_confidence": np.mean(incs),
            # "deletion_auc": np.mean(del_aucs),
            # "insertion_auc": np.mean(ins_aucs),
            # "infidelity": np.mean(infids),
            "average_confidence": np.mean(cfds),
            
        }])
        df = pd.concat([avg_row, df], ignore_index=True)
        
       
        # Lưu vào file Excel
        sheet_name = cam_method if c is None else f"{cam_method}_K{c}"
        mode = "a" if os.path.exists(full_path) else "w"
        with pd.ExcelWriter(full_path, engine="openpyxl", mode=mode, if_sheet_exists=("replace" if mode=="a" else None)) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved sheet {sheet_name} in {full_path}")
        
        #=================================Lưu các deletion curves và insetion curves vao another file Excel==========================================================================#
        
        # # 1. Chuyển thành DataFrame và rename để rõ ràng
        # df_del = pd.DataFrame(curves_records).rename(
        #     columns={
        #         "pixel_removed_perc": "pixel_removed_perc",
        #         "confidence":         "deletion_confidence"
        #     }
        # )
        # df_ins = pd.DataFrame(ins_curves_records).rename(
        #     columns={
        #         "pixel_restored_perc": "pixel_restored_perc",
        #         "confidence":          "insertion_confidence"
        #     }
        # )

        # # 2. Merge theo image_path và point_idx
        # df_curves = pd.merge(
        #     df_del,
        #     df_ins,
        #     on=["image_path", "point_idx"],
        #     how="outer"   # nếu bạn muốn giữ cả những điểm có trong 1 trong 2
        # )

        # # 3. Ghi vào sheet method duy nhất trong file *_deletion_curves.xlsx
        # root, ext = os.path.splitext(full_path)
        # curves_path = f"{root}_deletion_curves{ext}"
        
        # curve_mode = "a" if os.path.exists(curves_path) else "w"
        # with pd.ExcelWriter(curves_path, engine="openpyxl", mode=curve_mode, if_sheet_exists="replace" if curve_mode == "a" else None) as writer:
        #     df_curves.to_excel(writer, sheet_name=sheet_name, index=False)

        # print(f"Saved combined curves sheet {sheet_name} in {curves_path}")

