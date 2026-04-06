import os
import argparse
import numpy as np
import cv2
import torch

from configs.configs import get_configs_avenue, get_configs_shanghai, get_configs_ucf_crime
from data.test_dataset import AbnormalDatasetGradientsTest
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8


def tensor_rgb_for_display(x):
    """
    x: [C, H, W] tensor in model scale [-1, 1] usually.
    Use first 3 channels only.
    """
    x = x[:3].detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))   # HWC

    # dataset target/frame is normalized as (img - 127.5)/127.5
    # so invert back to [0,255]
    x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return x


def map_to_heatmap(x):
    """
    x: [H, W] tensor/array -> JET heatmap
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.nan_to_num(x.astype(np.float32))
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    x_u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x_u8, cv2.COLORMAP_JET)


def overlay_heatmap(rgb, heatmap, alpha=0.45):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 1 - alpha, heatmap, alpha, 0)


def build_model(args, device):
    if args.dataset == "avenue":
        model = mae_cvt_patch16(
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss
        ).float()
    elif args.dataset in ["shanghai", "ucf_crime"]:
        model = mae_cvt_patch8(
            norm_pix_loss=args.norm_pix_loss,
            img_size=args.input_size,
            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
            abnormal_score_func=args.abnormal_score_func,
            masking_method=args.masking_method,
            grad_weighted_loss=args.grad_weighted_rec_loss
        ).float()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model.to(device)
    return model


def load_student_teacher_weights(model, output_dir, device):
    student_ckpt = torch.load(os.path.join(output_dir, "checkpoint-best-student.pth"), map_location=device)
    teacher_ckpt = torch.load(os.path.join(output_dir, "checkpoint-best.pth"), map_location=device)

    student = student_ckpt["model"]
    teacher = teacher_ckpt["model"]

    for key in student:
        if "student" in key:
            teacher[key] = student[key]

    model.load_state_dict(teacher, strict=False)
    model.eval()
    model.train_TS = True
    return model


def find_sample_index(dataset, wanted_video, wanted_frame):
    """
    Fast lookup using dataset.data directly.
    Works for the npy-based format where dataset.data entries are (video_name, frame_idx).
    """
    wanted_video = str(wanted_video)
    wanted_frame = int(wanted_frame)

    for i, sample_ref in enumerate(dataset.data):
        if isinstance(sample_ref, tuple):
            video_name, frame_idx = sample_ref
            if str(video_name) == wanted_video and int(frame_idx) == wanted_frame:
                return i
        else:
            # old image-file format fallback
            path = sample_ref
            video_name = path.split("/")[-2]
            frame_name = path.split("/")[-1].split(".")[0]
            if str(video_name) == wanted_video and int(frame_name) == wanted_frame:
                return i

    raise ValueError(f"Could not find sample for video={wanted_video}, frame={wanted_frame}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucf_crime")
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Experiment folder containing checkpoint-best.pth and checkpoint-best-student.pth")
    parser.add_argument("--save_dir", type=str, default="recon_vis")
    parser.add_argument("--seed", type=int, default=0)
    args_cli = parser.parse_args()

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    if args_cli.dataset == "avenue":
        args = get_configs_avenue()
    elif args_cli.dataset == "shanghai":
        args = get_configs_shanghai()
    elif args_cli.dataset == "ucf_crime":
        args = get_configs_ucf_crime()
    else:
        raise ValueError(f"Unknown dataset: {args_cli.dataset}")

    args.dataset = args_cli.dataset
    args.output_dir = args_cli.output_dir

    os.makedirs(args_cli.save_dir, exist_ok=True)

    device = args.device
    dataset = AbnormalDatasetGradientsTest(args)
    print("Dataset built")
    print("Finding target sample...")
    idx = find_sample_index(dataset, args_cli.video_name, args_cli.frame_idx)
    print(f"Found sample index: {idx}")

    idx = find_sample_index(dataset, args_cli.video_name, args_cli.frame_idx)
    print(f"Using dataset index {idx}")

    img, grad, target, label, video_name, full_id = dataset[idx]

    samples = torch.tensor(img).unsqueeze(0).to(device)
    grads = torch.tensor(grad).unsqueeze(0).to(device)
    targets = torch.tensor(target).unsqueeze(0).to(device)

    print("Loading selected sample...")
    img, grad, target, label, video_name, full_id = dataset[idx]
    print(f"Loaded sample: {full_id}")

    model = build_model(args, device)
    model = load_student_teacher_weights(model, args.output_dir, device)

    if args.dataset in ["avenue"]:
        model.abnormal_score_func_TS = "L2"
    else:
        model.abnormal_score_func_TS = "L1"

    with torch.no_grad():
        details = model.forward_inference_details(
            samples,
            targets=targets,
            grad_mask=grads,
            mask_ratio=args.mask_ratio,
            do_erosion=True
        )

    teacher_score = float(details["teacher_score"][0].detach().cpu().item())
    ts_score = float(details["ts_score"][0].detach().cpu().item())
    combined_score = teacher_score + ts_score

    target_rgb = tensor_rgb_for_display(details["target_recon"][0])
    teacher_rgb = tensor_rgb_for_display(details["teacher_recon"][0])
    student_rgb = tensor_rgb_for_display(details["student_recon"][0])

    teacher_err_map = details["teacher_img_map"][0, :3].mean(dim=0)   # use RGB channels only
    ts_err_map = details["ts_img_map"][0, :3].mean(dim=0)             # use RGB channels only

    teacher_heatmap = map_to_heatmap(teacher_err_map)
    ts_heatmap = map_to_heatmap(ts_err_map)

    teacher_overlay = overlay_heatmap(target_rgb, teacher_heatmap)
    ts_overlay = overlay_heatmap(target_rgb, ts_heatmap)

    stem = f"{video_name}_{full_id.split('/')[-1]}"

    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_input.png"),
                cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_teacher_recon.png"),
                cv2.cvtColor(teacher_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_student_recon.png"),
                cv2.cvtColor(student_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_teacher_heatmap.png"),
                teacher_heatmap)
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_ts_heatmap.png"),
                ts_heatmap)
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_teacher_overlay.png"),
                teacher_overlay)
    cv2.imwrite(os.path.join(args_cli.save_dir, f"{stem}_ts_overlay.png"),
                ts_overlay)

    np.save(os.path.join(args_cli.save_dir, f"{stem}_teacher_patch_map.npy"),
            details["teacher_patch_map"][0].detach().cpu().numpy())
    np.save(os.path.join(args_cli.save_dir, f"{stem}_ts_patch_map.npy"),
            details["ts_patch_map"][0].detach().cpu().numpy())
    np.save(os.path.join(args_cli.save_dir, f"{stem}_ts_patch_map_processed.npy"),
            details["ts_patch_map_processed"][0].detach().cpu().numpy())

    with open(os.path.join(args_cli.save_dir, f"{stem}_scores.txt"), "w") as f:
        f.write(f"video={video_name}\n")
        f.write(f"frame={full_id.split('/')[-1]}\n")
        f.write(f"label={label}\n")
        f.write(f"teacher_score={teacher_score}\n")
        f.write(f"ts_score={ts_score}\n")
        f.write(f"combined_score={combined_score}\n")

    print(f"Saved outputs to: {args_cli.save_dir}")
    print(f"teacher_score={teacher_score:.6f}")
    print(f"ts_score={ts_score:.6f}")
    print(f"combined_score={combined_score:.6f}")


if __name__ == "__main__":
    main()