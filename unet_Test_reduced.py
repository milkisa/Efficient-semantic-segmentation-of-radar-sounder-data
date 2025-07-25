import os
import glob
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import rasterio
import matplotlib.pyplot as plt
from data_loader import SalObjDataset, ToTensorLab
from model import U2NET, U2NETP


# -----------------------
# Utilities
# -----------------------

def normalize_prediction(d):
    return (d - torch.min(d)) / (torch.max(d) - torch.min(d))


def color_mapping(mask):
    color_map = {
        0: [75, 0, 130],    # dark blue
        1: [70, 130, 180],  # light blue
        2: [60, 179, 113],  # green
        3: [255, 255, 0],   # yellow
        4: [153, 255, 255], # cyan
        5: [153, 0, 0],     # red
    }
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, v in color_map.items():
        rgb[mask == k] = v
    return rgb


def compute_metrics(pred, label):
    pred = np.argmax(pred.cpu().numpy(), axis=0)
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    label_np = label_np.squeeze()
    metrics = {}

    for cls in [1, 2, 3]:
        tp = np.sum((pred == cls) & (label_np == cls))
        fp = np.sum((pred == cls) & (label_np != cls) & (label_np != 0))
        fn = np.sum((pred != cls) & (label_np == cls))
        tn = np.sum((pred != cls) & (label_np != cls) & (label_np != 0))

        metrics[f"recall_{cls}"] = tp / (tp + fn + 1e-10)
        metrics[f"precision_{cls}"] = tp / (tp + fp + 1e-10)
        metrics[f"accuracy_{cls}"] = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    return metrics, pred



def save_visuals(pred, label, out_dir, idx):
    os.makedirs(out_dir, exist_ok=True)
    pred_rgb = color_mapping(pred)
    label_rgb = color_mapping(label.squeeze())

    plt.imsave(os.path.join(out_dir, f"{idx}.png"), pred_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx}_label.png"), label_rgb)


# -----------------------
# Main
# -----------------------

def main():
    print(">>> Starting Inference Pipeline")

    # --- Load raster and label data
    raster = rasterio.open("Data_20101104_06_extended_aligned_dB_target").read()
    label = rasterio.open("Data_20101104_06_extended_aligned_dB_target_rois_for_classification").read()

    rs_image, rs_label = [], []
    for x in range(64, raster.shape[2], 64):
        rs_image.append(raster[:, :, x - 64:x])
        rs_label.append(label[:, :, x - 64:x])

    rs_image = np.array(rs_image).reshape(-1, 410, 64, 1)[298:426]
    rs_label = np.array(rs_label).reshape(-1, 410, 64, 1)[298:426]

    # --- Paths and model selection
    model_name = "u2netp"
    model_path = os.path.join("new_experiment", "more_time", "test_298+",
                              "no_augmentation_400-64", "saved_model",
                              "u2net_bce_itr_200000_train_0.005431_tar_0.000005_ time_15720.988585.pth")
    output_dir = os.path.join("test_data", f"{model_name}_results")

    # --- Dataset and Dataloader
    dataset = SalObjDataset(img_name_list=rs_image, lbl_name_list=rs_label,
                            transform=transforms.Compose([ToTensorLab(flag=0)]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- Load model
    model = U2NETP(1, 4) if model_name == "u2netp" else U2NET(1, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    metrics_log = {k: [] for k in ["recall_1", "recall_2", "recall_3",
                                   "precision_1", "precision_2", "precision_3",
                                   "accuracy_1", "accuracy_2", "accuracy_3"]}

    # --- Inference loop
    for idx, data in enumerate(dataloader, start=298):
        image = data["image"].to(device).float()
        label = data["label"]

        with torch.no_grad():
            d1, *_ = model(image)

        metrics, pred = compute_metrics(d1[0], label[0])
        for k, v in metrics.items():
            metrics_log[k].append(v)

        save_visuals(pred, label[0].numpy(), output_dir, idx)

    # --- Reporting
    print("\n>>> Average Performance Metrics:")
    for k, v in metrics_log.items():
        print(f"{k}: {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
