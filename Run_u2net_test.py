import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import rasterio
import matplotlib.pyplot as plt
from data_loader import SalObjDataset, ToTensorLab
from model import U2NET, U2NETP
from implementation.metrics import calc_metrics  # Assuming your metric code is in implementation/metrics.py


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
    model_path = os.path.join("saved_models", "no_augmentation",
                              "u2net_bce_itr_1200000_train_0.005313_tar_0.000001_ time_94105.246779.pth")
    output_dir = os.path.join("test_data", f"{model_name}_results")
    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset and Dataloader
    dataset = SalObjDataset(img_name_list=rs_image, lbl_name_list=rs_label,
                            transform=transforms.Compose([ToTensorLab(flag=0)]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- Load model
    model = U2NETP(1, 4) if model_name == "u2netp" else U2NET(1, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    rs_pred = []
    rs_lab = []

    # --- Inference loop
    for idx, data in enumerate(dataloader, start=298):
        image = data["image"].to(device).float()
        label = data["label"]

        with torch.no_grad():
            d1, *_ = model(image)

        # Get predicted class map
        pred_np = np.argmax(d1[0].cpu().numpy(), axis=0)
        label_np = label[0].cpu().numpy().squeeze()

        # Save predictions and labels for metric calculation
        rs_pred.append(pred_np)
        rs_lab.append(label_np)

        # Optional: Save visualization
        # rgb = color_mapping(pred_np)
        # label_rgb = color_mapping(label_np)
        # plt.imsave(os.path.join(output_dir, f"{idx}.png"), rgb)
        # plt.imsave(os.path.join(output_dir, f"{idx}_label.png"), label_rgb)

        del d1

    # --- Metric computation using sklearn
    avg_recall, avg_precision, avg_accuracy = calc_metrics(rs_pred, rs_lab)

    # --- Final summary
    print("\n================ Final Evaluation Summary ================")
    print(f"Overall Accuracy (ignoring 0): {avg_accuracy * 100:.2f}%")
    for i, (rec, prec) in enumerate(zip(avg_recall, avg_precision), start=1):
        f1 = 2 * rec * prec / (rec + prec + 1e-10)
        print(f"Class {i}: Recall = {rec:.4f}, Precision = {prec:.4f}, F1 Score = {f1:.4f}")


if __name__ == "__main__":
    main()
