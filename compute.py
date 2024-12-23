import os
import numpy as np
from paired_metrics import metric
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute average metrics (PSNR, SSIM, LPIPS, LOE) given GT and prediction directories.")
    parser.add_argument("--gt_dir", type=str, default="data/FiveK_C/test/target", required=False, help="Path to the ground truth images directory.")
    parser.add_argument("--pred_dir", type=str, default="final_Fivek", required=False, help="Path to the predicted images directory.")
    # parser.add_argument("--gt_dir", type=str, default="data/lolv1-test/target", required=False, help="Path to the ground truth images directory.")
    # parser.add_argument("--pred_dir", type=str, default="final_lol", required=False, help="Path to the predicted images directory.")
    args = parser.parse_args()

    gt_files = os.listdir(args.gt_dir)
    gt_files = [f for f in gt_files if os.path.isfile(os.path.join(args.gt_dir, f))]
    gt_files.sort()

    psnr_list, ssim_list, lpips_list, loe_list = [], [], [], []

    for f in gt_files:
        gt_path = os.path.join(args.gt_dir, f)

        base_name = os.path.splitext(f)[0]
        ref_name = base_name + '.png'
        ref_path = os.path.join(args.pred_dir, ref_name)

        if not os.path.exists(ref_path):
            print(f"Warning: Corresponding png not found for {f} (expected {ref_name}), skipping.")
            continue
        
        psnr, ssim, lp = metric(gt_path, ref_path)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lp)

    if len(psnr_list) == 0:
        print("No matching images found or no metrics computed.")
        exit(0)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")