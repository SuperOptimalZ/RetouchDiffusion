
from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()
from PIL import Image
from torchvision import transforms
import cv2
import einops
import numpy as np
import torch
import random
import glob
import os
import argparse
from fivek_dataset import create_webdataset
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from PE_Net import PE_Net
from torch.utils.data import DataLoader
from cldm.ddim_hacked import DDIMSampler

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='checkpoints/FiveK-epoch=120-step=280000.ckpt', type=str)
parser.add_argument("--same_folder", default='result', type=str)
parser.add_argument("--input_folder", default='test_data/input', type=str)
parser.add_argument("--target_folder", default='test_data/target', type=str)
# parser.add_argument("--same_folder", default='output_lol', type=str)
# parser.add_argument("--input_folder", default='data/lolv1-test', type=str)


parser.add_argument("--use_float16", default=False, type=bool)
parser.add_argument("--save_memory", default=False, type=bool) # Cannot use. Has bugs

if __name__ == '__main__':

    args = parser.parse_args()
    checkpoint_file = args.checkpoint

    pe_net = PE_Net()
    pe_net_weights = torch.load('savemodel/pe_net.pt', map_location='cpu')  
    pe_net.load_state_dict(pe_net_weights)
    pe_net = pe_net.cuda().eval()

    # Load pretrained Stable Diffusion v1.5
    model = create_model('./models/cldm_v15.yaml').cpu()
    
    model.load_state_dict(load_state_dict(checkpoint_file, location='cpu'))
    model = model.cuda().eval()
    
    print("====== Finish loading parameters ======")

    if args.use_float16:
        model = model.cuda().to(dtype=torch.float16)
    else:
        model = model.cuda()
    diffusion_sampler = DPMSolverSampler(model)

    def process(input_image, ref, prompt="", num_samples=1, image_resolution=512, diffusion_steps=10, guess_mode=False, strength=1.0, scale=9.0, seed=0, eta=0.0):
        with torch.no_grad():
            
            detected_map = resize_image(HWC3(input_image), image_resolution)
            ref_detected_map = resize_image(HWC3(ref), image_resolution)
            
            H, W, C = detected_map.shape

            if args.use_float16:
                control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
                ref_control = torch.from_numpy(ref_detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
            else:
                control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
                ref_control = torch.from_numpy(ref_detected_map.copy()).cuda() / 255.0
            control = pe_net(control.permute(2, 0, 1).unsqueeze(0), ref_control.permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0)

            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            print(control)
            control_image = (control[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(f'control_image_{seed}.png', control_image)  # 将图像保存为文件

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
            shape = (4, H // 8, W // 8)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            # ddim_steps = 50
            # samples, intermediates = diffusion_sampler.sample(ddim_steps, num_samples,
            #                                          shape, cond, verbose=False, eta=eta,
            #                                          unconditional_guidance_scale=scale,
            #                                          unconditional_conditioning=un_cond)
            samples, intermediates = diffusion_sampler.sample(diffusion_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond,
                                                        dmp_order=3)
            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)
            
            if args.use_float16:
                x_samples = model.decode_first_stage(samples.to(dtype=torch.float16))
            else:
                x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            # print(x_samples)
            results = [x_samples[i] for i in range(num_samples)]
        return results

    
    os.makedirs(args.same_folder, exist_ok=True)
    
    save_path = os.path.join(args.same_folder, "1.png")
    input_path = os.path.join(args.input_folder, "1.png")
    ref_path = os.path.join(args.target_folder, "1.jpg")
    input_image = cv2.imread(input_path)
    ref_image = cv2.imread(ref_path)
    
    output = process(input_image, ref_image, num_samples=1)[0]
    
    cv2.imwrite(save_path, output)
