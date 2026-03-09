#!/usr/bin/env python3  
#  
# GES Watermark Evaluation Script with Image Saving
# Tests PSNR, SSIM, LPIPS, and bit_accuracy for watermarked GES models  
#  
  
import os  
import json  
import torch  
import numpy as np  
import cv2
from argparse import ArgumentParser  
from functools import reduce  
from tqdm import tqdm  
  
from scene import LaplacianModel  
from gaussian_renderer import render_laplacian  
from utils.graphics_utils import focal2fov, getProjectionMatrix  
from scene.cameras import MiniCam  
from utils.loss_utils import ssim  
from utils.image_utils import psnr  
from wm_utils import load_decoder_and_message, bit_accuracy  
from arguments import ModelParams, PipelineParams, OptimizationParams  
  
def getTestCameras(camera_path):  
    """Load test cameras from JSON file"""  
    with open(camera_path) as jf:  
        clist = json.load(jf)  
  
    def getSE3(r, t):  
        SE3 = np.eye(4)  
        SE3[:3, :3] = np.array(r)  
        SE3[:3, 3] = np.array(t)  
        return SE3  
  
    w2cs = [np.linalg.inv(getSE3(c['rotation'], c['position'])).transpose() for c in clist]  
    w2cs = torch.from_numpy(np.array(w2cs)).cuda().float()  
    params = {  
        'width'  : clist[0]['width'],  
        'height' : clist[0]['height'],  
        'fovx'   : focal2fov(clist[0]['fx'], clist[0]['width']),  
        'fovy'   : focal2fov(clist[0]['fy'], clist[0]['height']),  
        'znear'  : 0.01,  
        'zfar'   : 100.0  
    }  
  
    proj_matrix = getProjectionMatrix(znear=params['znear'], zfar=params['zfar'], fovX=params['fovx'], fovY=params['fovy']).transpose(0,1).cuda()  
    projs = w2cs.bmm(proj_matrix.repeat(len(w2cs), 1, 1))  
    return [MiniCam(world_view_transform=w2c, full_proj_transform=proj, **params) for w2c, proj in zip(w2cs, projs)]  
  
def extract_rendered_views_and_gts(gaussians, ges_wm, cameras, pipe, background, save_dir=None):  
    """Extract rendered views from both models and save images"""
    with torch.no_grad():  
        pds, gts = [], []  
        
        # 创建保存目录
        if save_dir:
            os.makedirs(os.path.join(save_dir, "watermarked"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "original"), exist_ok=True)
        
        for i, viewpoint in enumerate(tqdm(cameras, desc="Rendering views")):  
            # 渲染水印模型
            watermarked_img = torch.clamp(render_laplacian(viewpoint, ges_wm, pipe, background)["render"], 0.0, 1.0)
            pds.append(watermarked_img[None])
            
            # 渲染原始模型
            original_img = torch.clamp(render_laplacian(viewpoint, gaussians, pipe, background)["render"], 0.0, 1.0)
            gts.append(original_img[None])
            
            # 保存图像用于验证
            if save_dir and i < 10:  # 只保存前10张用于检查
                # 转换为numpy格式 (C, H, W) -> (H, W, C)
                watermarked_np = watermarked_img.permute(1, 2, 0).cpu().numpy() * 255
                original_np = original_img.permute(1, 2, 0).cpu().numpy() * 255
                
                # 保存为PNG
                cv2.imwrite(os.path.join(save_dir, "watermarked", f"view_{i:03d}.png"), watermarked_np[:, :, ::-1])  # RGB to BGR
                cv2.imwrite(os.path.join(save_dir, "original", f"view_{i:03d}.png"), original_np[:, :, ::-1])
                
                # 清理内存
                del watermarked_np, original_np
                torch.cuda.empty_cache()
                
        return torch.cat(pds), torch.cat(gts)  
  
@torch.no_grad()  
def eval_image_similarity(pds, gts, batch_size=32):  
    """Calculate PSNR, SSIM, LPIPS with batch processing"""  
    PSNR, SSIM, LPIPS = [], [], 0  
    from wm_utils import lpips  
      
    with torch.no_grad():  
        for idx in tqdm(range((len(gts) + batch_size - 1) // batch_size), desc="Calculating image metrics"):  
            start_idx = idx * batch_size  
            end_idx = min((idx + 1) * batch_size, len(gts))  
              
            # Batch processing for PSNR and SSIM  
            batch_pds = pds[start_idx:end_idx]  
            batch_gts = gts[start_idx:end_idx]  
              
            PSNR.append(psnr(batch_pds, batch_gts).cpu())  
            SSIM.append(ssim(batch_pds, batch_gts, size_average=False).cpu())  
              
            # Batch processing for LPIPS  
            lpips_batch = lpips(batch_pds, batch_gts).item()  
            LPIPS += lpips_batch  
          
    return {  
        'PSNR'  : torch.cat(PSNR).mean().item(),  
        'SSIM'  : torch.cat(SSIM).mean().item(),  
        'LPIPS' : LPIPS / len(pds)  
    }  
  
@torch.no_grad()  
def eval_bit_accuracy(pds, watermark_model, watermark_message, batch_size=32):  
    """Calculate bit accuracy with batch processing"""  
    decoded_messages = []  
      
    with torch.no_grad():  
        for idx in tqdm(range((len(pds) + batch_size - 1) // batch_size), desc="Extracting watermark"):  
            start_idx = idx * batch_size  
            end_idx = min((idx + 1) * batch_size, len(pds))  
              
            batch_images = pds[start_idx:end_idx]  
            resized_batch = torch.nn.functional.interpolate(batch_images, size=[224, 224], mode='bilinear', align_corners=False)  
            decoded_messages.append(watermark_model(resized_batch))  
      
    # Calculate bit accuracy  
    all_outputs = torch.cat(decoded_messages)  
    target = watermark_message.unsqueeze(0).repeat(len(pds), 1)  
    predicted_bits = (all_outputs > 0.5).float()  
    bit_acc = (predicted_bits == target).float().mean().item() * 100  
      
    return bit_acc  
  
def evaluate_ges_watermark(modelwm_path, original_path, camera_path, watermark_dir, msg_len, num_views, dataset, opt, pipe, save_images=False):  
    """Evaluate GES watermark performance"""  
    print("=== GES Watermark Evaluation ===")  
      
    # 创建保存目录
    save_dir = None
    if save_images:
        timing_now = "unknown"  
        if "timing=" in modelwm_path:
            timing_now = modelwm_path.split("timing=")[-1].split("\\")[0].split("/")[-1] 
        save_dir = f"./render_checks/{timing_now}"
        print(f"Images will be saved to: {save_dir}")
    
    # Load models  
    print("Loading GES models...")  
    gaussians = LaplacianModel(dataset.sh_degree)  
    gaussians.load_ply(original_path)  
      
    ges_wm = LaplacianModel(dataset.sh_degree)  
    ges_wm.load_ply(modelwm_path)  
      
    # Setup rendering  
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  
      
    # Load cameras  
    print(f"Loading {num_views} cameras...")  
    cameras = getTestCameras(camera_path)  
    cameras = cameras[:num_views]  # Limit to specified number  
      
    # Render views and save images  
    print("Rendering views...")  
    pds, gts = extract_rendered_views_and_gts(gaussians, ges_wm, cameras, pipe, background, save_dir)  
    
    # 打印渲染统计信息
    print(f"\nRendered {len(pds)} views:")
    print(f"  Watermarked images shape: {pds.shape}")
    print(f"  Original images shape: {gts.shape}")
    print(f"  Image range: [{pds.min().item():.3f}, {pds.max().item():.3f}]")
    
    # Load watermark components  
    print("Loading watermark components...")  
    watermark_model, watermark_message = load_decoder_and_message(msg_len, watermark_dir)  
      
    # Calculate image quality metrics with batch processing  
    print("Calculating image quality metrics...")  
    image_results = eval_image_similarity(pds, gts, batch_size=32)  
      
    # Extract watermark with batch processing  
    print("Extracting watermark...")  
    bit_acc = eval_bit_accuracy(pds, watermark_model, watermark_message, batch_size=32)  
      
    results = {  
        'PSNR': image_results['PSNR'],  
        'SSIM': image_results['SSIM'],  
        'LPIPS': image_results['LPIPS'],  
        'Bit_Accuracy': bit_acc  
    }  
    
    # 保存结果到文件
    if save_dir:
        results_file = os.path.join(save_dir, "evaluation_results.txt")
        with open(results_file, 'w') as f:
            f.write("=== GES Watermark Evaluation Results ===\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.6f}\n")
            f.write(f"\nModels evaluated:\n")
            f.write(f"  Watermarked: {modelwm_path}\n")
            f.write(f"  Original: {original_path}\n")
            f.write(f"  Views rendered: {len(pds)}\n")
        
        print(f"\nResults saved to: {results_file}")
        print(f"Sample images saved in: {save_dir}/")
      
    return results  
  
if __name__ == "__main__":  
    parser = ArgumentParser(description="GES Watermark Evaluation")  
    lp = ModelParams(parser)  
    op = OptimizationParams(parser)  
    pp = PipelineParams(parser)  
      
    parser.add_argument('--modelwm_path', type=str, required=True, help='Path to watermarked GES model')  
    parser.add_argument('--original_path', type=str, required=True, help='Path to original GES model')  
    parser.add_argument('--camera_path', type=str, required=True, help='Path to cameras.json file')  
    parser.add_argument('--watermark_dir', type=str, required=True, help='Watermark directory')  
    parser.add_argument('--msg_len', type=int, default=32, help='Message length')  
    parser.add_argument('--num_views', type=int, default=200, help='Number of views to render')  
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')  
    parser.add_argument('--save_images', action='store_true', help='Save rendered images for verification')  
      
    args = parser.parse_args()  
      
    results = evaluate_ges_watermark(  
        args.modelwm_path, args.original_path, args.camera_path,  
        args.watermark_dir, args.msg_len, args.num_views,  
        lp.extract(args), op.extract(args), pp.extract(args),
        save_images=args.save_images
    )  
      
    print("\n" + "="*50)  
    print("=== Evaluation Results ===")  
    print("="*50)  
    for metric, value in results.items():  
        print(f"{metric}: {value:.6f}")
    
    print("\n" + "="*50)
    print("Note: To save rendered images for verification, use --save_images flag")
    print("Images will be saved in ./render_checks/ directory")