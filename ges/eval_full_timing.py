#!/usr/bin/env python3  
import os  
import subprocess  
import sys  
  
# 配置  
original_path = r"D:\3dgsRelated\ges_wm\ges\outputs\t_and_t\original\point_cloud\iteration_40000\point_cloud.ply"  
camera_path = r"D:\3dgsRelated\ges_wm\ges\outputs\t_and_t\original\cameras.json"  
watermark_dir = r"D:\3dgsRelated\ges_wm\ges\decoders"  
  
# 测试的时间点  
timing_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  
  
print("=== GES Watermark Timing Evaluation ===")  
print(f"Original model: {original_path}")  
print(f"Camera file: {camera_path}")  
print(f"Watermark dir: {watermark_dir}")  
print()  
  
results = []  
  
for ratio in timing_ratios:  
    wmmodel_path = fr"D:\3dgsRelated\ges_wm\ges\outputs\t_and_t\timing={ratio}\point_cloud\iteration_40000\point_cloud.ply"  
      
    if not os.path.exists(wmmodel_path):  
        print(f"❌ Skipping ratio {ratio}: Model not found")  
        continue  
      
    print(f"🔄 Evaluating ratio {ratio}...")  
      
    cmd = f"python eval_timing_with_picture.py --modelwm_path {wmmodel_path} --original_path {original_path} --camera_path {camera_path} --watermark_dir {watermark_dir} --msg_len 4 --num_views 200 --save_images"  
      
    try:  
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)  
          
        if result.returncode == 0:  
            print(f"✅ Ratio {ratio} completed successfully")  
            # 简单提取结果  
            output = result.stdout  
            results.append(f"Ratio {ratio}: {output}")  
        else:  
            print(f"❌ Ratio {ratio} failed: {result.stderr}")  
            results.append(f"Ratio {ratio}: FAILED")  
    except subprocess.TimeoutExpired:  
        print(f"⏰ Ratio {ratio} timed out")  
        results.append(f"Ratio {ratio}: TIMEOUT")  
    except Exception as e:  
        print(f"❌ Ratio {ratio} error: {e}")  
        results.append(f"Ratio {ratio}: ERROR")  
      
    print()  
  
print("=== SUMMARY ===")  
for result in results:  
    print(result)