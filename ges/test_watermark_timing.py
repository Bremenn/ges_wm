#!/usr/bin/env python3  
import subprocess  
import os  
  
# 测试不同的水印开始时间点  
start_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  
  
for ratio in start_ratios:  
    output_dir = f"./outputs/t_and_t/timing={ratio}"  
    cmd =f"python train_ges_wm.py -s ./tandt_db/tandt/train   -m {output_dir}   --enable_watermark   --watermark_start_ratio {ratio}   --watermark_msg_len 4"  
      
    print(f"Running experiment with start ratio {ratio}")  
    print(f"Command: {cmd}")  
      
    # 执行训练  
    subprocess.run(cmd, shell=True)  
      
    print(f"Completed experiment with start ratio {ratio}")  
    print("=" * 50)