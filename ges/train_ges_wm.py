#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations
import datetime
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui , render_laplacian
import sys
from scene import Scene, GaussianModel , LaplacianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr , apply_dog_filter
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb

# 添加水印导入  
try:  
    from wm_utils import load_decoder_and_message, bit_accuracy, lpips  
    from attack_utils import Attacker  
    from nerf_utils import get_data_infos, get_cameras  
      
except ImportError:  
    print("Warning: watermark utilities not available")  
     
# finish

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, watermark_args=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = LaplacianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 添加水印初始化  
    watermark_model = None  
    watermark_message = None  
    shape_offsets = None  
    watermark_optimizer = None  
    WATERMARK_AVAILABLE = True
      
    if watermark_args and WATERMARK_AVAILABLE:  
        print("Initializing watermark components...")  
        watermark_model, watermark_message = load_decoder_and_message(watermark_args.msg_len, watermark_args.decoder_path)  
          
        shape_features = gaussians._shape.clone()  
        shape_offsets = torch.nn.Parameter(  
            torch.zeros_like(shape_features).cuda().requires_grad_(True)  
        )  
          
        watermark_optimizer = torch.optim.Adam([shape_offsets], lr=watermark_args.learning_rate)  
          
        os.makedirs(watermark_args.watermark_dir, exist_ok=True)  
        message_text = ''.join([str(x) for x in watermark_message.int().cpu().numpy()])  
        with open(os.path.join(watermark_args.watermark_dir, 'message.txt'), 'w') as f:  
            f.write(message_text)  
        print(f"Watermark message: {message_text}")  
  
    watermark_start_iter = int(opt.iterations * watermark_args.start_ratio)  
    # finish


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):    
        freq = (iteration / opt.iterations) * 100
    
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_laplacian(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 添加形状偏移量应用  
        if shape_offsets is not None and iteration >= watermark_start_iter:  
            original_shape = gaussians._shape.data.clone()    
      
            # 检查尺寸匹配，如果不匹配则调整偏移量    
            if original_shape.shape[0] != shape_offsets.shape[0]:    
                print(f"Adjusting shape offsets from {shape_offsets.shape[0]} to {original_shape.shape[0]}")      
      
                # 保存现有偏移量  
                existing_offsets = shape_offsets.data.clone()  
                existing_count = existing_offsets.shape[0]  
                new_count = original_shape.shape[0]  
                
                # 创建新的偏移量张量  
                new_offsets = torch.zeros(new_count, 1, device='cuda')  
                
                # 处理尺寸变化  
                if new_count >= existing_count:  
                    # 扩展情况：保留现有值，新增点用小随机值  
                    new_offsets[:existing_count] = existing_offsets  
                    new_offsets[existing_count:] = torch.randn(new_count - existing_count, 1, device='cuda') * 0.01  
                else:  
                    # 缩减情况：截取前N个值  
                    new_offsets = existing_offsets[:new_count]  
                
                # 重新创建参数  
                shape_offsets = torch.nn.Parameter(new_offsets.requires_grad_(True))  
                watermark_optimizer = torch.optim.Adam([shape_offsets], lr=watermark_args.learning_rate)  
                
            gaussians._shape.data = original_shape + shape_offsets
            #finish
        render_pkg = render_laplacian(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]



        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        mask = apply_dog_filter(image.unsqueeze(0), freq=freq, scale_factor=opt.im_laplace_scale_factor).squeeze(0)
        mask_loss = l1_loss(image * mask, gt_image * mask)
        loss = (1.0 - opt.lambda_dssim -opt.lambda_im_laplace ) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_im_laplace * mask_loss
        # 添加水印损失  
        total_loss = loss  
        if shape_offsets is not None and iteration >= watermark_start_iter:  
            try:  
                resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=[224, 224], mode='bilinear', align_corners=False)  
                  
                with torch.no_grad():  
                    watermark_output = watermark_model(resized_image)  
                  
                target = watermark_message.unsqueeze(0)  
                msg_loss = torch.nn.functional.binary_cross_entropy_with_logits(watermark_output, target)  
                shape_loss = torch.mean(torch.abs(shape_offsets))  
                  
                total_loss = loss + watermark_args.lambda_msg * msg_loss + watermark_args.lambda_off * shape_loss  
            except Exception as e:  
                print(f"Watermark loss computation failed: {e}")  
                total_loss = loss
            total_loss.backward()
        #finish
        else:
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,mask_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # 添加最终水印固化  
                if shape_offsets is not None and iteration >= watermark_start_iter:  
                    gaussians._shape.data = gaussians._shape.data + shape_offsets.data  
                      
                    if watermark_args:  
                        torch.save({  
                            'shape_offsets': shape_offsets.data,  
                            'message': watermark_message,  
                            'iteration': iteration  
                        }, os.path.join(watermark_args.watermark_dir, 'watermark_info.pth')) 
                #finish
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration > opt.densify_from_iter and iteration % opt.shape_pruning_interval == 0:
                    gaussians.size_prune(opt.prune_shape_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                if iteration % opt.shape_reset_interval == 0 : # or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_shape()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 在这里添加水印优化器更新  
                if watermark_optimizer is not None and iteration >= watermark_start_iter:  
                    watermark_optimizer.step()  
                    watermark_optimizer.zero_grad()
            #finish
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss,mask_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        wandb.log({'train_loss_patches/l1_loss': Ll1.item(),
            'train_loss_patches/total_loss': loss.item(),
            'train_loss_patches/mask_loss': mask_loss.item(),
            'iter_time': elapsed,
            'scene/total_points': scene.gaussians.get_xyz.shape[0],
            'scene/small_points':(scene.gaussians.get_shape < 0.5).sum().item(),
            'scene/average_shape':scene.gaussians.get_shape.mean().item(),
            'scene/large_shapes':scene.gaussians.get_shape[scene.gaussians.get_shape>=1.0].mean().item(),
            'scene/small_shapes':scene.gaussians.get_shape[scene.gaussians.get_shape<1.0].mean().item(),
            'scene/opacity_grads':scene.gaussians._opacity.grad.data.norm(2).item(),
            'scene/shape_grads':scene.gaussians._shape.grad.data.norm(2).item(),
            
        })        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        # tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        wandb.log({"renders/{}_view_{}/render".format(config['name'], viewpoint.image_name): 
                                [wandb.Image(image, caption="Render at iteration {}".format(iteration))],
                            })
                        if iteration == testing_iterations[0]:
                            # tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            wandb.log({"renders/{}_view_{}/ground_truth".format(config['name'], viewpoint.image_name): 
                [wandb.Image(gt_image, caption="Ground truth at iteration {}".format(iteration))],
            })
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    wandb.log({
                        "metrics/"+config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                        "metrics/"+config['name'] + '/loss_viewpoint - psnr': psnr_test,
                    })
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            opacity_data = [[val] for val in scene.gaussians.get_opacity.cpu().squeeze().tolist()]
            shape_data = [[val] for val in scene.gaussians.get_shape.cpu().squeeze().tolist()]
            wandb.log({
            "scene/opacity_histogram": wandb.plot.histogram(wandb.Table(data=opacity_data, columns=["opacity"]), "opacity", title="Opacity Histogram"),
            "scene/shape_histogram": wandb.plot.histogram(wandb.Table(data=shape_data, columns=["shape"]), "shape", title="Shape Histogram"),
            })
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            # tb_writer.add_histogram("scene/shape_histogram", scene.gaussians.get_shape, iteration)
            # tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 添加水印参数类  
class WatermarkArgs:  
    def __init__(self):  
        self.enable_watermark = False  
        self.msg_len = 4  
        self.decoder_path = 'decoders'  
        self.watermark_dir = './watermarks'  
        self.learning_rate = 5e-3  
        self.lambda_msg = 0.03  
        self.lambda_off = 10.0
        self.start_ratio = 0.05
# finish
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--nowandb", action="store_false", dest='wandb')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    # 添加水印参数  
    parser.add_argument('--enable_watermark', action='store_true', help='Enable watermark embedding')  
    parser.add_argument('--watermark_msg_len', type=int, default=4, help='Watermark message length')  
    parser.add_argument('--watermark_dir', type=str, default='./watermarks', help='Watermark output directory')  
    parser.add_argument('--watermark_lr', type=float, default=5e-3, help='Watermark learning rate')  
    parser.add_argument('--watermark_lambda_msg', type=float, default=0.03, help='Watermark message loss weight')  
    parser.add_argument('--watermark_lambda_off', type=float, default=10.0, help='Watermark offset loss weight')
    parser.add_argument('--watermark_start_ratio', type=float, default=0.05, help='Watermark training start ratio (0.05-0.95)')
    # finish 
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 在这里创建水印参数对象  
    watermark_args = None  
    if args.enable_watermark:  
        watermark_args = WatermarkArgs()  
        watermark_args.msg_len = args.watermark_msg_len  
        watermark_args.watermark_dir = args.watermark_dir  
        watermark_args.learning_rate = args.watermark_lr  
        watermark_args.lambda_msg = args.watermark_lambda_msg  
        watermark_args.lambda_off = args.watermark_lambda_off
        watermark_args.start_ratio = args.watermark_start_ratio  
    # finish

    # Initialize system state (RNG)
    exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # random_id()
    args.model_path = args.model_path + "_" + args.exp_set + "_" +  exp_id
    print("Optimizing " + args.model_path)
    safe_state(args.quiet, args.seed)
    setup = vars(args)
    setup["exp_id"] = exp_id
    if args.wandb:
        wandb_id = args.model_path.replace('outputs', '').lstrip('./').replace('/', '---')
        wandb.init(project="ges", id=wandb_id, config = setup ,sync_tensorboard=False,settings=wandb.Settings(_service_wait=600))

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, watermark_args)

    # All done
    print("\nTraining complete.")
    if args.wandb:
        wandb.finish()