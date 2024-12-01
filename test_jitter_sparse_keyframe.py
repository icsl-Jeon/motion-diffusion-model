from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, create_gaussian_diffusion
import pickle
import torch
import pickle, os
from tqdm import tqdm
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import matplotlib.pyplot as plt


def compute_derivatives(sample):
    n_joints = 22
    b = sample.shape[0]
    sample = recover_from_ric(sample.cpu().permute(0, 2, 3, 1), n_joints)
    joint_positions = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    velocity_vectors = torch.diff(joint_positions, dim=3, prepend=torch.zeros(b, n_joints, 3, 1))
    acceleration_vectors = torch.diff(velocity_vectors, dim=3, prepend=torch.zeros(b, n_joints, 3, 1))
    jerk_vectors = torch.diff(acceleration_vectors, dim=3, prepend=torch.zeros(b, n_joints, 3, 1))

    velocity_magnitudes = torch.norm(velocity_vectors, dim=1) 
    acceleration_magnitudes = torch.norm(acceleration_vectors, dim=1)
    jerk_magnitudes = torch.norm(jerk_vectors, dim=2)
    return velocity_magnitudes, acceleration_magnitudes, jerk_magnitudes


def compute_inpainting_cost(sample_in_batch, keyframes, length, window_size=4):
    traj = sample_in_batch.squeeze(1)
    jerk_squared = torch.diff(traj, n=3, dim=1).square()
    jerk_sum = 0
    intergration_horizon = 0
    for keyframe in keyframes:
        intergration_start = max(keyframe - window_size, 0)
        intergration_end = min(keyframe + window_size, length-2)
        intergration_horizon += (intergration_end - intergration_start)
        jerk_sum += torch.trapz(jerk_squared[..., intergration_start:intergration_end])
    return jerk_sum.sum().item() / intergration_horizon


def find_true_regions(inpainting_vector):
    regions = []
    in_region = False
    start = 0
    for i, val in enumerate(inpainting_vector):
        if val and not in_region:  
            in_region = True
            start = i
        elif not val and in_region:  
            in_region = False
            regions.append((start, i))
    if in_region:  
        regions.append((start, len(inpainting_vector)))
    return regions


if __name__ == "__main__":
    fixseed(10)

    data_suite_folder = "test_suite"
    with open(os.path.join(data_suite_folder, 'model_1000.bin'), 'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(data_suite_folder, 'args_1000.bin'), 'rb') as file:
        args = pickle.load(file)
    diffusion = create_gaussian_diffusion(args)

    model.to("cuda:0")
    model.eval()  

    with open(os.path.join(data_suite_folder, 'model_kwargs.bin'), 'rb') as file:
        model_kwargs_default = pickle.load(file)
    import copy

    KEYFRAME_RATIOS = [0, 0.5, 1.0]

    input_motions = model_kwargs_default['y']['inpainted_motion']
    lengths = model_kwargs_default['y']['lengths'].cpu().numpy()
    model_kwargs_list = []
    padding_step_list = range(0, 2)
    sparse_keyframes_list = []
    for padding_step in padding_step_list:
        model_kargs = {}
        model_kargs = copy.deepcopy(model_kwargs_default)
        for b, length in enumerate(lengths):
            sparse_keyframes = []
            model_kargs['y']['inpainting_mask'][b] = torch.zeros_like(input_motions[b], dtype=torch.bool,
                                                                    device=input_motions.device)
            for keyframe_idx in [min(int(ratio*length), length-1) for ratio in KEYFRAME_RATIOS]:
                keyframe_window_start = max(keyframe_idx - padding_step, 0)
                keyframe_window_end = min(keyframe_idx + padding_step, length)
                if keyframe_window_start == keyframe_window_end:
                    model_kargs['y']['inpainting_mask'][b, ..., keyframe_window_start] = True    
                    sparse_keyframes.append(keyframe_window_start)
                else:
                    model_kargs['y']['inpainting_mask'][b, ..., keyframe_window_start: keyframe_window_end] = True
            if padding_step == 0:
                sparse_keyframes_list.append(sparse_keyframes)
        model_kwargs_list.append(model_kargs)


    sample_fn = diffusion.p_sample_loop
    batch_size = len(model_kwargs_default['y']['lengths'].cpu().numpy())
    max_frames = 196
    sample_list = []
    for model_kwargs in tqdm(model_kwargs_list):
        sample = sample_fn(
            model,
            (batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        for b, length in zip(range(batch_size), model_kwargs_default['y']['lengths'].cpu().numpy()):
            sample[b][...,length:] = 0

        sample_list.append(sample)


    joint_idx = 0
    dense_index = 8
    sparse_index = 0

    fig, axes = plt.subplots(batch_size, 2, figsize=(15, 30), sharex=True)
    colors = ['blue', 'red']
    for i, index in enumerate([dense_index, sparse_index]):
        for batch_idx in range(batch_size):
            length = model_kwargs_default['y']['lengths'].cpu().numpy()[batch_idx]
            
            true_regions = find_true_regions(list(model_kwargs_list[index]['y']['inpainting_mask'][batch_idx,0,0].cpu().numpy()))

            for start, end in true_regions:
                axes[batch_idx, 0].axvspan(start, end-1, color=colors[i], alpha=0.2)
            input_traj = input_motions[batch_idx][joint_idx][0].cpu().numpy()[:length]
            inpainting_cost = \
                compute_inpainting_cost(sample_list[sparse_index][batch_idx][joint_idx].unsqueeze(0), sparse_keyframes_list[batch_idx], length)
            traj = sample_list[index][batch_idx][joint_idx][0].cpu().numpy()
            axes[batch_idx, 0].plot(abs(traj[:length]-input_traj), color=colors[i])
            axes[batch_idx, 0].set_title(f'Position Error / cost of sparse = {inpainting_cost:2f}')
            axes[batch_idx, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
            vel = abs((traj[1:] - traj[:-1]))


            axes[batch_idx, 1].plot(vel[:length], color=colors[i])
            for start, end in true_regions:
                axes[batch_idx, 1].axvspan(start, end-1, color=colors[i], alpha=0.1)
            axes[batch_idx, 1].set_title(f'Velocity')


    plt.tight_layout()
    plt.savefig('fig.png')
