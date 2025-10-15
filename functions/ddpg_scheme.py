import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from datasets import inverse_data_transform, data_transform
import shutil
import matplotlib.pyplot as plt
import cv2

class_num = 951

def save_img(i,path,config,img,b_num):
    if not os.path.exists(path):
        os.mkdir(path)
    tvu.save_image(
        inverse_data_transform(config, img), 
        os.path.join(path,f'{i}_{b_num}.png')
    )
def save_heatmap(image, path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
def get_face_bbox(face_detector,image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray)
    if faces is None or len(faces)==0:
        return None,None,None,None
    x, y, w, h = faces[0]
    return x, y, w, h
def save_gaussian_heatmap(kernel, path):
    """
    Saves the Gaussian kernel as a heatmap image.
    :param kernel: The Gaussian kernel (2D numpy array).
    :param path: The path where the heatmap image will be saved.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(kernel, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
def obfuscate_tensor_region(tensor, epsilon, bbox):
    x, y, w, h = bbox
    # Extract the region of interest (ROI)
    roi = tensor[:, :, y:y+h, x:x+w]
    
    # Get the current min and max of the ROI to determine pixel range
    pixel_range = roi.max() - roi.min()
    
    # Compute sensitivity for the ROI
    n = w * h  # Pixel count within the bounding box per channel
    c = tensor.size(1)  # Number of channels
    sensitivity = pixel_range * n * c

    # Compute Laplace scale
    #scale = sensitivity / epsilon
    scale = sensitivity / (sensitivity*epsilon)

    # Generate Laplace noise for the ROI
    noise = torch.from_numpy(
        np.random.laplace(0, scale.item(), roi.shape).astype(np.float32)
    ).to(tensor.device)

    # Add noise to the ROI
    obfuscated_roi = roi + noise

    # Replace the obfuscated region back in the original tensor
    tensor[:, :, y:y+h, x:x+w] = obfuscated_roi
    return tensor
def create_gaussian_kernel(shape, bbox, variance):
    """
    Creates a Gaussian kernel with the specified shape and variance centered on the bbox center,
    only applied within the bbox area, and sets the area outside the bbox to ones.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :param variance: Variance of the Gaussian kernel.
    :return: Gaussian kernel of shape (H, W).
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]
    
    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)
    
    # Create a grid within the bounding box area
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    
    # Calculate the Gaussian only within the bounding box area
    gaussian = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance))
    inverted_gaussian = 1 - gaussian  # Invert the values

    # Place the inverted Gaussian in the bounding box area of the kernel
    kernel[y_start:y_end, x_start:x_end] = inverted_gaussian

    return kernel
def create_laplacian_kernel(shape, bbox, variance):
    """
    Creates a Laplacian kernel centered on the bbox center,
    applied only within the bbox area, and sets the area outside the bbox to ones.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :param variance: Variance (scale factor) of the Laplacian function.
    :return: Laplacian kernel of shape (H, W).
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)

    # Create a grid within the bounding box area
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    # Compute the Laplacian function inside the bbox
    # Mexican Hat (Laplacian of Gaussian, LoG) function
    distance_squared = (x - x_center) ** 2 + (y - y_center) ** 2
    laplacian = (1 - (distance_squared / (2 * variance))) * np.exp(-distance_squared / (2 * variance))

    # Normalize and invert the Laplacian
    inverted_laplacian = 1 - laplacian  # Invert values

    # Place the inverted Laplacian in the bounding box area of the kernel
    kernel[y_start:y_end, x_start:x_end] = inverted_laplacian

    return kernel
def create_zero_kernel(shape, bbox):
    """
    Creates a kernel with ones everywhere except for the bbox region, which is set to zero.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :return: Kernel of shape (H, W) with zeros in the bbox area.
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)

    # Set the bounding box area to zero
    kernel[y_start:y_end, x_start:x_end] = 0

    return kernel
def get_average_bbox_and_std(rectangles_list):
    # Calculate average and standard deviation if we have more than one rectangle
    if len(rectangles_list) > 1:
        rect_coords = np.array(rectangles_list)
        avg_rect_coords = np.mean(rect_coords, axis=0)
        std_dev_rect_coords = np.std(rect_coords, axis=0)
        avg_rect = (
            int(avg_rect_coords[0]), int(avg_rect_coords[1]),
            int(avg_rect_coords[2]), int(avg_rect_coords[3])
        )
        std_dev_rect = std_dev_rect_coords  # Standard deviation in each coordinate
    else:
        avg_rect = rectangles_list[0]
        std_dev_rect = np.zeros(4)  # No deviation with only one rectangle
    return avg_rect,std_dev_rect
def plot_alphas(alphas,path):
    # Generate x values as indices
    x_values = list(range(len(alphas)))

    # Plot the values
    plt.plot(x_values, alphas, marker='o', linestyle='-')

    # Label the axes
    plt.xlabel("t")
    plt.ylabel("alphas")

    # Show the plot
    # Flip only the x-axis labels

    tick_positions = np.linspace(0, len(x_values) - 1, num=10, dtype=int)  # Choose only 10 tick positions
    plt.xticks(ticks=tick_positions, labels=[x_values[i] for i in tick_positions][::-1])

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_percentile_index(lst, percentile=70):
    idx = int(len(lst) * (percentile / 100))  # Compute index
    idx = min(idx, len(lst) - 1)  # Ensure it's within bounds
    return idx  # Return the index


# -------------------------------------------------------
# Modify ddpg diffusion to accept kernels as a parameters
# -------------------------------------------------------

def create_dog_kernel(shape, bbox, variance1, variance2):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    gauss1 = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance1))
    gauss2 = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance2))

    dog = gauss1 - gauss2
    dog = (dog - dog.min()) / (dog.max() - dog.min())  # Normalize 0 to 1
    inverted_dog = 1 - dog

    kernel[y_start:y_end, x_start:x_end] = inverted_dog
    return kernel

def create_circular_mask_kernel(shape, bbox, radius):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    mask = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2
    kernel[y_start:y_end, x_start:x_end][mask] = 0

    return kernel

def create_linear_gradient_kernel(shape, bbox):
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)

    width = x_end - x_start
    gradient = np.linspace(0, 1, width, dtype=np.float32)
    gradient = np.tile(gradient, (y_end - y_start, 1))

    kernel[y_start:y_end, x_start:x_end] = gradient
    return kernel

def create_sigmoid_kernel(shape, bbox, slope=0.1):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    dist = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    sigmoid = 1 / (1 + np.exp(slope * (dist - bbox[2] / 2)))
    inverted = 1 - sigmoid

    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

def create_radial_linear_kernel(shape, bbox):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    max_dist = np.sqrt((bbox[2] / 2) ** 2 + (bbox[3] / 2) ** 2)
    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    dist = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    radial = dist / max_dist
    radial = np.clip(radial, 0, 1)
    inverted = 1 - radial

    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

def create_sinusoidal_kernel(shape, bbox, frequency=0.1):
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    sinusoid = 0.5 * (1 + np.sin(frequency * x + frequency * y))
    inverted = 1 - sinusoid

    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

import numpy as np
from scipy.ndimage import uniform_filter

def create_blur_kernel(shape, bbox, kernel_size=5):
    """
    Creates an average blur kernel applied only inside bbox.
    Outside bbox is ones.
    """
    H, W = shape
    kernel = np.ones((H, W), dtype=np.float32)

    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Create an initial mask with zeros inside bbox (to be blurred)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y_start:y_end, x_start:x_end] = 1

    # Apply uniform filter to mask to simulate blur kernel effect (normalized box filter)
    blurred_mask = uniform_filter(mask, size=kernel_size)

    # Invert blur effect inside bbox for consistency with your other kernels
    inverted = 1 - blurred_mask

    kernel[y_start:y_end, x_start:x_end] = inverted[y_start:y_end, x_start:x_end]
    return kernel

def create_bilateral_kernel(shape, bbox, spatial_variance=10, intensity_variance=0.1, intensity_center=0.5):
    """
    Creates a bilateral-like kernel applied only within bbox.
    Assumes intensity_center is the center intensity value for range kernel.
    """
    H, W = shape
    kernel = np.ones((H, W), dtype=np.float32)

    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    # Spatial Gaussian
    spatial_dist_sq = (x - x_center) ** 2 + (y - y_center) ** 2
    spatial_gauss = np.exp(-spatial_dist_sq / (2 * spatial_variance))

    # Intensity Gaussian centered at intensity_center (simulate pixel intensity similarity)
    intensity_diff_sq = (intensity_center - 0.5) ** 2  # Fixed for demo
    intensity_gauss = np.exp(-intensity_diff_sq / (2 * intensity_variance))

    bilateral = spatial_gauss * intensity_gauss
    inverted = 1 - bilateral

    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

def create_salt_and_pepper_kernel(shape, bbox, salt_prob=0.05, pepper_prob=0.05):
    """
    Creates a kernel mask simulating salt and pepper noise inside bbox.
    Outside bbox is ones.
    """
    H, W = shape
    kernel = np.ones((H, W), dtype=np.float32)

    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    noise_area = np.ones((bbox[3], bbox[2]), dtype=np.float32)

    # Generate salt and pepper noise
    random_vals = np.random.rand(bbox[3], bbox[2])
    noise_area[random_vals < pepper_prob] = 0.0  # pepper (black pixels)
    noise_area[random_vals > 1 - salt_prob] = 1.0  # salt (white pixels)

    kernel[y_start:y_end, x_start:x_end] = noise_area
    return kernel








kernels = {
    'gaussian': create_gaussian_kernel,
    'laplacian': create_laplacian_kernel,
}


def ddpg_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None,
                   deid=False, ckpt_imgs_path=None, face_bbox=None, gaussian_kern=False, diff_priv=False, per=1,
                   kernels=None):  # <-- added kernels param
    if ckpt_imgs_path is not None:
        if os.path.exists(ckpt_imgs_path):
            shutil.rmtree(ckpt_imgs_path)
        os.mkdir(ckpt_imgs_path)
    alphas = []
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps // config.sampling.T_sampling

        x0_preds = []
        k_avg = False
        if isinstance(x, list):
            x = torch.cat(x, dim=0)

        xs = [x.to('cuda')]

        # generate time schedule
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        percentile_to_mean = get_percentile_index(time_pairs, percentile=100 * per)
        if face_bbox:
            # get the dimensions of the bounding box
            face_bbox_width, face_bbox_height = face_bbox[2], face_bbox[3]
            max_variance = face_bbox_width * face_bbox_height
            min_variance = 0.01 * max_variance
            # Generate the list of variances corresponding to the number of steps
            num_steps = len(time_pairs)
            variances = np.linspace(min_variance, max_variance, num_steps)
            sampled_variance = np.random.choice(variances)

        # reverse diffusion sampling
        total_steps = len(time_pairs)
        mean_step = 0
        for step_idx, (i, j) in tqdm(enumerate(time_pairs), total=len(time_pairs)):
            xt = xs[-1].to(x.device)
            n = xt.size(0)
            i, j = i * skip, j * skip
            if j < 0: j = -1

            if j < i:  # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                if k_avg:
                    mean_step += 1
                    next_t = (torch.ones(n + 1) * j).to(x.device)
                else:
                    next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                alphas.append(at[0].item())
                at_next = compute_alpha(b, next_t.long())

                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # estimate x0
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                if k_avg:
                    # Compute the mean along the batch dimension (dim=0)
                    mean_tensor_x_0 = x0_t.mean(dim=0, keepdim=True)  # Keep the batch dimension
                    # Concatenate along the batch dimension
                    x0_t = torch.cat([x0_t, mean_tensor_x_0], dim=0)
                    n = n + 1

                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path, 'x0_t')
                    for b_idx in range(x0_t.size(0)):
                        save_img(i, path_for_img, config, x0_t[b_idx].unsqueeze(0).clone(), b_idx)
                if sigma_y == 0.:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = (at_next) ** args.gamma
                    weight_noise_t = delta_t

                eta_reg = max(1e-4, sigma_y ** 2 * args.eta_tilde)
                if args.eta_tilde < 0:
                    eta_reg = 1e-4 + args.xi * (sigma_y * 255.0) ** 2

                scale_gLS = args.scale_ls  # e.g. <= 1/A_funcs.singulars().max()**2

                guidance_BP = A_funcs.A_pinv_add_eta(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1), eta_reg
                ).reshape(*x0_t.size())
                guidance_LS = A_funcs.At(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                # Handle kernels if provided
                if face_bbox is not None and kernels is not None:
                    kernel_tensors = []
                    for name, kernel_source in kernels.items():
                        if callable(kernel_source):
                            # Determine variance to pass to kernel function
                            if 'variance' in kernel_source.__code__.co_varnames:
                                current_variance = variances[step_idx]
                                kernel_np = kernel_source((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                            else:
                                # If variance not in function signature, pass sampled_variance or skip
                                kernel_np = kernel_source((x0_t.size(2), x0_t.size(3)), face_bbox, sampled_variance)
                        else:
                            kernel_np = kernel_source

                        kernel_tensor = torch.tensor(kernel_np, dtype=torch.float32).to(x0_t.device)
                        if kernel_tensor.dim() == 2:
                            kernel_tensor = kernel_tensor.unsqueeze(0)  # add channel dim
                        kernel_tensors.append(kernel_tensor)

                        if ckpt_imgs_path is not None:
                            path_for_img = os.path.join(ckpt_imgs_path, f'{name}_kernel_heatmap')
                            if not os.path.exists(path_for_img):
                                os.mkdir(path_for_img)
                            save_gaussian_heatmap(kernel_np, os.path.join(path_for_img, f'{i}.png'))

                    batch_kernel_tensor = torch.stack(kernel_tensors, dim=0)  # (num_kernels, 1, H, W)

                    # Combine kernels as average (you can modify this logic)
                    combined_kernel = batch_kernel_tensor.mean(dim=0)  # (1, H, W)
                else:
                    combined_kernel = None

                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path, 'guidance_BP')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    for b_idx in range(guidance_BP.size(0)):
                        save_heatmap(
                            inverse_data_transform(config, guidance_BP[b_idx].unsqueeze(0).clone()).to('cpu').squeeze().permute(1, 2, 0),
                            os.path.join(path_for_img, f"{i}_b_{b_idx}.png"),
                        )

                    # Normalize guidance_LS to have the same range as guidance_BP
                    min_BP, max_BP = guidance_BP.min(), guidance_BP.max()
                    min_LS, max_LS = guidance_LS.min(), guidance_LS.max()
                    guidance_LS_normalized = (guidance_LS - min_LS) / (max_LS - min_LS)
                    guidance_LS_standardized = guidance_LS_normalized * (max_BP - min_BP) + min_BP
                    path_for_img = os.path.join(ckpt_imgs_path, 'guidance_LS')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    for b_idx in range(guidance_LS_standardized.size(0)):
                        save_heatmap(
                            inverse_data_transform(config, guidance_LS_standardized[b_idx].unsqueeze(0).clone()).to('cpu').squeeze().permute(1, 2, 0),
                            os.path.join(path_for_img, f"{i}_b_{b_idx}.png"),
                        )

                if k_avg:
                    at = compute_alpha(b, (torch.ones(n) * i).to(x.device).long()).to(x.device)
                if args.step_size_mode == 0:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = 1
                elif args.step_size_mode == 1:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = (1 - at_next) / (1 - at)
                elif args.step_size_mode == 2:
                    step_size_LS = (1 - at_next) / (1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"

                if deid and face_bbox is not None and combined_kernel is not None and not diff_priv:
                    guidance_BP = guidance_BP * combined_kernel
                    guidance_LS = guidance_LS * combined_kernel
                    xt_next_tilde = x0_t - step_size * (
                        step_size_BP * (1 - delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS
                    )
                elif deid and face_bbox and diff_priv and combined_kernel is None:
                    epsilon = 1.25
                    guidance_BP_obfuscated = obfuscate_tensor_region(guidance_BP, epsilon, face_bbox)
                    guidance_LS_obfuscated = obfuscate_tensor_region(guidance_LS, epsilon, face_bbox)
                    xt_next_tilde = x0_t - step_size * (
                        step_size_BP * (1 - delta_t) * guidance_BP_obfuscated + step_size_LS * delta_t * scale_gLS * guidance_LS_obfuscated
                    )
                elif deid and face_bbox is not None and combined_kernel is not None and diff_priv:
                    epsilon = 1.25
                    guidance_BP_obfuscated = obfuscate_tensor_region(guidance_BP, epsilon, face_bbox) * combined_kernel
                    guidance_LS_obfuscated = obfuscate_tensor_region(guidance_LS, epsilon, face_bbox) * combined_kernel
                    xt_next_tilde = x0_t - step_size * (
                        step_size_BP * (1 - delta_t) * guidance_BP_obfuscated + step_size_LS * delta_t * scale_gLS * guidance_LS_obfuscated
                    )
                else:
                    xt_next_tilde = x0_t - step_size * (
                        step_size_BP * (1 - delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS
                    )

                xs.append(xt_next_tilde)

        return xs, alphas



















def ddpg_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None,
                   deid=False,ckpt_imgs_path=None,face_bbox = None, gaussian_kern=False, diff_priv=False, per = 1):
    if ckpt_imgs_path is not None:
        if os.path.exists(ckpt_imgs_path):
            shutil.rmtree(ckpt_imgs_path)
        os.mkdir(ckpt_imgs_path)
    alphas = []
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.sampling.T_sampling
        
        x0_preds = []
        k_avg = False
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        
        xs = [x.to('cuda')]   
        

        # generate time schedule
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))
        percentile_to_mean = get_percentile_index(time_pairs, percentile=100*per)        
        if face_bbox:
            # get the dimensions of the bounding box
            face_bbox_width, face_bbox_height = face_bbox[2], face_bbox[3]
            max_variance = face_bbox_width * face_bbox_height
            min_variance = 0.01 * max_variance
            # Generate the list of variances corresponding to the number of steps
            num_steps = len(time_pairs)
            variances = np.linspace(min_variance, max_variance, num_steps)
            sampled_variance = np.random.choice(variances)
        # reverse diffusion sampling
        total_steps = len(time_pairs)
        mean_step = 0
        for step_idx, (i, j) in tqdm(enumerate(time_pairs), total=len(time_pairs)):
            xt = xs[-1].to(x.device)
            n = xt.size(0)
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                if k_avg:
                    mean_step += 1
                    next_t = (torch.ones(n+1) * j).to(x.device)
                else:
                    next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                alphas.append(at[0].item())
                at_next = compute_alpha(b, next_t.long())
                
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # estimate x0
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                if k_avg:
                    # Compute the mean along the batch dimension (dim=0)
                    mean_tensor_x_0 = x0_t.mean(dim=0, keepdim=True)  # Keep the batch dimension
                    # Concatenate along the batch dimension
                    x0_t = torch.cat([x0_t, mean_tensor_x_0], dim=0)
                    n = n+1
                '''
                x0_t:
                This variable corresponds to the estimation of the original image (x0) 
                given the current noisy image (xt) at time t. It follows equation (16) 
                in the algorithm
                This matches the equation, where et is the predicted noise
                '''
                    
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'x0_t')
                    for b_idx in range(x0_t.size(0)):
                        save_img(i,path_for_img,config,x0_t[b_idx].unsqueeze(0).clone(),b_idx)
                if sigma_y==0.:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = (at_next) ** args.gamma
                    weight_noise_t = delta_t


                eta_reg = max(1e-4, sigma_y**2 * args.eta_tilde )
                if args.eta_tilde < 0:
                    eta_reg = 1e-4 + args.xi * (sigma_y*255.0)**2

                scale_gLS = args.scale_ls  #e.g. <= 1/A_funcs.singulars().max()**2 

                guidance_BP = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1), eta_reg).reshape(*x0_t.size())
                guidance_LS = A_funcs.At(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)).reshape(*x0_t.size())
                '''
                        guidance_BP: This variable refers to the back-projection (BP) guidance 
                        term (g_BP) used to correct the estimate based on the observed measurements. 
                        According to the algorithm:
                        gBP=A^T(AA^T+ηIm)^−1(Ax_{0∣t}−y)
                        In the code, it is computed using a function (A_funcs.A_pinv_add_eta) that likely 
                        implements the pseudoinverse calculation with regularization (η). 
                        It projects the difference between the observation (y) and the 
                        current estimate (A(x0_t)).

                        guidance_LS: This variable corresponds to the least squares (LS) guidance 
                        (g_LS) term, used as an alternative or complement to guidance_BP. 
                        The algorithm defines it as:

                        gLS=cA^T(Ax_{0∣t}−y)
                        This indicates that it's the gradient of the least squares loss, 
                        scaled by a constant c
                '''
                if face_bbox is not None and gaussian_kern:
                    # Retrieve the variance for the current step
                    current_variance = variances[step_idx]
                    # Create the Gaussian kernel based on the bounding box and the calculated variance
                    gaussian_kernel = create_gaussian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                    if ckpt_imgs_path is not None:
                        path_for_img = os.path.join(ckpt_imgs_path,'gaussian_kernel_heatmap')
                        if not os.path.exists(path_for_img):
                            os.mkdir(path_for_img)
                        # Save the Gaussian kernel as a heatmap image
                        save_gaussian_heatmap(gaussian_kernel, os.path.join(path_for_img,f'{i}.png'))
                    laplacian_kernel = create_laplacian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                    if ckpt_imgs_path is not None:
                        path_for_img = os.path.join(ckpt_imgs_path,'laplacian_kernel_heatmap')
                        if not os.path.exists(path_for_img):
                            os.mkdir(path_for_img)
                        # Save the Gaussian kernel as a heatmap image
                        save_gaussian_heatmap(laplacian_kernel, os.path.join(path_for_img,f'{i}.png'))
                    
                    #zerokernel = create_zero_kernel((x0_t.size(2), x0_t.size(3)), face_bbox)
                    gaussian_kernel_fixed = create_gaussian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, sampled_variance)
                    if ckpt_imgs_path is not None and n==3:
                        path_for_img = os.path.join(ckpt_imgs_path,f'gaussian_kernel_fixed_{str(sampled_variance)}_heatmap')
                        if not os.path.exists(path_for_img):
                            os.mkdir(path_for_img)
                        # Save the Gaussian kernel as a heatmap image
                        save_gaussian_heatmap(gaussian_kernel_fixed, os.path.join(path_for_img,f'{i}.png'))
                    gaussian_kernel_fixed_tensor = torch.tensor(gaussian_kernel_fixed, dtype=torch.float32).to(x0_t.device)
                    zerokernel = create_zero_kernel((x0_t.size(2), x0_t.size(3)), face_bbox)#np.ones((x0_t.size(2), x0_t.size(3)), dtype=np.float32)
                    if ckpt_imgs_path is not None:
                        path_for_img = os.path.join(ckpt_imgs_path,f'zerokernel_fixed_heatmap')
                        if not os.path.exists(path_for_img):
                            os.mkdir(path_for_img)
                        # Save the Gaussian kernel as a heatmap image
                        save_gaussian_heatmap(zerokernel, os.path.join(path_for_img,f'{i}.png'))
                    zerokernel_tensor = torch.tensor(zerokernel, dtype=torch.float32).to(x0_t.device)
                    # Convert the Gaussian kernel to a tensor and move it to the device
                    gaussian_kernel_tensor = torch.tensor(gaussian_kernel, dtype=torch.float32).to(x0_t.device)
                    laplacian_kernel_tensor = torch.tensor(laplacian_kernel, dtype=torch.float32).to(x0_t.device)
                    
                    if n==3:
                        # Ensure each kernel has shape (1, 256, 256) by adding a channel dimension
                        gaussian_kernel_tensor = gaussian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        laplacian_kernel_tensor = laplacian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        zerokernel_tensor = zerokernel_tensor.unsqueeze(0)

                        # Stack along the batch dimension (B=3)
                        batch_kernel_tensor = torch.stack([
                            gaussian_kernel_tensor, 
                            laplacian_kernel_tensor, 
                            zerokernel_tensor
                        ], dim=0)  # Shape: (3, 1, 256, 256)
                    elif n==2 and not k_avg:
                        # Ensure each kernel has shape (1, 256, 256) by adding a channel dimension
                        gaussian_kernel_tensor = gaussian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        laplacian_kernel_tensor = laplacian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        # Stack along the batch dimension (B=3)
                        batch_kernel_tensor = torch.stack([
                            gaussian_kernel_tensor, 
                            laplacian_kernel_tensor
                        ], dim=0)  # Shape: (3, 1, 256, 256)
                    else:
                        batch_kernel_tensor = gaussian_kernel_tensor
                
                
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'guidance_BP')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    for b_idx in  range(guidance_BP.size(0)):
                        save_heatmap(inverse_data_transform(config, guidance_BP[b_idx].unsqueeze(0).clone()).to('cpu').squeeze().permute(1, 2, 0), 
                                    os.path.join(path_for_img,f"{i}_b_{b_idx}.png"))

                    # Normalize guidance_LS to have the same range as guidance_BP
                    min_BP, max_BP = guidance_BP.min(), guidance_BP.max()
                    min_LS, max_LS = guidance_LS.min(), guidance_LS.max()
                    # Standardize guidance_LS to the range of guidance_BP
                    guidance_LS_normalized = (guidance_LS - min_LS) / (max_LS - min_LS)  # Normalize guidance_LS to [0, 1]
                    guidance_LS_standardized = guidance_LS_normalized * (max_BP - min_BP) + min_BP  # Scale to [min_BP, max_BP]
                    path_for_img = os.path.join(ckpt_imgs_path,'guidance_LS')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    for b_idx in  range(guidance_LS_standardized.size(0)):
                        save_heatmap(inverse_data_transform(config, 
                                                            guidance_LS_standardized[b_idx].unsqueeze(0).clone()).to('cpu').squeeze().permute(1, 2, 0), 
                                    os.path.join(path_for_img,f"{i}_b_{b_idx}.png"))
                if k_avg:
                    at = compute_alpha(b, (torch.ones(n) * i).to(x.device).long()).to(x.device)
                if args.step_size_mode==0:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = 1
                elif args.step_size_mode==1:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = (1 - at_next)/(1 - at)
                elif args.step_size_mode==2:
                    step_size_LS = (1 - at_next)/(1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"
                    
                if deid and face_bbox is not None and gaussian_kern and not diff_priv:
                    guidance_BP = guidance_BP*batch_kernel_tensor
                    guidance_LS = guidance_LS*batch_kernel_tensor
                    # data fidelity guidance
                    xt_next_tilde = x0_t - step_size * ( step_size_BP * (1-delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS )
                elif deid and face_bbox and diff_priv and not gaussian_kern:
                    # Set epsilon for privacy
                    epsilon = 1.25  # Adjust as needed

                    # Obfuscate guidance_BP and guidance_LS
                    guidance_BP_obfuscated = obfuscate_tensor_region(guidance_BP, epsilon,face_bbox)
                    guidance_LS_obfuscated = obfuscate_tensor_region(guidance_LS, epsilon,face_bbox)
                    # data fidelity guidance
                    xt_next_tilde = x0_t - step_size * ( step_size_BP * (1-delta_t) * guidance_BP_obfuscated + step_size_LS * delta_t * scale_gLS * guidance_LS_obfuscated )
                elif deid and face_bbox is not None and gaussian_kern and diff_priv:
                    # Set epsilon for privacy
                    epsilon = 1.25  # Adjust as needed

                    # Obfuscate guidance_BP and guidance_LS
                    guidance_BP_obfuscated = obfuscate_tensor_region(guidance_BP, epsilon,face_bbox)*batch_kernel_tensor
                    guidance_LS_obfuscated = obfuscate_tensor_region(guidance_LS, epsilon,face_bbox)*batch_kernel_tensor
                    # data fidelity guidance
                    xt_next_tilde = x0_t - step_size * ( step_size_BP * (1-delta_t) * guidance_BP_obfuscated + step_size_LS * delta_t * scale_gLS * guidance_LS_obfuscated )
                else:
                    # data fidelity guidance
                    xt_next_tilde = x0_t - step_size * ( step_size_BP * (1-delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS )
                '''
                    xt_next_tilde: This is the intermediate state before noise is reintroduced 
                    in the next step of the diffusion process. It is calculated using a 
                    combination of the BP and LS guidance terms:
                    x~t−1=x_{0∣t}−μt((1−δt)gBP+δtgLS)
                    This combines the BP and LS terms according to the weighting delta_t and 
                    scaling factors for the guidance terms.
                '''
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'xt_next_tilde')
                    for b_idx in  range(xt_next_tilde.size(0)):
                        save_img(i,path_for_img,config,xt_next_tilde[b_idx].unsqueeze(0).clone(),b_idx)
                # compute effective noise
                if k_avg:
                    # Extract the last element (mean tensor we added)
                    xt_next_tilde_mean = xt_next_tilde[-1].unsqueeze(0).clone()  # or x_augmented[3] if batch size was originally 3
                    
                    xt_next_tilde = xt_next_tilde[:-1]
                    #revert it to the original shape
                    x0_t = x0_t[:-1]
                    next_t = next_t[:-1]#(torch.ones(n) * j).to(x.device)
                    at_next = at_next[:-1]#compute_alpha(b, next_t.long())
                    at = at[:-1]#compute_alpha(b, (torch.ones(n) * i).to(x.device).long())
                    weight_noise_t = weight_noise_t[:-1]
                et_hat = ( xt - at.sqrt() * xt_next_tilde ) / (1 - at).sqrt()

                c1 = 0
                c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1-zeta)  * weight_noise_t
                et_gauss = torch.randn_like(x0_t)
                xt_next = at_next.sqrt() * xt_next_tilde + c1 * et_gauss + c2 * et_hat

                if k_avg:
                    # Compute the mean along the batch dimension (dim=0)
                    et_hat_avg = et_hat.mean(dim=0, keepdim=True)  # Keep the batch dimension
                    et_gauss_mean = torch.randn_like(mean_tensor_x_0)
                    at_next_mean = at_next[-1].sqrt().unsqueeze(0)
                    xt_next_mean =  at_next_mean * xt_next_tilde_mean + c1[-1].unsqueeze(0) * et_gauss_mean + c2[-1].unsqueeze(0) * et_hat_avg
                    xt_next = torch.cat([xt_next, xt_next_mean], dim=0)

                '''
                xt_next:
                        This is the final state of the image after adding noise to the 
                        intermediate state (xt_next_tilde). It represents the next 
                        iteration’s state (xt-1).
                '''
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'xt_next')
                    for b_idx in  range(xt_next.size(0)):
                        save_img(i,path_for_img,config,xt_next[b_idx].unsqueeze(0).clone(),b_idx)

                if ckpt_imgs_path is not None and k_avg:
                    path_for_img = os.path.join(ckpt_imgs_path,'xt_next_mean')
                    for b_idx in  range(xt_next_mean.size(0)):
                        save_img(i,path_for_img,config,xt_next_mean[b_idx].unsqueeze(0).clone(),b_idx)
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

                if (step_idx == percentile_to_mean) and deid:
                    k_avg = True
                if mean_step ==1:
                    k_avg = False
                    mean_step = 0

            else: 
                assert 1, "Unexpected case"
        
        if sigma_y != 0.:  # if there is noise, take the denoised result
            xs.append(x0_t.to('cpu'))
    if ckpt_imgs_path is not None:
        plot_alphas(alphas,os.path.join(ckpt_imgs_path,'alphas_plot.pdf'))

    return [xs[-1]], [x0_preds[-1]]


# code from RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)

        assert t <= T_sampling, (t, T_sampling)
