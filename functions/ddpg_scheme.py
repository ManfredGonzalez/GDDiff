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

def create_gaussian_noise_kernel(shape, bbox, variance):
    """
    Creates a Gaussian noise kernel with the specified variance within the bbox,
    and sets the area outside the bbox to ones.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :param variance: Variance of the Gaussian noise.
    :return: Noise kernel of shape (H, W).
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)

    # Generate Gaussian noise within the bounding box
    noise = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=(y_end - y_start, x_end - x_start))

    # Place the noise in the bounding box area of the kernel
    kernel[y_start:y_end, x_start:x_end] = noise

    return kernel
    
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

def create_binary_gaussian_kernel(shape, bbox, variance, threshold=0.5):
    """
    Creates a binary Gaussian kernel: values >= threshold are 1, others 0.
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    gaussian = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance))
    binary = (gaussian >= threshold).astype(np.float32)

    kernel[y_start:y_end, x_start:x_end] = binary
    return kernel

def create_log_gaussian_kernel(shape, bbox, variance):
    """
    Creates a log-Gaussian kernel: log of Gaussian values normalized to [0, 1].
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    gaussian = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance))
    log_gaussian = np.log(gaussian + 1e-6)  # Avoid log(0)
    normalized = (log_gaussian - log_gaussian.min()) / (log_gaussian.max() - log_gaussian.min())

    kernel[y_start:y_end, x_start:x_end] = normalized
    return kernel


def create_flat_top_gaussian_kernel(shape, bbox, variance, clip_value=0.8):
    """
    Creates a Gaussian kernel with a flat top (values clipped to a max).
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    gaussian = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance))
    flat_top = np.clip(gaussian, clip_value, 1.0)

    kernel[y_start:y_end, x_start:x_end] = flat_top
    return kernel


def create_double_gaussian_kernel(shape, bbox, variance1, variance2, weight=0.5):
    """
    Combines two Gaussians: wide and narrow. Useful for emphasizing central peak with context.
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    gaussian1 = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance1))
    gaussian2 = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * variance2))
    combined = weight * gaussian1 + (1 - weight) * gaussian2

    kernel[y_start:y_end, x_start:x_end] = combined
    return kernel


def create_gabor_kernel(shape, bbox, variance, frequency=0.2, theta=0):
    """
    Creates a Gabor kernel in the bbox area, ones outside.
    """
    import numpy as np

    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    x_shifted = x - x_center
    y_shifted = y - y_center

    # Rotate the coordinates
    x_theta = x_shifted * np.cos(theta) + y_shifted * np.sin(theta)
    y_theta = -x_shifted * np.sin(theta) + y_shifted * np.cos(theta)

    gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * variance))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta)

    gabor = gaussian * sinusoid
    kernel[y_start:y_end, x_start:x_end] = gabor

    return kernel


def create_gaussian_polynomial_kernel(shape, bbox, variance, a=1.0, degree=2):
    """
    Gaussian * polynomial kernel applied in bbox, ones elsewhere.
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]

    dist_sq = (x - x_center) ** 2 + (y - y_center) ** 2
    gaussian = np.exp(-dist_sq / (2 * variance))
    polynomial = (a * dist_sq + 1) ** degree

    combined = gaussian * polynomial
    kernel[y_start:y_end, x_start:x_end] = combined

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

def create_unsharp_masking_kernel(shape, bbox, amount):
    """
    Creates an Unsharp Masking kernel centered on the bbox,
    applied only within the bbox area, and sets the area outside the bbox to ones.
    Unsharp masking enhances edges by subtracting a blurred version of the image.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :param amount: Weight of the sharpening effect.
    :return: Unsharp Masking kernel of shape (H, W).
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)

    # Create a basic unsharp mask kernel (sharpening filter)
    # Kernel: (1 + amount) * Identity - amount * Gaussian Blur
    # Using a fixed small Gaussian for simplicity (3x3)
    blur_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32)
    blur_kernel /= blur_kernel.sum()

    # Unsharp kernel is delta kernel minus scaled blur kernel
    unsharp_kernel = (1 + amount) * np.eye(3)[1][np.newaxis, :]  # Center pixel
    unsharp_kernel = np.array([[0, -amount, 0],
                               [-amount, 1 + 4 * amount, -amount],
                               [0, -amount, 0]], dtype=np.float32)

    # Apply the unsharp kernel only within the bbox
    for i in range(y_start + 1, y_end - 1):
        for j in range(x_start + 1, x_end - 1):
            kernel[i-1:i+2, j-1:j+2] = unsharp_kernel

    return kernel


def create_motion_blur_kernel(shape, bbox, length):
    """
    Creates a Motion Blur kernel applied within the bbox area,
    simulating horizontal motion blur. The area outside the bbox is set to ones.
    :param shape: Tuple (H, W) for the image dimensions.
    :param bbox: Tuple (x, y, w, h) representing the bounding box.
    :param length: Length of the motion blur (must be odd).
    :return: Motion Blur kernel of shape (H, W).
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize the kernel with ones
    kernel = np.ones((H, W), dtype=np.float32)

    # Create a horizontal motion blur kernel of given length
    if length % 2 == 0:
        length += 1  # Ensure odd length

    motion_kernel = np.zeros((1, length), dtype=np.float32)
    motion_kernel[0] = 1.0 / length

    # Apply the motion blur kernel inside the bbox
    for i in range(y_start, y_end):
        for j in range(x_start + length // 2, x_end - length // 2):
            kernel[i, j - length // 2:j + length // 2 + 1] = motion_kernel

    return kernel




def create_epanechnikov_kernel(shape, bbox, radius):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance_squared = (x - x_center) ** 2 + (y - y_center) ** 2
    mask = distance_squared <= radius ** 2
    epanechnikov = np.zeros_like(distance_squared, dtype=np.float32)
    epanechnikov[mask] = 1 - (distance_squared[mask] / radius ** 2)
    inverted = 1 - epanechnikov
    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel


def create_triangular_kernel(shape, bbox, radius):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    triangular = np.clip(1 - (distance / radius), 0, 1)
    inverted = 1 - triangular
    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

def create_exponential_kernel(shape, bbox, scale):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    exponential = np.exp(-distance / scale)
    inverted = 1 - exponential
    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel


def create_cauchy_kernel(shape, bbox, scale):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance_squared = (x - x_center) ** 2 + (y - y_center) ** 2
    cauchy = 1 / (1 + (distance_squared / (scale ** 2)))
    inverted = 1 - cauchy
    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel


def create_cosine_kernel(shape, bbox, radius):
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    cosine = np.zeros_like(distance)
    mask = distance <= radius
    cosine[mask] = 0.5 * (1 + np.cos(np.pi * distance[mask] / radius))
    inverted = 1 - cosine
    kernel[y_start:y_end, x_start:x_end] = inverted
    return kernel

def create_blur_kernel(shape, bbox):
    """
    Applies a uniform blur (box filter) within the bounding box.
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    kernel = np.ones((H, W), dtype=np.float32)
    h, w = y_end - y_start, x_end - x_start

    # Box filter (same value everywhere inside bbox)
    blur_value = 0.5  # You can adjust this
    kernel[y_start:y_end, x_start:x_end] = blur_value
    return kernel

def create_fog_kernel(shape, bbox, scale=20.0):
    """
    Creates a fog-like kernel: smooth, gradual increase from the center of the bbox
    to the edges, simulating fog density.
    
    Parameters:
        shape : tuple
            (H, W) size of the kernel/image.
        bbox : list or tuple
            Bounding box [x, y, w, h].
        scale : float
            Controls how fast the fog density decays from the center. Larger = slower decay.
            
    Returns:
        kernel : np.ndarray
            2D kernel of shape (H, W) with fog pattern applied inside bbox.
    """
    H, W = shape
    x_center = bbox[0] + bbox[2] // 2
    y_center = bbox[1] + bbox[3] // 2
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]

    # Initialize full kernel to 1 (no fog)
    kernel = np.ones((H, W), dtype=np.float32)

    # Coordinate grid within bbox
    y, x = np.ogrid[y_start:y_end, x_start:x_end]
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Fog effect: smooth exponential decay from center
    fog = 1 - np.exp(-distance / scale)

    # Insert fog into kernel
    kernel[y_start:y_end, x_start:x_end] = fog
    return kernel


def create_salt_and_pepper_kernel(shape, bbox, amount=0.05):
    """
    Applies salt-and-pepper noise (0 or 1) randomly within the bounding box.
    """
    H, W = shape
    x_start, x_end = bbox[0], bbox[0] + bbox[2]
    y_start, y_end = bbox[1], bbox[1] + bbox[3]
    h, w = y_end - y_start, x_end - x_start

    kernel = np.ones((H, W), dtype=np.float32)

    # Start with a region of ones
    region = np.ones((h, w), dtype=np.float32)

    # Number of pixels to corrupt
    num_pixels = int(amount * h * w)
    coords = np.random.choice(h * w, num_pixels, replace=False)

    # Randomly choose 0 or 1 to apply
    salt_pepper = np.random.choice([0.0, 1.0], size=num_pixels)

    # Flatten, apply noise, and reshape
    flat_region = region.flatten()
    flat_region[coords] = salt_pepper
    region_noised = flat_region.reshape(h, w)

    kernel[y_start:y_end, x_start:x_end] = region_noised
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

                    # ---------------------------------------------
                    # Create new kernels
                    # ---------------------------------------------
                    epanechnikov_kernel = create_epanechnikov_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, 20)
                    triangular_kernel = create_triangular_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, 25)
                    exponential_kernel = create_exponential_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, 10)
                    cauchy_kernel = create_cauchy_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, 15)
                    cosine_kernel = create_cosine_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, 20)
                    unsharp_masking_kernel = create_unsharp_masking_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, amount=1.0)
                    motion_blur_kernel = create_motion_blur_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, length=15)
                    salt_and_pepper_kernel = create_salt_and_pepper_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, amount=0.9)
                    fog_kernel = create_fog_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, scale=50)

                    polynomial_kernel = create_gaussian_polynomial_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                    gabor_kernel = create_gabor_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                    double_gaussian_kernel = create_double_gaussian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance, current_variance/2)
                    log_gaussian_kernel = create_log_gaussian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)


                    epanechnikov_kernel_tensor = torch.tensor(epanechnikov_kernel, dtype=torch.float32).to(x0_t.device)
                    triangular_kernel_tensor = torch.tensor(triangular_kernel, dtype=torch.float32).to(x0_t.device)
                    exponential_kernel_tensor = torch.tensor(exponential_kernel, dtype=torch.float32).to(x0_t.device)
                    cauchy_kernel_tensor = torch.tensor(cauchy_kernel, dtype=torch.float32).to(x0_t.device)
                    cosine_kernel_tensor = torch.tensor(cosine_kernel, dtype=torch.float32).to(x0_t.device)
                    unsharp_masking_kernel_tensor = torch.tensor(unsharp_masking_kernel, dtype=torch.float32).to(x0_t.device)
                    motion_blur_kernel_tensor = torch.tensor(motion_blur_kernel, dtype=torch.float32).to(x0_t.device)
                    salt_and_pepper_kernel_tensor = torch.tensor(salt_and_pepper_kernel, dtype=torch.float32).to(x0_t.device)
                    fog_kernel_tensor = torch.tensor(fog_kernel, dtype=torch.float32).to(x0_t.device)

                    polynomial_kernel_tensor = torch.tensor(polynomial_kernel, dtype=torch.float32).to(x0_t.device)
                    gabor_kernel_tensor = torch.tensor(gabor_kernel, dtype=torch.float32).to(x0_t.device)
                    double_gaussian_kernel_tensor = torch.tensor(double_gaussian_kernel, dtype=torch.float32).to(x0_t.device)
                    log_gaussian_kernel_tensor = torch.tensor(log_gaussian_kernel, dtype=torch.float32).to(x0_t.device)

                    #epanechnikov_kernel_tensor = epanechnikov_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #triangular_kernel_tensor = triangular_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #exponential_kernel_tensor = exponential_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #cauchy_kernel_tensor = cauchy_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #cosine_kernel_tensor = cosine_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    # ---------------------------------------------


                    # Convert the Gaussian kernel to a tensor and move it to the device
                    gaussian_kernel_tensor = torch.tensor(gaussian_kernel, dtype=torch.float32).to(x0_t.device)
                    laplacian_kernel_tensor = torch.tensor(laplacian_kernel, dtype=torch.float32).to(x0_t.device)
                    
                    #if n==3:
                        # Ensure each kernel has shape (1, 256, 256) by adding a channel dimension
                    #    gaussian_kernel_tensor = gaussian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #    laplacian_kernel_tensor = laplacian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                    #    zerokernel_tensor = zerokernel_tensor.unsqueeze(0)

                        # Stack along the batch dimension (B=3)
                    #    batch_kernel_tensor = torch.stack([
                    #        gaussian_kernel_tensor, 
                    #        laplacian_kernel_tensor, 
                    #        zerokernel_tensor
                    #    ], dim=0)  # Shape: (3, 1, 256, 256)

                    if n==3:
                        #print("Number channels n=3 -----------------------------------------------")
                        # Ensure each kernel has shape (1, 256, 256) by adding a channel dimension
                        gaussian_kernel_tensor = gaussian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        laplacian_kernel_tensor = laplacian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        zerokernel_tensor = zerokernel_tensor.unsqueeze(0) 
                        epanechnikov_kernel_tensor = epanechnikov_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        triangular_kernel_tensor = triangular_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        exponential_kernel_tensor = exponential_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        cauchy_kernel_tensor = cauchy_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        cosine_kernel_tensor = cosine_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        unsharp_masking_kernel_tensor = unsharp_masking_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        motion_blur_kernel_tensor = motion_blur_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        salt_and_pepper_kernel_tensor = salt_and_pepper_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        fog_kernel_tensor = fog_kernel_tensor.unsqueeze(0) 
                        
                        polynomial_kernel_tensor = polynomial_kernel_tensor.unsqueeze(0)
                        gabor_kernel_tensor = gabor_kernel_tensor.unsqueeze(0)
                        double_gaussian_kernel_tensor = double_gaussian_kernel_tensor.unsqueeze(0)
                        log_gaussian_kernel_tensor = log_gaussian_kernel_tensor.unsqueeze(0)

                        # Stack along the batch dimension (B=3)
                        batch_kernel_tensor = torch.stack([
                            gaussian_kernel_tensor, 
                            #laplacian_kernel_tensor, 
                            #epanechnikov_kernel_tensor,
                            #triangular_kernel_tensor,
                            #exponential_kernel_tensor,
                            #cauchy_kernel_tensor,
                            #cosine_kernel_tensor,
                            #unsharp_masking_kernel_tensor,
                            #motion_blur_kernel_tensor,
                            salt_and_pepper_kernel_tensor,
                            #fog_kernel_tensor,
                            #polynomial_kernel_tensor,
                            #gabor_kernel_tensor,
                            #double_gaussian_kernel_tensor,
                            #log_gaussian_kernel_tensor,

                            zerokernel_tensor,
                        ], dim=0)  # Shape: (3, 1, 256, 256)




                    elif n==2 and not k_avg:
                        #print("Number channels n=2 -----------------------------------------------")
                        # Ensure each kernel has shape (1, 256, 256) by adding a channel dimension
                        gaussian_kernel_tensor = gaussian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        laplacian_kernel_tensor = laplacian_kernel_tensor.unsqueeze(0)  # (1, 256, 256)

                        epanechnikov_kernel_tensor = epanechnikov_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        triangular_kernel_tensor = triangular_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        exponential_kernel_tensor = exponential_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        cauchy_kernel_tensor = cauchy_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        cosine_kernel_tensor = cosine_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        unsharp_masking_kernel_tensor = unsharp_masking_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        motion_blur_kernel_tensor = motion_blur_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        salt_and_pepper_kernel_tensor = salt_and_pepper_kernel_tensor.unsqueeze(0)  # (1, 256, 256)
                        fog_kernel_tensor = fog_kernel_tensor.unsqueeze(0) 

                        polynomial_kernel_tensor = polynomial_kernel_tensor.unsqueeze(0)
                        gabor_kernel_tensor = gabor_kernel_tensor.unsqueeze(0)
                        double_gaussian_kernel_tensor = double_gaussian_kernel_tensor.unsqueeze(0)
                        log_gaussian_kernel_tensor = log_gaussian_kernel_tensor.unsqueeze(0)

                        # Stack along the batch dimension (B=3)
                        batch_kernel_tensor = torch.stack([
                            gaussian_kernel_tensor, 
                            #laplacian_kernel_tensor,
                            #laplacian_kernel_tensor, 
                            #epanechnikov_kernel_tensor,
                            #triangular_kernel_tensor,
                            #exponential_kernel_tensor,
                            #cauchy_kernel_tensor,
                            #cosine_kernel_tensor,
                            #unsharp_masking_kernel_tensor,
                            #motion_blur_kernel_tensor,
                            salt_and_pepper_kernel_tensor,
                            #fog_kernel_tensor
                            #polynomial_kernel_tensor,
                            #gabor_kernel_tensor,
                            #double_gaussian_kernel_tensor,
                            #log_gaussian_kernel_tensor,

                        ], dim=0)  # Shape: (3, 1, 256, 256)
                    else:
                        print("Else --------------------------------------")
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

