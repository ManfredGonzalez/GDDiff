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

def save_img(i,path,config,img):
    if not os.path.exists(path):
        os.mkdir(path)
    tvu.save_image(
        inverse_data_transform(config, img), 
        os.path.join(path,f'{i}.png')
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
def ddpg_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None,
                   deid=False,ckpt_imgs_path=None,face_detector=None, detections=None):
    if ckpt_imgs_path is not None:
        if os.path.exists(ckpt_imgs_path):
            shutil.rmtree(ckpt_imgs_path)
        os.mkdir(ckpt_imgs_path)
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.sampling.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))        
        
        bbox_found = False
        # reverse diffusion sampling
        for step_idx, (i, j) in tqdm(enumerate(time_pairs), total=len(time_pairs)):

            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 

                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
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
                '''
                x0_t:
                This variable corresponds to the estimation of the original image (x0) 
                given the current noisy image (xt) at time t. It follows equation (16) 
                in the algorithm
                This matches the equation, where et is the predicted noise
                '''

                # get the coordinates of the face from the x0_t image
                if deid and not bbox_found:
                    # Convert the RGB image obtained from inverse_data_transform
                    rgb_image = inverse_data_transform(config, x0_t).to('cpu').squeeze().permute(1, 2, 0).numpy()
                    # Convert the image to uint8
                    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
                    # Convert the RGB image to BGR (OpenCV format)
                    bgr_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR)
                    # Get the bounding box coordinates of the face
                    x_box, y_box, width, height = get_face_bbox(face_detector, bgr_image)
                    if x_box is not None:
                        bbox_found = True
                        face_bbox = (x_box, y_box, width, height)
                        detections.append(face_bbox)
                        if len(detections)>300:
                            average_bbox,_ = get_average_bbox_and_std(detections)
                            detections.clear()
                            detections.append(average_bbox)
                    else:
                        face_bbox,_ = get_average_bbox_and_std(detections)
                    
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'x0_t')
                    save_img(i,path_for_img,config,x0_t)
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
                if bbox_found:
                    # Calculate variance based on the step (e.g., gradually decreasing towards step 0)
                    max_variance = 10000  # adjust this based on your image size
                    min_variance = 100  # minimum variance when close to step 0
                    # Generate the list of variances corresponding to the number of steps
                    num_steps = len(time_pairs)
                    variances = np.linspace(min_variance, max_variance, num_steps)
                    # Retrieve the variance for the current step
                    current_variance = variances[step_idx]
                    # Create the Gaussian kernel based on the bounding box and the calculated variance
                    gaussian_kernel = create_gaussian_kernel((x0_t.size(2), x0_t.size(3)), face_bbox, current_variance)
                    gaussian_kernel = torch.tensor(gaussian_kernel, dtype=torch.float32).to(x0_t.device)
                    # Create the Gaussian kernel based on the bounding box and the calculated variance
                    H, W = x0_t.size(2), x0_t.size(3)
                    gaussian_kernel = create_gaussian_kernel((H, W), face_bbox, current_variance)
                    if ckpt_imgs_path is not None:
                        path_for_img = os.path.join(ckpt_imgs_path,'gaussian_kernel_heatmap')
                        if not os.path.exists(path_for_img):
                            os.mkdir(path_for_img)
                        # Save the Gaussian kernel as a heatmap image
                        save_gaussian_heatmap(gaussian_kernel, os.path.join(path_for_img,f'{i}.png'))
                    # Convert the Gaussian kernel to a tensor and move it to the device
                    gaussian_kernel_tensor = torch.tensor(gaussian_kernel, dtype=torch.float32).to(x0_t.device)
                
                
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'guidance_BP')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    save_heatmap(inverse_data_transform(config, guidance_BP).to('cpu').squeeze().permute(1, 2, 0), 
                                 os.path.join(path_for_img,f"{i}.png"))

                    # Normalize guidance_LS to have the same range as guidance_BP
                    min_BP, max_BP = guidance_BP.min(), guidance_BP.max()
                    min_LS, max_LS = guidance_LS.min(), guidance_LS.max()
                    # Standardize guidance_LS to the range of guidance_BP
                    guidance_LS_normalized = (guidance_LS - min_LS) / (max_LS - min_LS)  # Normalize guidance_LS to [0, 1]
                    guidance_LS_standardized = guidance_LS_normalized * (max_BP - min_BP) + min_BP  # Scale to [min_BP, max_BP]
                    path_for_img = os.path.join(ckpt_imgs_path,'guidance_LS')
                    if not os.path.exists(path_for_img):
                        os.mkdir(path_for_img)
                    save_heatmap(inverse_data_transform(config, 
                                                        guidance_LS_standardized).to('cpu').squeeze().permute(1, 2, 0), 
                                 os.path.join(path_for_img,f"{i}.png"))
                if deid and bbox_found:
                    if i==0:
                        print('stop')
                    guidance_BP = guidance_BP*gaussian_kernel_tensor
                    guidance_LS = guidance_LS*gaussian_kernel_tensor
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
                    save_img(i,path_for_img,config,xt_next_tilde)
                # compute effective noise
                et_hat = ( xt - at.sqrt() * xt_next_tilde ) / (1 - at).sqrt()

                c1 = 0
                c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1-zeta)  * weight_noise_t

                xt_next = at_next.sqrt() * xt_next_tilde + c1 * torch.randn_like(x0_t) + c2 * et_hat
                '''
                xt_next:
                        This is the final state of the image after adding noise to the 
                        intermediate state (xt_next_tilde). It represents the next 
                        iteration’s state (xt-1).
                '''
                if ckpt_imgs_path is not None:
                    path_for_img = os.path.join(ckpt_imgs_path,'xt_next')
                    save_img(i,path_for_img,config,xt_next)

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

            else: 
                assert 1, "Unexpected case"
        
        if sigma_y != 0.:  # if there is noise, take the denoised result
            xs.append(x0_t.to('cpu'))

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


