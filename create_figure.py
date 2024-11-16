import os
import matplotlib.pyplot as plt
from PIL import Image

def create_diffusion_figure(base_path, directories, steps, output_path):
    # Check if the subject has "deid" in its name to decide if we add the Gaussian kernel column
    subject = os.path.basename(base_path)
    include_gaussian = "deid" in subject.lower()
    
    # If including Gaussian kernels, add the directory name to the list and adjust the column count
    if include_gaussian:
        directories = ["gaussian_kernel_heatmap"] + directories

    # Set up the figure size based on the number of directories (columns) and steps (rows)
    fig, axs = plt.subplots(len(steps), len(directories), figsize=(15, 10))
    
    # Loop through each step and each directory to fill in the images
    for row, step in enumerate(steps):
        for col, directory in enumerate(directories):
            # Build the image path
            img_path = os.path.join(base_path, directory, f"{step}.png")
            
            # Load the image
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axs[row, col].imshow(img)
                axs[row, col].axis('off')  # Hide axes for a cleaner look
            else:
                # If the image does not exist, just leave the subplot empty
                axs[row, col].text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)
                axs[row, col].axis('off')
                
    # Add column titles (directory names)
    for col, directory in enumerate(directories):
        axs[0, col].set_title(directory, fontsize=12)
    
    # Add row titles (step numbers)
    for row, step in enumerate(steps):
        axs[row, 0].set_ylabel(f"Step {step}", rotation=0, labelpad=40, va='center', fontsize=12)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Define the parameters
base_path = "exp/image_samples/ZigaEmersic_deid"
subject = os.path.basename(base_path)
directories = ["xt_next", "xt_next_tilde", "x0_t", "guidance_BP", "guidance_LS"]
#steps = [990, 890, 790, 590, 250, 100, 0]
steps = [990, 790, 250, 0]
output_path = f"diffusion_steps_figure_{subject}_2.png"

# Call the function to create the figure
create_diffusion_figure(base_path, directories, steps, output_path)