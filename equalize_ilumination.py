import cv2
import numpy as np

def histogram_equalization(image):
    """
    Apply histogram equalization to an image.
    :param image: Input image (assumed to be grayscale or BGR).
    :return: Histogram-equalized image.
    """
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image (BGR)
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Equalize the Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        raise ValueError("Unsupported image format!")

# Example usage
image = cv2.imread('/home/tico/Desktop/master_classes/IBB/project/Mask_DeID_DDPG/exp/datasets/celeba_hq/fri/img_0a1f62ae8c6ab679df3ec46b93aa071d.png')
equalized_image = histogram_equalization(image)
cv2.imwrite('/home/tico/Desktop/master_classes/IBB/project/Mask_DeID_DDPG/exp/datasets/celeba_hq/fri/img_0a1f62ae8c6ab679df3ec46b93aa071d_equalized.png', equalized_image)