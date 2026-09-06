import cv2


def load_img(filepath):
    """Load an RGB image with OpenCV."""
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    """Save an RGB image with OpenCV."""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
