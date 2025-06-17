import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.util import img_as_ubyte


def load_image(image_path):
    """加载并预处理红外图像"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像，请检查路径: {image_path}")
    return img_as_ubyte(img)


def apply_clahe(img, clip_limit=0.02, grid_size=(16, 16)):
    """应用CLAHE增强"""
    return exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=grid_size)


def denoise_bilateral(img, d=15, sigma_color=75, sigma_space=75):
    """双边滤波降噪"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def plot_processing_steps(original, denoised, enhanced):
    """绘制处理过程的所有步骤"""
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('1. 原始红外图像', fontproperties='SimHei')
    plt.axis('off')

    # 原始直方图
    plt.subplot(2, 3, 4)
    plt.hist(original.ravel(), bins=256, range=(0, 1))
    plt.title('原始直方图', fontproperties='SimHei')

    # 降噪后图像
    plt.subplot(2, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('2. 双边滤波降噪后', fontproperties='SimHei')
    plt.axis('off')

    # 降噪直方图
    plt.subplot(2, 3, 5)
    plt.hist(denoised.ravel(), bins=256, range=(0, 1))
    plt.title('降噪后直方图', fontproperties='SimHei')

    # CLAHE增强后图像
    plt.subplot(2, 3, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title('3. CLAHE增强后', fontproperties='SimHei')
    plt.axis('off')

    # 增强后直方图
    plt.subplot(2, 3, 6)
    plt.hist(enhanced.ravel(), bins=256, range=(0, 1))
    plt.title('增强后直方图', fontproperties='SimHei')

    plt.tight_layout()
    plt.show()


def save_results(enhanced, output_path):
    """保存处理结果"""
    enhanced_8bit = (enhanced * 255).astype(np.uint8)
    cv2.imwrite(output_path, enhanced_8bit)
    print(f"增强图像已保存至: {output_path}")


def main():
    # 文件路径配置
    input_path = r"C:/Users/Administrator/Desktop/dianligui.jpg"
    output_path = r"C:/Users/Administrat
