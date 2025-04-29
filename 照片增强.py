import os
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm

# 灰度转换
def convert_to_grayscale(img):
    return img.convert('L')

# 图像去噪
def denoise_image(img):
    return img.filter(ImageFilter.MedianFilter(size=3))

# 图像增强
def augment(img):
    # 灰度转换
    img = convert_to_grayscale(img)
    # 图像去噪
    img = denoise_image(img)
    # 亮度增强
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    # 对比度增强
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    # 色彩增强
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    # 清晰度增强（增加照片锐度）
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))

    return img

def augment_image(input_path, output_path, num_augmented_images):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历每一个文件夹
    for category in os.listdir(input_path):
        category_path = os.path.join(input_path, category)
        output_category_path = os.path.join(output_path, category)

        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        # 遍历每一张文件
        for filename in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            img_path = os.path.join(category_path, filename)

            try:
                img = Image.open(img_path)

                augmented_images = [img]
                for i in range(num_augmented_images):
                    augmented_img = augment(img)
                    augmented_images.append(augmented_img)

                for idx, augmented_img in enumerate(augmented_images):
                    output_filename = f"{filename.split('.')[0]}_aug_{idx + 1}.{filename.split('.')[-1]}"
                    output_img_path = os.path.join(output_category_path, output_filename)

                    # 转换成 RGB通道
                    augmented_img = augmented_img.convert("RGB")
                    augmented_img.save(output_img_path)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

input_folder_path = r"C:\Users\19588\Desktop\计设+服创\鱼类识别\NA_Fish_Dataset"

output_folder_path = r"C:\Users\19588\Desktop\计设+服创\鱼类识别\new_NA_Fish_Dataset"

num_augmented_images =5

augment_image(input_folder_path, output_folder_path, num_augmented_images)

