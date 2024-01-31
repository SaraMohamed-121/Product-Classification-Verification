import cv2
import os
import warnings

def set_background_image(product_image_path, background_image, output_image_path):
    product_image = cv2.imread(product_image_path, cv2.IMREAD_UNCHANGED)
    background_image = cv2.resize(background_image, (product_image.shape[1], product_image.shape[0]))
    alpha_channel = product_image[:, :, 0]
    mask = cv2.merge((alpha_channel, alpha_channel, alpha_channel))
    combined_image = cv2.bitwise_and(background_image, cv2.bitwise_not(mask))
    combined_image = cv2.add(combined_image, product_image[:, :, :3])
    
    cv2.imwrite(output_image_path, combined_image)
    cv2.imshow("Result", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
            image_path = os.path.join(folder_path, filename)
            original_image = cv2.imread(image_path)

            # Perform augmentation (ex: rotate, flip, and brightness adjustment)
            # Rotate the image by 90 degrees
            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
            # Flip the image horizontally
            flipped_image = cv2.flip(original_image, 1)  # 1 for horizontal flip
            # Adjust brightness
            brightness_increase = cv2.convertScaleAbs(original_image, alpha=0.7, beta=50)

            # Save the augmented images with modified filenames
            rotated_image_path = os.path.join(folder_path, f"rotated_{filename}")
            flipped_image_path = os.path.join(folder_path, f"flipped_{filename}")
            brightness_image_path = os.path.join(folder_path, f"brightness_{filename}")

            cv2.imwrite(rotated_image_path, rotated_image)
            cv2.imwrite(flipped_image_path, flipped_image)
            cv2.imwrite(brightness_image_path, brightness_increase)    


base_folder = 'Data\\Product Classification'
background_image = cv2.imread('background.jpg')

for folder_number in range(1, 21):
    product_folder = os.path.join(base_folder, str(folder_number), 'Train')

    augment_images_in_folder(product_folder)
    for filename in os.listdir(product_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            product_image_path = os.path.join(product_folder, filename)
            output_image_path = os.path.join(product_folder, f"back_{filename}")
            set_background_image(product_image_path, background_image, output_image_path)
            warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

