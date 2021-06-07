import numpy as np
import cv2
import matplotlib.pyplot as plt


def deprocess_image(image):
    image = image.copy()
    mean = image.mean()
    std = image.std()
    image -= mean
    image /= std + 1e-05
    image *= 0.25
    # clip to [0, 1]
    image += 0.5
    image = np.clip(image, 0, 1)
    # Convert to RGB array
    image *= 255.0
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(save_path_and_name, image):
    plt.imsave(save_path_and_name, image)


def plot_inference_and_visualization(image, pet_bbox, pet_class, saliency, visualization='gbp', name='visualize_'):
    start_point = (int(pet_bbox[0]), int(pet_bbox[1]))
    end_point = (int(pet_bbox[2]), int(pet_bbox[3]))
    image = cv2.rectangle(image, start_point, end_point, (255, 255, 0), 2)
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(image)
    plt.title(pet_class)
    plt.subplot(122)
    if visualization == 'gbp':
        plt.imshow(saliency)
    elif visualization == 'grad_cam':
        plt.imshow(saliency)
    plt.title(visualization)
    plt.show()
    if name == 'visualize_':
        fig_name = 'visualize_' + visualization + '.jpg'
    else:
        fig_name = 'visualize_' + visualization + '_' + str(name) + '.jpg'
    plt.savefig(fig_name)
