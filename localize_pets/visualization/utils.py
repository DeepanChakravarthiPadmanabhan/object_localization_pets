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


def visualize_image_grayscale(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    image_gray = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    return image_gray


def visualize_image_diverging(image_3d, percentile=99):
    image_2d = np.sum(image_3d, axis=2)
    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span
    image_diverging = np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)
    return image_diverging


def save_image(save_path_and_name, image, visualize_type='grayscale'):
    if visualize_type == 'grayscale':
        image = visualize_image_grayscale(image)
        plt.imsave(save_path_and_name, image, cmap='gray', vmin=0, vmax=1)
    elif visualize_type == 'diverging':
        image = visualize_image_diverging(image)
        plt.imsave(save_path_and_name, image, cmap='jet')



def plot_inference_and_visualization(image,
                                     pet_bbox,
                                     pet_class,
                                     saliency,
                                     visualization='gbp',
                                     name='visualize_',
                                     additional_fig_features=None):
    start_point = (int(pet_bbox[0]), int(pet_bbox[1]))
    end_point = (int(pet_bbox[2]), int(pet_bbox[3]))
    image = cv2.rectangle(image.astype('uint8'), start_point, end_point, (255, 255, 0), 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(image)
    ax1.set_title(pet_class)
    if visualization == 'gbp':
        ax2.imshow(saliency)
    elif visualization == 'grad_cam':
        ax2.imshow(saliency)
        min_intensity = additional_fig_features['min_intensity']
        max_intensity = additional_fig_features['max_intensity']
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                            orientation='vertical',
                            fraction=0.046,
                            pad=0.04)
        m1 = 0  # colorbar min value
        m4 = 1  # colorbar max value
        m2 = min_intensity / 255
        m3 = max_intensity / 255
        if m3 < m4 - 0.2:
            cbar.set_ticks([m1, m2, m3, m4])
            cbar.set_ticklabels([0, min_intensity, max_intensity, 255])
        else:
            cbar.set_ticks([m1, m2, m3, m4])
            cbar.set_ticklabels([0, min_intensity, max_intensity, ''])

    ax2.set_title(visualization)
    if name == 'visualize_':
        fig_name = 'visualize_' + visualization + '.jpg'
    else:
        fig_name = 'visualize_' + visualization + '_' + str(name) + '.jpg'
    plt.savefig(fig_name)
    plt.show()
    plt.close()


def get_mpl_colormap(cmap_name='jet'):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2:: -1]
    return color_range.reshape(256, 1, 3)
