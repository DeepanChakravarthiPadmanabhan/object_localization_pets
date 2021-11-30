import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from localize_pets.visualization.utils import plot_inference_and_visualization
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
from localize_pets.utils.misc import CLASS_MAPPING
from localize_pets.visualization.utils import visualize_saliency_grayscale


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, layer_name=None, visualize_idx=None):
        self.model = model
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        if self.layer_name == None:
            self.layer_name = self.find_target_layer()
        self.gbModel = self.build_guided_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError(
            "Could not find 4D layer. Cannot apply guided backpropagation."
        )

    def build_guided_model(self):
        gbModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
        )
        try:
            base_model_layers = [act_layer
                                 for act_layer in gbModel.layers[1].layers
                                 if hasattr(act_layer, 'activation')]
        except:
            base_model_layers = [act_layer
                                 for act_layer in gbModel.layers
                                 if hasattr(act_layer, 'activation')]
        head_layers = [layer for layer in gbModel.layers[1:]
                       if hasattr(layer, "activation")]
        all_layers = base_model_layers + head_layers
        for layer in all_layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu

        if 'class' in self.layer_name:
            gbModel.get_layer(self.layer_name).activation = None
        return gbModel

    def guided_backpropagation(self, image):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            conv_outs, preds = self.gbModel(inputs)
            conv_outs = conv_outs[0][self.visualize_idx]
        print('Conv outs shape: ', conv_outs)
        grads = tape.gradient(conv_outs, inputs)[0]
        grads = np.asarray(grads)
        pet_class = CLASS_MAPPING[np.argmax(preds[0].numpy())]
        pet_coord = preds[1].numpy()[0]
        return grads, pet_coord, pet_class


description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-i", "--image_path",
                    default="/media/deepan/externaldrive1/datasets_project_repos/pets_data/images/Abyssinian_30.jpg",
                    type=str, help="Image path")
# basset_hound_163
# Abyssinian_30
parser.add_argument("-m", "--model_path",
                    default="save_checkpoint_simplemodel/pets_model/",
                    type=str, help="Model path")
parser.add_argument("-l", "--layer_name", default="class_out",
                    type=str, help="Layer to visualize")
parser.add_argument("--visualize_idx", default=None, type=int,
                    help="Index to visualize. Corresponds to the class.")
parser.add_argument("-iw", "--image_width", default=224, type=int,
                    help="Input image width")
parser.add_argument("-ih", "--image_height", default=224, type=int,
                    help="Input image height")
parser.add_argument("-n", "--normalize", default="resnet50", type=str,
                    help="Normalization strategy. Available options: "
                         "max, same, vgg19, resnet50."
                         " Max for SimpleNet, vgg19 for VGG19, "
                         "resnet50 for ResNet50 and same for EfficientNet")
parser.add_argument("--resize", default=True, type=bool,
                    help="Whether to resize the image")
args = parser.parse_args()
config = vars(args)
model_path = config["model_path"]
layer_name = config["layer_name"]
image_path = config["image_path"]
image_height = config["image_height"]
image_width = config["image_width"]
visualize_idx = config["visualize_idx"]
transforms = dict()
if config["resize"]:
    transforms["resize"] = [config["image_height"], config["image_width"]]
if config["normalize"]:
    transforms["normalize"] = config["normalize"]

raw_image = cv2.imread(image_path)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
display_image = cv2.resize(raw_image,
                           (config["image_width"], config["image_height"]),
                           interpolation=cv2.INTER_NEAREST)
image_input, _ = process_bbox_image(raw_image, None, transforms)
image_input = image_input[np.newaxis]
print("Initial Image Stats: ", np.min(image_input), np.max(image_input),
      np.mean(image_input))
assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(
    model_path, custom_objects={"IOU": IOU(name="iou")})
print(model.summary())

gbp = GuidedBackpropagation(model, layer_name, visualize_idx)
saliency, pet_bbox, pet_class = gbp.guided_backpropagation(image_input)

saliency_stat = (np.min(saliency), np.max(saliency))
saliency = visualize_saliency_grayscale(saliency[np.newaxis])
fig_title = 'Guided Backpropagation explanation for x_min decision'
fig_title = fig_title
plot_inference_and_visualization(image=display_image,
                                 pet_bbox=pet_bbox,
                                 pet_class=pet_class,
                                 saliency=saliency,
                                 visualization='GuidedBackpropagation',
                                 saliency_stat=saliency_stat,
                                 name='offset0', fig_title=fig_title)

sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
sorted_indices = np.vstack(sorted_flat_indices).T

# Only for altering most important pixels.
print('STARTING FINAL STUDIES')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from localize_pets.visualization.utils import plot_saliency
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(4.5*4, 5.25))
fig.subplots_adjust(left=0.05, right=0.93, bottom=0.02,
                    top=0.9, wspace=0.3, hspace=0.5)
fig_title1 = 'Guided Backpropagation for x_min decision. Bounding box on changing most important pixels in the image.\n\n'
fig_title2 = r'Number of pixels changed $\rightarrow$'
fig_title = fig_title1 + fig_title2
ax = [ax1, ax2, ax3, ax4]
linspace = [0, 200, 2000, 20000]
for i in range(4):
    image_inp = cv2.resize(raw_image,
                       (config["image_width"], config["image_height"]),
                       interpolation=cv2.INTER_NEAREST)
    change_pixels = sorted_indices[:linspace[i]]
    image_inp[change_pixels[:, 0], change_pixels[:, 1], :] = 0

    transforms = dict()
    if config["normalize"]:
        transforms["normalize"] = config["normalize"]
    image, _ = process_bbox_image(image_inp, None, transforms)
    image = image[np.newaxis]

    print("End Image Stats: ", np.min(image), np.max(image), np.mean(image))
    gbp = GuidedBackpropagation(model, layer_name, visualize_idx)
    saliency, pet_bbox, pet_class = gbp.guided_backpropagation(image)
    saliency_stat = (np.min(saliency), np.max(saliency))
    saliency = visualize_saliency_grayscale(saliency[np.newaxis])
    saliency_title = 'Pixels changed: ' + str(linspace[i]) if i != 0 else (
        'Original | Pixels changed: 0')
    plot_saliency(saliency, ax[i], title=saliency_title,
                  saliency_stat=saliency_stat)

    start_point = (int(pet_bbox[0]), int(pet_bbox[1]))
    end_point = (int(pet_bbox[2]), int(pet_bbox[3]))
    width_rect = int(pet_bbox[2] - pet_bbox[0])
    height_rect = int(pet_bbox[3] - pet_bbox[1])
    ax[i].imshow(image_inp, alpha=0.4)
    rect = patches.Rectangle(start_point, width_rect, height_rect,
                             linewidth=1, edgecolor='r', facecolor='none')
    ax[i].add_patch(rect)
fig.suptitle(fig_title)
plt.savefig('altered.jpg')
plt.show()
plt.close()




