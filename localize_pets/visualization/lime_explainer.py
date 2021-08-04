import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from localize_pets.loss_metric.iou import IOU
from localize_pets.transforms.transforms import process_bbox_image
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

class CallFunction:
    def __init__(self, model, transforms):
        self.model = model
        self.transforms = transforms
    def batch_predict(self, images):
        images_processed = []
        for i in images:
            ex_image, _ = process_bbox_image(i,
                                             None,
                                             self.transforms)
            images_processed.append(ex_image)
        explain_images = np.stack(images_processed, axis=0)
        return self.model(explain_images).numpy()

def plot_relevance(method, im, dataset, mask, l, det, save_path):
    H, W, C = im.shape
    classes = CLASS_MAPPING
    num_classes = len(CLASS_MAPPING)
    true_class = CLASS_MAPPING[l]
    pred_class = CLASS_MAPPING[l]
    result = 'TP'
    score = str(round(det[0][1] * 100))
    [xmin, ymin, xmax, ymax] = np.round(det[0][2:]).astype(int)
    colorsB = np.linspace(250, 100, num=num_classes).astype(int)
    colorsG = np.linspace(100, 250, num=num_classes).astype(int)
    colorsR = np.linspace(0, 255, num=num_classes).astype(int)
    # original image
    image = im.copy()
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
    # heatmap
    p = mask
    p_norm = np.max(np.abs(p)) + 1e-8
    p_image = 255 * np.ones((im.shape))
    pB = 255.0 * p.copy() / p_norm
    pR = 255.0 * p.copy() / p_norm
    pB[pB > 0.0] = 0.0
    pR[pR < 0.0] = 0.0

    p_image[:, :, 0] -= pR
    p_image[:, :, 1] -= pR
    p_image[:, :, 2] -= pR / 4

    p_image[:, :, 0] += pB / 4
    p_image[:, :, 1] += pB
    p_image[:, :, 2] += pB

    # bar image
    b_image = np.zeros((H, 20, C))
    b_image[0:H // 2, ::] = np.stack((np.tile(np.linspace(0, 255, num=H // 2)[..., np.newaxis], 20),
                                 np.tile(np.linspace(0, 255, num=H // 2)[..., np.newaxis], 20),
                                 np.tile(np.linspace(192, 255, num=H // 2)[..., np.newaxis], 20)), axis=2)
    b_image[H // 2:H, ::] = np.stack((np.tile(np.linspace(255, 192, num=H // 2)[..., np.newaxis], 20),
                                 np.tile(np.linspace(255, 0, num=H // 2)[..., np.newaxis], 20),
                                 np.tile(np.linspace(255, 0, num=H // 2)[..., np.newaxis], 20)), axis=2)
    # overlayed image
    g_image = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    o_image = np.tile(g_image[..., np.newaxis], 3)
    alpha = 0.6
    cv2.addWeighted(p_image, alpha, im, 1 - alpha, 0, o_image)
    cv2.rectangle(o_image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

    # draw
    back = [200, 200, 200]
    im1 = cv2.copyMakeBorder(image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=back)
    im2 = cv2.copyMakeBorder(p_image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=back)
    im3 = cv2.copyMakeBorder(b_image, 40, 10, 10, 90, cv2.BORDER_CONSTANT, value=back)
    im4 = cv2.copyMakeBorder(o_image, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=back)
    #
    text1 = true_class + '(' + result + ', conf=' + score + '%)'
    text2 = method + ': ' + pred_class
    text3 = ' '
    textP3 = "+{:1.6f}".format(p_norm)
    textZ3 = " {:1.1f}".format(0.0)
    textN3 = "-{:1.6f}".format(p_norm)
    text4 = 'Overlayed'
    #
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    textOrg = (12, 22)
    #
    cv2.putText(im1, text1, textOrg, fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im2, text2, textOrg, fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, text3, textOrg, fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textP3, (30, 0 + 40), fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textZ3, (30, H // 2 + 40), fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im3, textN3, (30, H + 40), fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    cv2.putText(im4, text4, textOrg, fontFace, fontScale, (0, 0, 0), thickness, 0, bottomLeftOrigin=False)
    #
    finalIm = cv2.hconcat((im1, im2, im3, im4))
    cv2.imwrite(save_path, finalIm)

CLASS_MAPPING = {0: "Cat", 1: "Dog", 2: "XMIN", 3: "YMIN", 4: "XMAX", 5: "YMAX"}
description = "Inference script for object localization task on pets dataset"
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    "-dd",
    "--data_dir",
    default="/media/deepan/externaldrive1/datasets_project_repos/pets_data/",
    type=str,
    help="Data directory for training",
)
parser.add_argument(
    "--train_samples", default=2800, type=int, help="Sample set size for training data"
)
parser.add_argument(
    "--test_samples", default=800, type=int, help="Sample set size for testing data"
)
parser.add_argument(
    "-i",
    "--image_path",
    default="/media/deepan/externaldrive1/datasets_project_repos/pets_data/images/basset_hound_163.jpg",
    type=str,
    help="Image path",
)
parser.add_argument(
    "-m",
    "--model_path",
    default="save_checkpoint_resnet/pets_model/",
    type=str,
    help="Model path",
)
parser.add_argument(
    "-l", "--layer_name", default="conv2d_1", type=str, help="Layer to visualize"
)
parser.add_argument(
    "--visualize_head", default="detection", type=str, help="Head to visualize"
)
parser.add_argument(
    "--visualize_idx", default=1, type=int, help="Index to visualize. Corresponds to the class."
)
parser.add_argument(
    "-iw", "--image_width", default=224, type=int, help="Input image width"
)
parser.add_argument(
    "-ih", "--image_height", default=224, type=int, help="Input image height"
)
parser.add_argument(
    "-n",
    "--normalize",
    default="resnet50",
    type=str,
    help="Normalization strategy. "
    "Available options: max, same, vgg19, resnet50. "
    "Max for SimpleNet, VGG19 and same_scale for EfficientNet",
)
parser.add_argument(
    "--resize",
    default=True,
    type=bool,
    help="Whether to resize the image",
)

args = parser.parse_args()
config = vars(args)
dataset_path = config["data_dir"]
model_path = config["model_path"]
layer_name = config["layer_name"]
image_path = config["image_path"]
image_height = config["image_height"]
image_width = config["image_width"]
visualize_head = config["visualize_head"]
visualize_idx = config["visualize_idx"]

transform_resize = dict()
transform_normalize = dict()
if config["resize"]:
    transform_resize["resize"] = [config["image_height"], config["image_width"]]
if config["normalize"]:
    transform_normalize["normalize"] = config["normalize"]

assert os.path.exists(model_path), "Model path does not exist."
model = tf.keras.models.load_model(model_path, custom_objects={"IOU": IOU(name="iou")})
input_layer = model.layers[0].input
outs = model.output
concatenated = tf.concat(outs, axis=1)
model = tf.keras.Model(inputs=input_layer, outputs=concatenated)
print(model.summary())

raw_image = cv2.imread(image_path)
input_image, _ = process_bbox_image(raw_image, None, transform_resize)
input_image = input_image[np.newaxis]
det = model(input_image).numpy()
print(det)

caller = CallFunction(model, transform_normalize)
pred_fn = caller.batch_predict
explainer = lime_image.LimeImageExplainer(verbose=True)
explanation = explainer.explain_instance(input_image[0].astype('double'),
                                         pred_fn,
                                         labels=np.arange(0, 6),
                                         top_labels=6,
                                         hide_color=0.0,
                                         num_samples=10,
                                         batch_size=8,
                                         random_seed=10)
to_visualize = 1
temp, mask = explanation.get_image_and_mask(explanation.top_labels[1],
                                            positive_only=True,
                                            num_features=5,
                                            min_weight=1e-3, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255., mask)
print(img_boundry1.shape, mask.shape)
save_path = 'lime.jpg'
plt.imsave(save_path, img_boundry1)
plot_relevance('LIME', input_image[0].astype('double'), 'MSCOCO', mask, to_visualize, det, save_path)