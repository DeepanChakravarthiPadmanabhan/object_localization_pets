import os
import xml.etree.ElementTree as ET

from localize_pets.abstract.loader import Loader


class Pets_Detection(Loader):
    """Data manager for Pets dataset.
     Aids in loading RGB images, segmentation
     masks and detection bounding box information.
    # Arguments
        image_path:
        annotations_path:
        split:
        class_names:
    """

    def __init__(self, data_dir, split, class_names="all"):
        if split not in ["trainval", "test"]:
            raise ValueError("Invalid split name: ", split)
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, "images")
        self.annotations_dir = os.path.join(self.data_dir, "annotations")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        self.masks_dir = os.path.join(self.annotations_dir, "trimaps")
        self.trainval_examples = os.path.join(self.annotations_dir, "trainval.txt")
        self.test_examples = os.path.join(self.annotations_dir, "test.txt")
        if class_names == "all":
            self.class_names = 1
        super(Pets_Detection, self).__init__(None, split, class_names, "Pets")

    @staticmethod
    def read_xml(bbox_file):
        tree = ET.parse(bbox_file)
        root = tree.getroot()
        value = {}
        for member in root.findall("object"):
            value = {
                "filename": root.find("filename").text,  # filename
                "height": int(root.find("size")[0].text),  # height
                "width": int(root.find("size")[1].text),  # width
                "class": member[0].text,  # species or class
                "xmin": int(member[4][0].text),  # xmin
                "ymin": int(member[4][1].text),  # ymin
                "xmax": int(member[4][2].text),  # xmax
                "ymax": int(member[4][3].text),  # ymax
            }

        return value

    def load_data(self):
        if self.split == "trainval":
            image_list_file = self.trainval_examples
            images_list = open(image_list_file, "r")
            dataset = []
            for line in images_list:
                image_name, label, species, _ = line.strip().split(" ")
                xml_name = image_name + ".xml"
                mask_name = image_name + ".png"
                image_name += ".jpg"
                species = int(species) - 1
                if (
                    os.path.exists(os.path.join(self.xmls_dir, xml_name))
                    and os.path.exists(os.path.join(self.images_dir, image_name))
                    and os.path.exists(os.path.join(self.masks_dir, mask_name))
                ):
                    xml_path = os.path.join(self.xmls_dir, xml_name)
                    image_path = os.path.join(self.images_dir, image_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    bbox_details = self.read_xml(xml_path)
                    bbox = [
                        bbox_details["xmin"],
                        bbox_details["ymin"],
                        bbox_details["xmax"],
                        bbox_details["ymax"],
                    ]
                    record = {
                        "image_path": image_path,
                        "label": int(label),
                        "species": int(species),
                        "xml_path": xml_path,
                        "mask_path": mask_path,
                        "bbox": bbox,
                    }
                    dataset.append(record)
        else:
            image_list_file = self.test_examples
            images_list = open(image_list_file, "r")
            dataset = []
            for line in images_list:
                image_name, label, species, _ = line.strip().split(" ")
                mask_name = image_name + ".png"
                image_name += ".jpg"
                species = int(species) - 1
                if os.path.exists(
                    os.path.join(self.images_dir, image_name)
                ) and os.path.exists(os.path.join(self.masks_dir, mask_name)):
                    image_path = os.path.join(self.images_dir, image_name)
                    mask_path = os.path.join(self.masks_dir, mask_name)
                    record = {
                        "image_path": image_path,
                        "label": int(label),
                        "species": int(species),
                        "mask_path": mask_path,
                    }
                    dataset.append(record)

        return dataset
