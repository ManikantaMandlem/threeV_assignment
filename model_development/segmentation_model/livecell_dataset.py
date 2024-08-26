from torch.utils.data import Dataset
import json
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch
import numpy as np
import os


class LivecellDataset(Dataset):
    def __init__(self, json_file, images_base, size=(224, 224), transform=None):
        self.data = json.load(open(json_file, "r"))
        self.image_id_name = {}
        # check for duplicates in id to name mappings
        for info in self.data["images"]:
            assert (
                info["id"] not in self.image_id_name
            ), "multiple image id mappings present"
            self.image_id_name[info["id"]] = info["file_name"]
        self.image_annotations = defaultdict(list)
        # sanity check for coco annotation format
        for ann in self.data["annotations"]:
            assert len(ann["segmentation"]) == 1, "inconsistency in annotation format"
            file_name = self.image_id_name[ann["image_id"]]
            # join all the polygons for single image as we will have binary mask
            self.image_annotations[file_name].append(ann["segmentation"][0])
        self.image_annotations = dict(self.image_annotations)
        self.all_images = list(self.image_annotations.keys())
        self.transform = transform
        self.image_width = size[0]
        self.image_height = size[1]
        self.images_base = images_base
        # customary image only transforms here
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    def __len__(self):
        return len(self.all_images)

    def create_mask(self, polygon):
        mask = Image.new("L", (704, 520), 0)
        for shape in polygon:
            points = [(shape[i], shape[i + 1]) for i in range(0, len(shape), 2)]
            # Create an empty mask
            draw = ImageDraw.Draw(mask)
            # Draw the polygon on the mask
            draw.polygon(points, outline=1, fill=1)
        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        mask = mask.unsqueeze(0)
        return mask

    def __getitem__(self, id):
        image_name = self.all_images[id]
        # extract polygons and create binary gt mask
        mask = self.image_annotations[image_name]
        mask = self.create_mask(mask)
        # load image and convert into tensor
        image_path = os.path.join(self.images_base, image_name)
        image = Image.open(image_path)
        image = self.to_tensor(image)
        # necessary transofmration for data aug.
        if self.transform:
            image, mask = self.transform(image, mask)
        # normalize the image for training stability
        image = self.normalize(image)
        # create the data sample
        sample = {"file_name": image_name, "image": image, "mask": mask}
        return sample
