import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.ssd import ssd300_vgg16


class MyGeneralizedRCNNTransform(GeneralizedRCNNTransform):
    def batch_images(self, images, size_divisible=32):
        return torch.stack(images)


def get_ssd():
    model = ssd300_vgg16(pretrained=True)
    image_mean = [0.48235, 0.45882, 0.40784]
    image_std = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
    size = (300, 300)
    model.transform = MyGeneralizedRCNNTransform(min(size), max(size), image_mean, image_std,
                                                 size_divisible=1, fixed_size=size)
    return model
