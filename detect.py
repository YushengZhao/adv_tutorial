import sys
import torchvision.models as models
from torchvision.io import read_image, write_png
from torchvision.utils import draw_bounding_boxes


def main():
    image_path = sys.argv[1]
    img = read_image(image_path)
    image = img.clone().float() / 255
    detector = models.detection.ssd300_vgg16(pretrained=True).eval()
    result = detector([image])[0]
    boxes = result['boxes']
    scores = result['scores']
    mask = scores > 0.3
    pred_boxes = boxes[mask]
    image_with_boxes = draw_bounding_boxes(img, pred_boxes, colors=['red'] * len(pred_boxes))
    print("%d objects detected." % len(pred_boxes))
    write_png(image_with_boxes, 'result.png')


if __name__ == '__main__':
    main()
