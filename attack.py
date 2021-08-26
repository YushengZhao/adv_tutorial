import sys
import torch
from torchvision.io import read_image, write_png
from model import get_ssd


def main():
    image_path = sys.argv[1]
    h_start = int(sys.argv[2])
    h_end = int(sys.argv[3])
    w_start = int(sys.argv[4])
    w_end = int(sys.argv[5])
    output_path = sys.argv[6]
    image = read_image(image_path).float() / 255
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = get_ssd().eval().to(device)
    image = image.to(device)

    # attack the model
    lr = 1 / 255
    patch_mask = torch.zeros((1, 500, 500), device=device)
    patch_mask[:, h_start:h_end, w_start:w_end] = 1
    for i in range(500):
        image.requires_grad = True
        result = detector([image])[0]
        scores = result['scores']
        mask = scores > 0.25
        loss = scores[mask].sum()
        print(loss.item())
        if loss.item() == 0:
            break
        loss.backward()
        grad_sign = torch.sign(image.grad)
        image = torch.clamp(image - grad_sign * lr * patch_mask, 0, 1).detach()

    # write the image back
    image = (image * 255).type(torch.uint8).cpu()
    write_png(image, output_path)


if __name__ == '__main__':
    main()
