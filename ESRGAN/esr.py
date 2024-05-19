import cv2
import numpy as np
import torch
import ESRGAN.RRDBNet_arch as arch


def enhance_image(image_path, model_path="/home/mazen/gui/ESRGAN/models/RRDB_ESRGAN_x4.pth", device="cpu"):
    # Load model
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)

    # Read input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Enhance image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Post-process and save the enhanced image
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    return output
