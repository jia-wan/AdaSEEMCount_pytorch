import numpy as np
import matplotlib.pyplot as plt
import sys
import torch


def show_mask(masks, ax, random_color=False):
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1)) 

def load_sam(mobile=True):
    sys.path.append('../MobileSAM/')
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mobile:
        model_type = "vit_t"
        sam_checkpoint = "../MobileSAM/weights/mobile_sam.pt"
    

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        predictor = SamPredictor(mobile_sam)
    else:
        sam = sam_model_registry["vit_h"](checkpoint="../checkpoints/sam_vit_h_4b8939.pth")
        sam.to(device=device)
        # mask_generator = SamAutomaticMaskGenerator(sam)
        predictor = SamPredictor(sam)    


    return predictor

