import os
from PIL import Image

import torch
import torch.nn.functional as F
import argparse
import whisper
import numpy as np
import cv2
import glob
from scipy import ndimage
from scipy.spatial import distance

from torchvision import transforms
from detectron2.data import MetadataCatalog
from utils.visualizer import Visualizer

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# from demo.seem.tasks import *

import sys
sys.path.append('../novel_loss/')
from models.vgg import vgg19


# define a function to find the local maximum in the density map
def find_local_max(dmap, threshold=0.1):
    # 1. do max pooling
    max_dmap = ndimage.maximum_filter(dmap, size=5, mode='constant')
    # 2. get the local max
    local_max = dmap == max_dmap
    # 3. threshold
    local_max = local_max & (dmap > threshold)
    # 4. get the local max value
    coods = np.where(local_max)
    return np.array(coods).T[:,::-1]

# check the model prediction from point prompts
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

def semantic_sam_from_points(image, model, point): 
    h, w = image.size
    mask_ori = np.zeros((h, w, 3))
    mask_ori[int(point[1]), int(point[0]), 0] = 1

    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    data = {"image": images, "height": height, "width": width}

    

    model.model.task_switch['spatial'] = True
    mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
    mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[None,]

    mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0)
    data['stroke'] = mask_ori

    batch_inputs = [data]
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    v_emb = results['pred_maskembs']
    s_emb = results['pred_pspatials']
    pred_masks = results['pred_masks']

    pred_logits = v_emb @ s_emb.transpose(1,2)
    logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
    logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
    logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
    pred_masks_pos = pred_masks[logits_idx]
    pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]

    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
    texts = [all_classes[pred_class[0]]]
    for idx, mask in enumerate(pred_masks_pos):
        # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
        out_txt = texts[idx] 
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0]%133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()
    return res, pred_masks_pos, pred_class[0]

# define a function to reSAM the image
def ReSAM(image, model, coods):

    W, H = image.size
    refine_map = np.ones((H, W)) * -1 # -1 is uncertain, 0 is background, >0 is person


    size = 128
    dists = distance.cdist(coods, coods)
    np.fill_diagonal(dists, np.inf)

    id = 1
    seg_info = []
    points = []
    for cood, dist in zip(coods, dists):
        nn_dist = dist.min()
        if nn_dist < 64:
            size = 128
        elif nn_dist < 128:
            size = 256
        elif nn_dist < 256:
            size = 512
        else:
            size = 1024
        # evaluate the current point
        patch = image.crop((cood[0]-size//2, cood[1]-size//2, cood[0]+size//2, cood[1]+size//2))
        # offset point
        point = [size//2, size//2]
        res, mask, category_id = semantic_sam_from_points(patch, model, point)
        # resize mask to patch size
        mask = cv2.resize(mask[0], (size, size))


        
        x0 = max(0, cood[0]-size//2)
        x1 = min(cood[0]+size//2, refine_map.shape[1])
        y0 = max(0, cood[1]-size//2)
        y1 = min(cood[1]+size//2, refine_map.shape[0])

        mx0 = max(0, size//2-cood[0])
        mx1 = min(size, size//2-cood[0]+refine_map.shape[1])
        my0 = max(0, size//2-cood[1])
        my1 = min(size, size//2-cood[1]+refine_map.shape[0])

        if x0 >= x1 or y0 >= y1:
            continue
        if category_id == 0:
            # if the region is background
            if refine_map[y0:y1, x0:x1][mask[my0:my1, mx0:mx1] > 0].mean() < 0.3:
                points.append(cood)
                refine_map[y0:y1, x0:x1][mask[my0:my1, mx0:mx1] > 0] = id            
                seg_info.append({'id':id, 'isthing': True, 'category_id': category_id, 'point': cood})
                id += 1

    return refine_map, seg_info, np.array(points)

# define a function to localize large image
def pred_dmap(counter, image, transform):
    W, H = image.size
    dmap = np.zeros((H, W))
    print(dmap.shape)
    if W > 2048 or H > 2048:
        for i in range(0, W - 1024, 1024):
            for j in range(0, H - 1024, 1024):
                if i + 2048 > W and j + 2048 > H:
                    patch = image.crop((i, j, W, H))
                    maxx = W 
                    maxy = H 
                elif j + 2048 > H:
                    patch = image.crop((i, j, i + 1024, H))
                    maxx = i + 1024
                    maxy = H
                elif i + 2048 > W:
                    patch = image.crop((i, j, W, j + 1024))
                    maxx = W
                    maxy = j + 1024
                else:
                    patch = image.crop((i, j, i + 1024, j + 1024))
                    maxx = i + 1024
                    maxy = j + 1024
                patch_tensor = transform(patch).unsqueeze(0).cuda()
                with torch.no_grad():
                    pred, _ = counter(patch_tensor)
                pred = pred.detach().cpu().numpy()
                pred = cv2.resize(pred[0,0], (patch.size[0], patch.size[1]))
                # print(patch.size, pred.shape, dmap[j:maxy, i:maxx].shape)
                dmap[j:maxy, i:maxx] = pred
    else:
        img_tensor = transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            pred, _ = counter(img_tensor)
        pred = pred.detach().cpu().numpy()
        pred = cv2.resize(pred[0,0], (image.size[0], image.size[1]))
        dmap = pred

    return dmap


def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args('')
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

opt['MODEL']['DECODER']['TEST']['OBJECT_MASK_THRESHOLD'] = 0.3
print(opt['MODEL']['DECODER']['TEST']['OBJECT_MASK_THRESHOLD'])


# META DATA
cur_model = 'None'
if 'focalt' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
    cur_model = 'Focal-T'
elif 'focal' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focall_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))
    cur_model = 'Focal-L'

print(cur_model)

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

audio = whisper.load_model("base")


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225], inplace=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

dataset = sys.argv[1]

if 'ucf' in dataset:
    img_folder = 'UCF-QNRF_ECCV18/Train'
    des_folder = 'UCF_SAM/train_stage3_0'
    # counter_path = 'logs/1106_vgg19_ot_exp_100_test_best.pth'
    counter_path = 'logs/ucf_stage2_best.pth'
elif 'sha' in dataset:
    img_folder = 'ShanghaiTech/part_A/train_data/images'
    des_folder = 'ShanghaiTech_SAM/part_A/train_stage4_0'
    counter_path = 'logs/sha_stage3_best.pth'




counter = vgg19().cuda()
counter.load_state_dict(torch.load(counter_path))
counter.eval()

img_list = sorted(glob.glob(f'../../data/{img_folder}/*.jpg'))[::-1]
print(len(img_list))
des_dir = f'../../data/{des_folder}/'

# create folder is not exist
if not os.path.exists(des_dir):
    os.makedirs(des_dir)

for img_path in img_list:
    des_path = os.path.join(des_dir, img_path.split('/')[-1].replace('.jpg', '.npy'))
    if os.path.exists(des_path):
        continue
    image = Image.open(img_path).convert('RGB') 

    pred = pred_dmap(counter, image, transform)
    coods = find_local_max(pred)

    # use coods as prompts to relocaize more persons in the image
    person_map, seg_info, points = ReSAM(image, model, coods)

    # save refine_map
    # resize to half of the original image size
    person_map = cv2.resize(person_map, (image.size[0]//2, image.size[1]//2), interpolation=cv2.INTER_NEAREST)
    person_map = person_map.astype(np.int16)
    # save to disk
    data_dict = {}
    data_dict['seg_info'] = seg_info
    data_dict['points'] = points
    data_dict['person_map'] = person_map
    np.save(des_path, data_dict)
    print(f'{img_path.split("/")[-1]} saved to {des_path}')