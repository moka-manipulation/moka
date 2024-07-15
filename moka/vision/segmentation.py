from segment_anything import build_sam, SamPredictor
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import load_model, predict, annotate
import os
import groundingdino.datasets.transforms as T
from PIL import Image
import torch
from detectron2.utils.visualizer import GenericMask
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def mask_to_polygon(mask):
    h, w = mask.shape[:2]
    gm = GenericMask(mask, h, w)
    vertices = []
    for p in gm.polygons:
        xy = p.reshape(-1, 2)
        vertices.append(xy)
    vertices = np.concatenate(vertices, axis=0)
    return vertices


def load_pil_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_pil)
    image_transformed, _ = transform(image_pil, None)
    return image, image_transformed


def prepare_dino(dino_path):
    HOME = dino_path
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
    CONFIG_PATH = os.path.join('./config', "grounding_dino.py")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    return model


def get_object_bboxes(image_torch, texts):
    # get dino path from environment variable
    dino_path = './ckpts'

    model = prepare_dino(dino_path=dino_path)

    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.22 # can be tuned for different tasks
    all_boxes, all_logits, all_phrases = [], [], []

    for text in texts:
        boxes, logits, phrases = predict(
            model=model,
            image=image_torch,
            caption=text,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        id = torch.argmax(logits)
        all_boxes.append(boxes[id])
        all_logits.append(logits[id])
        all_phrases.append(phrases[id])

    # prompting order matters, and prompting one-by-one works the best
    return torch.stack(all_boxes, dim=0), torch.stack(all_logits, dim=0), all_phrases


def show_mask(masks, image, random_color=True):
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    for i in range(masks.shape[0]):
        mask = masks[i][0]
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, mask_image_pil)

    return np.array(annotated_frame_pil)


def get_scene_object_bboxes(image, texts, visualize=False, logdir=None):
    image_np, image_torch = load_pil_image(image)
    # get bounding box
    boxes, logits, phrases = get_object_bboxes(image_torch, texts)
    H, W, _ = image_np.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    if visualize:
        annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
        # %matplotlib inline
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        annotated_frame_pil = Image.fromarray(annotated_frame)

        if logdir is not None:
            annotated_frame_pil.save(os.path.join(logdir, 'bbox.png'))
        plt.imshow(annotated_frame_pil)
        plt.show()

    return boxes, logits, phrases


def get_segmentation_masks(image, texts, boxes, logits, phrases, visualize=False, logdir=None):
    image_np, image_torch = load_pil_image(image)
    # # get bounding box
    # boxes, logits, phrases = get_object_bboxes(image_torch, texts)
    #
    H, W, _ = image_np.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    if visualize:
        annotated_frame = annotate(image_source=image_np, boxes=boxes, logits=logits, phrases=phrases)
        # %matplotlib inline
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        annotated_frame_pil = Image.fromarray(annotated_frame)
        plt.imshow(annotated_frame_pil)
        plt.show()

    # get segmentation based on bounding box
    sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to('cuda')
    sam_predictor = SamPredictor(sam)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to('cuda')

    sam_predictor.set_image(image_np)
    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        masks = masks.cpu().numpy()

    if visualize:
        mask_annotated_img = show_mask(masks, annotated_frame)
        mask_annotated_img_pil = Image.fromarray(mask_annotated_img)
        plt.imshow(mask_annotated_img_pil)
        plt.show()
        if logdir is not None:
            mask_annotated_img_pil.save(os.path.join(logdir, 'mask.png'))

    # postprocess mask
    filtered_masks = {}
    masks = masks[:, 0]
    for i in range(len(phrases)):
        # choose corresponding object name
        obj_name = None
        for t in texts:
            if phrases[i] in t:
                obj_name = t
                break
        if obj_name is None:
            assert False, 'object name not found! error with detection module'

        cur_mask = masks[i]
        vertices = mask_to_polygon(cur_mask)
        if obj_name not in filtered_masks.keys():
            filtered_masks[obj_name] = {'mask': cur_mask, 'vertices': vertices}
        else:
            filtered_masks[obj_name]['mask'] = np.logical_or(cur_mask, filtered_masks[obj_name]['mask'])
            filtered_masks[obj_name]['vertices'] = np.concatenate([vertices, filtered_masks[obj_name]['vertices']], axis=0)

    # get bbox from segmask
    for k in filtered_masks.keys():
        vertices = filtered_masks[k]['vertices']
        x = vertices[:, 0]
        y = vertices[:, 1]
        filtered_masks[k]['bbox'] = [int(np.min(y)), int(np.min(x)), int(np.max(y)), int(np.max(x))]

    if len(filtered_masks.keys()) != len(texts):
        print('missing objects! error with detection module')

    return filtered_masks
