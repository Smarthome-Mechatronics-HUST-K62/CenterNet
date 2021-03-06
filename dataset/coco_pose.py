import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import coco
from pycocotools.cocoeval import COCOeval
from dataset.utils import *

class COCOHP(Dataset):
    def __init__(self,img_dir,json_anno_path,train_mode=True,model_inp_size=512,down_ratio=4):
        self.train_mode = train_mode
        self.img_dir = img_dir
        self.coco = coco.COCO(json_anno_path)
        self.images_ids = self.coco.getImgIds()
        self.max_objs = 32
        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]
        self.model_inp_size = model_inp_size
        self.model_out_size = model_inp_size // down_ratio
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.num_joints = 17
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
              [11, 12], [13, 14], [15, 16]]
    
    def __len__(self):
        return len(self.images_ids)
    
    def __getitem__(self,idx):
        img_idx = self.images_ids[idx]

        #Get image, image center location and img_height vs img_width
        img_filename = self.coco.loadImgs([img_idx])[0]["file_name"]
        img = np.array(Image.open(os.path.join(self.img_dir,img_filename)))
        
        img_height = img.shape[0]
        img_width = img.shape[1]

        #Get Annos
        anno_ids = self.coco.getAnnIds([img_idx])
        annos = self.coco.loadAnns(anno_ids)
        
        #Reject some case with iscrowd = 1
        annos = list(filter(lambda x : x['category_id'] == 1 and x['iscrowd'] != 1, annos))
        num_objs = len(annos)

        #Get center and scale for affine transformation
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.train_mode:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = get_border(128, img.shape[1])
            h_border = get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  img_width - c[0] - 1
        
        
        trans_input = get_affine_transform(
          c, s, rot, [self.model_inp_size, self.model_inp_size])
        inp = cv2.warpAffine(img, trans_input, 
                             (self.model_inp_size, self.model_inp_size),
                             flags=cv2.INTER_LINEAR)
        
        inp = (inp.astype(np.float32) / 255.)
        if self.train_mode:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - np.array(self.mean).astype(np.float32)) / np.array(self.std).astype(np.float32)

        output_res = self.model_out_size
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
        
        hm = np.zeros((1, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_umich_gaussian
        gt_det = []
        
        for k in range(num_objs):
            ann = annos[k]
            bbox = coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            
            if flipped:
                bbox[[0, 2]] = img_width - bbox[[2, 0]] - 1
                pts[:, 0] = img_width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
             
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius =  max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                           
                #keypoint  x 2    
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = max(0, int(hp_radius)) 
                for j in range(num_joints):
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                            pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                               ct[0] + w / 2, ct[1] + h / 2, 1] + 
                               pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        ret = {
            "input": transforms.ToTensor()(inp),
            
            #for object detection
            "hm" : torch.from_numpy(hm).float(), # (1,output_res,output_res) 
            "reg": torch.from_numpy(reg).float(), # (max_objs,2)
            "wh" : torch.from_numpy(wh).float(), # (max_objs,2)
            "reg_mask": torch.from_numpy(reg_mask).long(), #(max_objs,)
            "inds": torch.from_numpy(ind).long(), #(max_objs,) (flat index )
            
            #for humanpose estimation
            "kps": torch.from_numpy(kps).float(), #(max_objs,num_joints * 2) : distance_x, distance_y from keypoints to center
            "kps_mask": torch.from_numpy(kps_mask).long(), #(max_objs,num_joints * 2)
            "hm_hp": torch.from_numpy(hm_hp).float(), #(num_joins,output_res,output_res)
            "hp_offset": torch.from_numpy(hp_offset).float(), #(max_objs * num_joints, 2)
            "hp_mask": torch.from_numpy(hp_mask).long(), # (max_objs * num_joints,)
            "hp_inds": torch.from_numpy(hp_ind).long() #(max_objs * num_joints,)
        }
        
        if self.train_mode != True:
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                   np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_idx': img_idx}
            ret['meta'] = meta
            
        return ret 

    
    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            category_id = 1
            for dets in all_bboxes[image_id][category_id]:
                bbox = dets[:4]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = dets[4]
                keypoint_prob = np.array(np.array(dets[39:56])>0.1).astype(np.int32).reshape(17,1)
                keypoints = np.array(dets[5:39], dtype=np.float32).reshape(-1, 2)
                bbox_out  = list(map(self._to_float, bbox))
                keypoints_pred = np.concatenate([
                keypoints, keypoint_prob], axis=1).reshape(51).tolist()
                keypoints_pred  = list(map(self._to_float, keypoints_pred))

                detection = {
                  "image_id": int(image_id),
                  "category_id": int(category_id),
                  "bbox": bbox_out,
                  "score": float("{:.2f}".format(score)),
                  "keypoints": keypoints_pred
                }
                detections.append(detection)
        return detections

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results):
        coco_dets = self.coco.loadRes(self.convert_eval_format(results))        
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]


# if __name__ == '__main__':
#     json_anno_path = "/home/hieunn/datasets/COCO/annotations/person_keypoints_val2017.json"
#     img_dir = "/home/hieunn/datasets/COCO/val/val2017"
#     ds = COCOHP(img_dir,json_anno_path)
#     print(len(ds))
#     ds[0]
#     for i in range(len(ds)):
#         ds[i]
        