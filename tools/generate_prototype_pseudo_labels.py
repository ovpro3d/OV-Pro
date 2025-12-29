import argparse
import copy
import glob
from pathlib import Path
import random
import numpy as np
import torch
import tqdm
import os
import pickle
import sys
import yaml
import gc

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, NuScenesDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
DEFAULT_TRACKER_PATH = "/home/exps/OV-Pro/ImmortalTracker-for-CTRL"
if os.path.exists(DEFAULT_TRACKER_PATH) and DEFAULT_TRACKER_PATH not in sys.path:
    sys.path.insert(0, DEFAULT_TRACKER_PATH)
try:
    from mot_3d.mot import MOTModel
    from mot_3d.data_protos import BBox
    from mot_3d.frame_data import FrameData
    HAS_IMMORTAL_TRACKER = True
    print(f"Successfully loaded ImmortalTracker from: {DEFAULT_TRACKER_PATH}")
except ImportError:
    HAS_IMMORTAL_TRACKER = False
    MOTModel = None
    BBox = None
    FrameData = None
    print(f"Notice: 'mot_3d' package not found. \n"
          f"Checked path: {DEFAULT_TRACKER_PATH}\n"
          f"Please ensure the folder 'ImmortalTracker-for-CTRL' exists.")
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/nuscenes_cross_view_instance_consistency.yaml', help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--folder', type=str, default="../data/pseudo_labels/nuscenes_prototypes/", help='folder name for extracted prototypes')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--tracker_path', type=str, default=DEFAULT_TRACKER_PATH, help='Path to the ImmortalTracker-for-CTRL repository')   
    default_tracker_config = os.path.join(DEFAULT_TRACKER_PATH, 'configs/nu_configs/immortal.yaml')
    parser.add_argument('--tracker_config', type=str, default=default_tracker_config, help='Path to the tracker config file')
    parser.add_argument('--skip_inference', action='store_true', help='Skip Phase 1 if pseudo labels already exist')    
    args = parser.parse_args()
    assert args.batch_size == 1, 'Prototype clustering requires batch_size == 1 to handle sequence tracking correctly'
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

class ImmortalTrackerWrapper:
    def __init__(self, tracker_path, config_path):
        global MOTModel, BBox, FrameData
        self.config_path = config_path
        
        if not HAS_IMMORTAL_TRACKER:
            if tracker_path and tracker_path not in sys.path:
                sys.path.insert(0, tracker_path)
            try:
                from mot_3d.mot import MOTModel as MM
                from mot_3d.data_protos import BBox as BB
                from mot_3d.frame_data import FrameData as FD
                MOTModel = MM
                BBox = BB
                FrameData = FD
            except ImportError as e:
                raise ImportError(f"Could not import 'mot_3d' modules from {tracker_path}. "
                                  f"Please check the path and dependencies. Error: {e}")
        if not os.path.exists(config_path):
             alt_config = os.path.join(tracker_path, 'configs/nu_configs/immortal.yaml')
             if os.path.exists(alt_config):
                 config_path = alt_config
                 print(f"Using auto-detected config path: {config_path}")
             else:
                 raise FileNotFoundError(f"Tracker config not found at {config_path}")
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        self.tracker = MOTModel(self.configs)
        self.frame_count = 0
        self.prev_timestamp = 0

    def reset(self):
        self.tracker = MOTModel(self.configs)
        self.frame_count = 0
        self.prev_timestamp = 0

    def update(self, boxes_global, labels, scores, class_names, timestamp=None, pose=None, points=None):
        if timestamp is None:
            time_lag = 0.1 
        else:
            if self.frame_count == 0:
                time_lag = 0.1 
            else:
                time_lag = timestamp - self.prev_timestamp
                if time_lag <= 0: time_lag = 0.1 
            
            self.prev_timestamp = timestamp

        detections = []
        det_types = []
        for i in range(len(boxes_global)):
            box = boxes_global[i]
            
            try:
                type_name = class_names[labels[i] - 1]
            except IndexError:
                type_name = 'unknown'

            bb = BBox(
                x=box[0], y=box[1], z=box[2],
                w=box[4], l=box[3], h=box[5],
                o=box[6],
                s=scores[i]
            )
            detections.append(bb)
            det_types.append(type_name)

        points_global = points
        if points is not None and pose is not None:
            xyz = points[:, :3]
            xyz_hom = np.pad(xyz, ((0,0),(0,1)), constant_values=1)
            xyz_global = (pose @ xyz_hom.T).T[:, :3]
            
            if points.shape[1] > 3:
                points_global = np.concatenate([xyz_global, points[:, 3:]], axis=1)
            else:
                points_global = xyz_global
        aux_info = {'is_key_frame': True}
        
        frame_data = FrameData(
            dets=detections,
            ego=pose, 
            pc=points_global,
            det_types=det_types,
            aux_info=aux_info,
            time_stamp=timestamp
        )
        results = self.tracker.frame_mot(frame_data)
        self.frame_count += 1        
        assigned_track_ids = [-1] * len(boxes_global)
        
        track_boxes = []
        track_ids = []
        
        for trk in results:
            bbox_obj = trk[0]
            track_boxes.append(BBox.bbox2array(bbox_obj))
            track_ids.append(trk[1])
            
        if len(track_boxes) > 0 and len(boxes_global) > 0:
            track_boxes = np.array(track_boxes) 
            input_boxes = boxes_global 
            
            dists = np.linalg.norm(input_boxes[:, :3][:, None, :] - track_boxes[:, :3][None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)            
            for i in range(len(input_boxes)):
                if min_dists[i] < 1.0:
                    matched_idx = min_indices[i]
                    assigned_track_ids[i] = track_ids[matched_idx]
                    
        return assigned_track_ids

def transform_to_canonical(points, box):
    if points.shape[0] == 0:
        return points
    points = points - box[:3] 
    rot = -box[6]
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    
    x = points[:, 0] * cos_r - points[:, 1] * sin_r
    y = points[:, 0] * sin_r + points[:, 1] * cos_r
    z = points[:, 2]
    
    return np.stack([x, y, z], axis=-1)
def run_phase_1_inference(model, loader, base_path, score_thresh=0.0):
    boxes_save_dir = os.path.join(base_path, 'pseudo_labels')
    if not os.path.exists(boxes_save_dir):
        os.makedirs(boxes_save_dir, exist_ok=True)    
    print("\n" + "="*50)
    print(f"PHASE 1: INFERENCE -> {boxes_save_dir}")
    print("="*50)
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(loader), leave=True, desc='Phase 1: Inference', dynamic_ncols=True)
        
        for i, data_dict in enumerate(loader):
            frame_ids = data_dict['frame_id']
            frame_id_str = str(frame_ids[0]).replace('.', '_')
            save_path = os.path.join(boxes_save_dir, f"{frame_id_str}.pth")

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            torch.save(pred_dicts, save_path)
            
            pbar.update(1)
    
    print("Phase 1 Complete. All detection results saved.\n")

def run_phase_2_tracking(tracker, loader, base_path, cfg, score_thresh=0.4):
    print("\n" + "="*50)
    print("PHASE 2: LOADING RESULTS & TRACKING & CLUSTERING")
    print("="*50)    
    boxes_save_dir = os.path.join(base_path, 'pseudo_labels')
    prototypes = {}
    prev_scene_token = None    
    pbar = tqdm.tqdm(total=len(loader), leave=True, desc='Phase 2: Tracking', dynamic_ncols=True)
    with torch.no_grad():
        for i, data_dict in enumerate(loader):
            frame_ids = data_dict['frame_id']
            frame_id_str = str(frame_ids[0]).replace('.', '_')
            load_path = os.path.join(boxes_save_dir, f"{frame_id_str}.pth")            
            if not os.path.exists(load_path):
                pbar.update(1)
                continue
            load_data_to_gpu(data_dict)
            current_scene_token = None
            current_timestamp = None
            current_pose = None
            if 'metadata' in data_dict:
                meta = data_dict['metadata'][0]
                if isinstance(meta, dict):
                    if 'scene_token' in meta:
                        current_scene_token = meta['scene_token']
                    if 'timestamp' in meta:
                        current_timestamp = float(meta['timestamp']) / 1e6
                    if 'pose' in meta:
                        current_pose = meta['pose']
            if prev_scene_token is not None and current_scene_token != prev_scene_token:
                tracker.reset()
            prev_scene_token = current_scene_token
            pred_dicts = torch.load(load_path)
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()            
            mask = pred_scores > score_thresh
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]            
            if len(pred_boxes) == 0:
                pbar.update(1)
                continue
            boxes_global = copy.deepcopy(pred_boxes)
            if current_pose is not None:
                centers_local = np.pad(pred_boxes[:, :3], ((0,0),(0,1)), constant_values=1) 
                centers_global = (current_pose @ centers_local.T).T 
                boxes_global[:, :3] = centers_global[:, :3]                
                rot_matrix = current_pose[:3, :3]
                headings = pred_boxes[:, 6]
                vecs = np.stack([np.cos(headings), np.sin(headings), np.zeros_like(headings)], axis=1)
                vecs_global = (rot_matrix @ vecs.T).T
                headings_global = np.arctan2(vecs_global[:, 1], vecs_global[:, 0])
                boxes_global[:, 6] = headings_global            
            batch_points = data_dict['points']
            batch_points = batch_points[batch_points[:, 0] == 0][:, 1:4] 
            batch_points_np = batch_points.cpu().numpy()
            track_ids = tracker.update(
                boxes_global, pred_labels, pred_scores, 
                class_names=cfg.CLASS_NAMES,
                timestamp=current_timestamp,
                pose=current_pose,
                points=batch_points_np 
            )            
            batch_points_cuda = batch_points
            pred_boxes_cuda = torch.tensor(pred_boxes, device=batch_points.device, dtype=torch.float32)            
            point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
                batch_points_cuda.unsqueeze(0),
                pred_boxes_cuda.unsqueeze(0)
            ).long().squeeze(0) 

            for box_idx, track_id in enumerate(track_ids):
                if track_id == -1:
                    continue
                pt_mask = (point_indices == box_idx)
                if pt_mask.sum() == 0:
                    continue                
                points_in_box = batch_points_np[pt_mask.cpu().numpy()]                
                box_local_np = pred_boxes[box_idx]
                points_canonical = transform_to_canonical(points_in_box, box_local_np)
                unique_track_id = f"{current_scene_token}_{track_id}"
                
                if unique_track_id not in prototypes:
                    prototypes[unique_track_id] = {
                        'label': pred_labels[box_idx],
                        'points_list': [],
                        'box_sum': np.zeros(7),
                        'count': 0,
                        'appearances': [] 
                    }
                
                prototypes[unique_track_id]['points_list'].append(points_canonical)
                prototypes[unique_track_id]['box_sum'] += box_local_np
                prototypes[unique_track_id]['count'] += 1
                
                prototypes[unique_track_id]['appearances'].append({
                    'frame_id': frame_id_str,
                    'box_local': box_local_np,
                    'score': pred_scores[box_idx],
                    'box_idx': box_idx,
                    'raw_track_id': track_id 
                })

            pbar.update(1)

    return prototypes
def voxel_downsample(points, voxel_size=0.05):
    if len(points) == 0:
        return points
    
    voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    
    voxel_dict = {}
    for i, coord in enumerate(voxel_coords):
        key = tuple(coord)
        if key not in voxel_dict:
            voxel_dict[key] = i 
    
    indices = list(voxel_dict.values())
    return points[indices]
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    INF_CONFIG = copy.deepcopy(cfg.DATA_CONFIG)
    if 'DATA_AUGMENTOR' in INF_CONFIG:
        INF_CONFIG['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] = []
    
    inf_set, inf_loader, inf_sampler = build_dataloader(
        dataset_cfg=INF_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=False, 
        merge_all_iters_to_one_epoch=False,
        total_epochs=1,
    )

    if hasattr(inf_set, 'infos'):
        print(f"Sorting {len(inf_set.infos)} samples by scene and timestamp for tracking continuity...")
        inf_set.infos.sort(key=lambda x: (x.get('scene_token', 'unknown'), float(x.get('timestamp', 0))))
    else:
        logger.warning("Dataset does not have 'infos' attribute. Assuming sorted.")

    labels_id = args.folder
    base_path = labels_id
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if not args.skip_inference:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inf_set)
        if args.ckpt is not None:
            model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        
        model.cuda()
        run_phase_1_inference(model, inf_loader, base_path)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Skipping Phase 1 (Inference) as requested.")

    print(f"Initializing ImmortalTracker from {args.tracker_path}...")
    try:
        tracker = ImmortalTrackerWrapper(args.tracker_path, args.tracker_config)
    except Exception as e:
        logger.error(f"Failed to initialize ImmortalTracker: {e}\nTraceback: ", exc_info=True)
        return
    prototypes = run_phase_2_tracking(tracker, inf_loader, base_path, cfg)
    print("Finalizing: Distributing aggregated prototypes back to frame-based pseudo labels...")   
    save_dir_prototypes = os.path.join(base_path, 'pseudo_labels_prototypes')
    if not os.path.exists(save_dir_prototypes):
        os.makedirs(save_dir_prototypes, exist_ok=True)    
    frame_objects = {}    
    for unique_tid, data in tqdm.tqdm(prototypes.items(), desc="Aggregating Prototypes"):
        if data['count'] < 3: 
            continue            
        all_points = np.concatenate(data['points_list'], axis=0)
        
        if len(all_points) > 1000:
             all_points = voxel_downsample(all_points, voxel_size=0.05)
        if len(all_points) > 15000:
            choice = np.random.choice(len(all_points), 15000, replace=False)
            all_points = all_points[choice]            
        for appearance in data['appearances']:
            fid = appearance['frame_id']            
            if fid not in frame_objects:
                frame_objects[fid] = {
                    'boxes': [], 'scores': [], 'labels': [], 'track_ids': [], 'prototypes': []
                }            
            frame_objects[fid]['boxes'].append(appearance['box_local'])
            frame_objects[fid]['scores'].append(appearance['score'])
            frame_objects[fid]['labels'].append(data['label'])
            frame_objects[fid]['track_ids'].append(appearance['raw_track_id'])
            frame_objects[fid]['prototypes'].append(all_points)
    logger.info(f"Saving {len(frame_objects)} frame files to {save_dir_prototypes}...")    
    for fid, objs in tqdm.tqdm(frame_objects.items(), desc="Saving Frames"):
        pred_boxes = np.array(objs['boxes'])
        pred_scores = np.array(objs['scores'])
        pred_labels = np.array(objs['labels'])
        pred_prototypes = objs['prototypes']         
        output_dict = {
            'pred_boxes': torch.tensor(pred_boxes, dtype=torch.float32),
            'pred_scores': torch.tensor(pred_scores, dtype=torch.float32),
            'pred_labels': torch.tensor(pred_labels, dtype=torch.int64),
            'pred_prototypes': pred_prototypes # List of numpy arrays
        }        
        output_data = [output_dict]        
        save_path = os.path.join(save_dir_prototypes, f"{fid}.pth")
        torch.save(output_data, save_path)   
    logger.info(f"Done. Files saved to {save_dir_prototypes}")

if __name__ == '__main__':
    main()