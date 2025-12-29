import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeInstanceMatching(nn.Module):
    def __init__(self, roi_size, point_cloud_range, voxel_size, out_size_factor=8):
        super().__init__()
        self.point_cloud_range = torch.tensor(point_cloud_range).float()
        self.voxel_size = torch.tensor(voxel_size).float()
        self.out_size_factor = out_size_factor
        self.roi_size = roi_size

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_2d: (B, C, H, W) Student BEV features (f_s)
                spatial_features_2d_mm: (B, C, H, W) Teacher BEV features (f_t)
                gt_boxes: (B, N, 7+C) 3D Boxes [x, y, z, dx, dy, dz, heading, label, ...]
        Returns:
            batch_dict: 更新包含 student_prototypes, teacher_prototypes, prototype_negative_mask 的字典
        """
        f_s = batch_dict['spatial_features_2d']
        
        # 必须存在教师特征且有 GT Boxes 才能进行匹配
        if 'spatial_features_2d_mm' not in batch_dict:
            return batch_dict
        f_t = batch_dict['spatial_features_2d_mm']
        
        boxes = batch_dict.get('gt_boxes', None)
        if boxes is None:
            return batch_dict

        batch_size = f_s.shape[0]
        p_s_list = []      # 学生原型特征列表
        p_t_list = []      # 教师原型特征列表
        batch_idx_list = [] 
        label_list = []    # 存储类别标签，用于生成负样本 Mask

        for b in range(batch_size):
            cur_boxes = boxes[b]
            # 过滤 Padding (全0) 的 Box
            mask = (torch.abs(cur_boxes).sum(dim=-1) > 0)
            cur_boxes = cur_boxes[mask]
            
            if cur_boxes.shape[0] == 0:
                continue
                
            # 1. 坐标映射: 3D Box -> BEV Grid ROI
            rois_bev = self.box3d_to_bev_rois(cur_boxes, f_s.device)
            
            # 2. ROI Align / Crop
            cur_f_s = f_s[b].unsqueeze(0) # (1, C, H, W)
            cur_f_t = f_t[b].unsqueeze(0) # (1, C, H, W)
            
            crops_s = self.roi_align_rotated(cur_f_s, rois_bev, self.roi_size)
            crops_t = self.roi_align_rotated(cur_f_t, rois_bev, self.roi_size)
            
            # 3. Average Pooling 得到原型特征 (N, C)
            p_s = F.avg_pool2d(crops_s, kernel_size=self.roi_size).view(crops_s.shape[0], -1)
            p_t = F.avg_pool2d(crops_t, kernel_size=self.roi_size).view(crops_t.shape[0], -1)
            
            p_s_list.append(p_s)
            p_t_list.append(p_t)
            
            # 记录 Batch Index
            num_boxes = cur_boxes.shape[0]
            batch_idx_list.append(torch.full((num_boxes,), b, dtype=torch.long, device=f_s.device))
            
            # 记录 Labels (OpenPCDet中gt_boxes最后一列通常是Label)
            # 检查是否有 label 列 (N, >=8) -> index 7 是 label
            if cur_boxes.shape[1] >= 8:
                label_list.append(cur_boxes[:, 7].long())

        if len(p_s_list) == 0:
            return batch_dict

        # 4. 拼接 Batch 内所有数据
        all_p_s = torch.cat(p_s_list, dim=0)
        all_p_t = torch.cat(p_t_list, dim=0)
        all_batch_idx = torch.cat(batch_idx_list, dim=0)
    
        batch_dict['student_prototypes'] = all_p_s
        batch_dict['teacher_prototypes'] = all_p_t
        batch_dict['prototype_batch_indices'] = all_batch_idx
        

        if len(label_list) > 0:
            all_labels = torch.cat(label_list, dim=0) # (Total_Instances,)
            batch_dict['prototype_labels'] = all_labels

            negative_mask = all_labels.unsqueeze(0) != all_labels.unsqueeze(1)
            batch_dict['prototype_negative_mask'] = negative_mask

        return batch_dict

    def box3d_to_bev_rois(self, boxes3d, device):
        pc_range = self.point_cloud_range.to(device)
        voxel_size = self.voxel_size.to(device)
        
        x_min, y_min = pc_range[0], pc_range[1]
        v_x, v_y = voxel_size[0], voxel_size[1]
        stride = self.out_size_factor
        
        center_x = (boxes3d[:, 0] - x_min) / (v_x * stride)
        center_y = (boxes3d[:, 1] - y_min) / (v_y * stride)
        dim_x = boxes3d[:, 3] / (v_x * stride)
        dim_y = boxes3d[:, 4] / (v_y * stride)
        heading = boxes3d[:, 6]
        
        rois = torch.stack([center_x, center_y, dim_x, dim_y, heading], dim=-1)
        return rois

    def roi_align_rotated(self, features, rois, output_size):
        num_rois = rois.shape[0]
        _, C, H, W = features.shape

        base_grid = F.affine_grid(
            torch.tensor([[[1, 0, 0], [0, 1, 0]]], device=features.device, dtype=features.dtype).repeat(num_rois, 1, 1),
            size=(num_rois, C, output_size, output_size),
            align_corners=True 
        )[:, 0:1, :, :].squeeze(1) 

        cx, cy, w, l, angle = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        grid_x = base_grid[..., 0] * (w.view(-1, 1, 1) / 2)
        grid_y = base_grid[..., 1] * (l.view(-1, 1, 1) / 2)

        rot_x = grid_x * cos_a.view(-1, 1, 1) - grid_y * sin_a.view(-1, 1, 1)
        rot_y = grid_x * sin_a.view(-1, 1, 1) + grid_y * cos_a.view(-1, 1, 1)
  
        sample_x = rot_x + cx.view(-1, 1, 1)
        sample_y = rot_y + cy.view(-1, 1, 1)

        norm_sample_x = (2.0 * sample_x / (W - 1)) - 1.0
        norm_sample_y = (2.0 * sample_y / (H - 1)) - 1.0
        
        sampling_grid = torch.stack([norm_sample_x, norm_sample_y], dim=-1)
        features_expanded = features.expand(num_rois, -1, -1, -1)

        crops = F.grid_sample(features_expanded, sampling_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        return crops