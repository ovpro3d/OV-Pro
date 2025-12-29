import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict



class HeightCompressionTS(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = self.get_bev_features(encoded_spconv_tensor) 
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        if self.training and 'multi_scale_3d_features_mm' in batch_dict:
            if 'encoded_spconv_tensor_mm' in batch_dict:
                sp_tensor_mm = batch_dict['encoded_spconv_tensor_mm']
            else:
                sp_tensor_mm = batch_dict['multi_scale_3d_features_mm']['x_conv4']
            
            spatial_features_mm = self.get_bev_features(sp_tensor_mm)
            
            batch_dict['spatial_features_mm'] = spatial_features_mm
            if 'encoded_spconv_tensor_stride_mm' in batch_dict:
                batch_dict['spatial_features_stride_mm'] = batch_dict['encoded_spconv_tensor_stride_mm']
            else:
                batch_dict['spatial_features_stride_mm'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
    def get_bev_features(self, encoded_spconv_tensor):
        x = encoded_spconv_tensor.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)
        
        return x