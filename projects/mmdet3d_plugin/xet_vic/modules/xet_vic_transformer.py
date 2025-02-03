# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16
from .transformer import PerceptionTransformer
@TRANSFORMER.register_module()
class XETVICPerceptionTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=2,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_timestamps=True,
                 timestamps_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(XETVICPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_timestamps = use_timestamps
        self.timestamps_norm = timestamps_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
            
        self.timestamps_mlp = nn.Sequential(
            nn.Linear(1, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.timestamps_norm:
            self.timestamps_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    
    @auto_fp16(apply_to=('mmdl_mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mmdl_mlvl_feats,
            mmdl_metas,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            ):
        """
        obtain bev features.
        """
        mdls = len(mmdl_mlvl_feats)
        bs = mmdl_mlvl_feats[0][0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        # obtain rotation angle and shift with ego motion
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = mmdl_metas[0][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        
        # add can bus signals
        can_bus = bev_queries.new_tensor([each['can_bus'] for each in mmdl_metas[0]])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        # yzw: add timestamps signals
        timestamps = mmdl_mlvl_feats[0][0].new_tensor(mmdl_metas[0][0]["timestamps"]).unsqueeze(-1)

        delta_x = np.array([each['can_bus'][0] for each in mmdl_metas[0]])
        delta_y = np.array([each['can_bus'][1] for each in mmdl_metas[0]])
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in mmdl_metas[0]])
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0) 

        mmdl_feat_flatten = []
        mmdl_spatial_shapes = []
        mmdl_level_start_index = []
        for i_mdl in range(mdls):
            feat_flatten = []
            spatial_shapes = []
            for lvl, feat in enumerate(mmdl_mlvl_feats[i_mdl]):
                bs, num_cam, c, h, w = feat.shape
                spatial_shape = (h, w)
                feat = feat.flatten(3).permute(1, 0, 3, 2)
                if self.use_cams_embeds:
                    feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
                if self.use_timestamps:
                    feat = feat + self.timestamps_mlp(timestamps)[:, None, None, :].to(feat.dtype) 
                spatial_shapes.append(spatial_shape)
                feat_flatten.append(feat)
            feat_flatten = torch.cat(feat_flatten, 2)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            feat_flatten = feat_flatten.permute(0, 2, 1, 3) 
            
            mmdl_feat_flatten.append(feat_flatten)
            mmdl_spatial_shapes.append(spatial_shapes)
            mmdl_level_start_index.append(level_start_index)

        bev_embed = self.encoder(
            bev_queries, 
            mmdl_feat_flatten, 
            mmdl_feat_flatten, 
            bev_h=bev_h, 
            bev_w=bev_w, 
            bev_pos=bev_pos, 
            mmdl_spatial_shapes=mmdl_spatial_shapes,
            mmdl_level_start_index=mmdl_level_start_index, 
            prev_bev=prev_bev,
            shift=shift,
            mmdl_metas=mmdl_metas,
            )

        return bev_embed
    
    def get_states_and_refs(
        self,
        bev_embed,
        object_query_embed,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None
    ):
        bs = bev_embed.shape[1]
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.sigmoid()

        init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas
        )
        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out

