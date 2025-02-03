#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os 
from .uniad_track import  UniADTrack
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet3d.models.builder import build_backbone, build_neck, build_voxel_encoder, build_middle_encoder
import mmcv
from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
from mmdet3d.ops import Voxelization
from torch.nn import functional as F

@DETECTORS.register_module()
class VICTrackNEW(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        voxel_layer,
        voxel_layer_inf,
        voxel_encoder,
        voxel_encoder_inf,
        middle_encoder,
        pc_backbone,
        pc_neck=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        is_cooperation=True,
        freeze_voxel_encoder=False,
        freeze_middle_encoder=False,
        freeze_pc_backbone=False,
        freeze_pc_neck=False,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        pc_range_inf=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        with_pc_neck=True,
        **kwargs,
    ):
        super(VICTrackNEW, self).__init__(
            is_cooperation=is_cooperation,
            pc_range=pc_range,
            inf_pc_range=pc_range_inf,
            post_center_range=post_center_range,
            freeze_img_backbone=freeze_img_backbone,
            freeze_img_neck=freeze_img_neck,
            freeze_bn=freeze_bn,
            **kwargs)
        if pc_backbone:
            self.pc_backbone = build_backbone(pc_backbone)
        if pc_neck:
            self.neck = build_neck(pc_neck)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_layer_inf = Voxelization(**voxel_layer_inf)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        self.voxel_encoder_inf = build_voxel_encoder(voxel_encoder_inf)
        self.middle_encoder = build_middle_encoder(middle_encoder)
        
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}   
        self.is_cooperation = is_cooperation
        
        if freeze_voxel_encoder:
            if freeze_bn:
                self.voxel_encoder.eval()
                self.voxel_encoder_inf.eval()
            for param in self.voxel_encoder.parameters():
                param.requires_grad = False
            for param_inf in self.voxel_encoder_inf.parameters():
                param_inf.requires_grad = False
        
        if freeze_middle_encoder:
            if freeze_bn:
                self.middle_encoder.eval()
            for param in self.middle_encoder.parameters():
                param.requires_grad = False
        
        if freeze_pc_backbone:
            if freeze_bn:
                self.pc_backbone.eval()
            for param in self.pc_backbone.parameters():
                param.requires_grad = False
        
        if freeze_pc_neck:
            if freeze_bn:
                self.neck.eval()
            for param in self.neck.parameters():
                param.requires_grad = False
        
    # def extract_feat_points(self, points, img_metas=None):
    #     """Extract features from points."""
    #     voxels, num_points, coors = self.voxelize(points)
    #     voxel_features = self.voxel_encoder(voxels, num_points, coors)
    #     batch_size = coors[-1, 0].item() + 1
    #     x = self.middle_encoder(voxel_features, coors, batch_size)
    #     x = self.backbone(x)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def voxelize_inf(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res.contiguous()
            res_voxels, res_coors, res_num_points = self.voxel_layer_inf(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def extract_pc_feat_queue(self, pc_queue, len_queue=None):
        num_lidars = len(pc_queue)
        if len_queue is not None:
            pc_feats_list = []
            for i in range(len_queue):
                pc_feats_list_single = []
                if num_lidars == 1:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats)
                elif num_lidars == 2:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats)
                    pc_feats_inf = self.extract_pc_feat_inf(pc=[pc_queue[1][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list_single.append(pc_feats_inf)
                # for pc_queue_vi in pc_queue:
                #     pc_feats = self.extract_pc_feat(pc=[pc_queue_vi[i][:,0:4]])  # bs, 1, C, H, W
                #     pc_feats_list_single.append(pc_feats)
                # pc_feats_list.append(pc_feats_list_vi)

                pc_feats_list_single_combined = []
                for lvl in range(len(pc_feats_list_single[0])):
                    pc_feats_single_qn_combined = torch.cat([pc_feats_list_single[j][lvl] for j in range(len(pc_feats_list_single))], dim=1)
                    pc_feats_single_qn_combined = torch.unsqueeze(pc_feats_single_qn_combined, dim=1)
                    pc_feats_list_single_combined.append(pc_feats_single_qn_combined)
                pc_feats_list.append(pc_feats_list_single_combined)
            pc_feats_list_combined = []
            for i in range(len(pc_feats_list[0])):
                pc_feats_single_q_combined = torch.cat([pc_feats_list[j][i] for j in range(len_queue)], dim=1)
                pc_feats_list_combined.append(pc_feats_single_q_combined)
            return pc_feats_list_combined
        else:
            pc_feats_list = []
            len_queue = len(pc_queue[0])
            for i in range(len_queue):
                if num_lidars == 1:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list.append(pc_feats)
                elif num_lidars == 2:
                    pc_feats = self.extract_pc_feat(pc=[pc_queue[0][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list.append(pc_feats)
                    pc_feats_inf = self.extract_pc_feat_inf(pc=[pc_queue[1][i][:,0:4]])  # bs, 1, C, H, W
                    pc_feats_list.append(pc_feats_inf)
                # for pc_queue_vi in pc_queue:
                #     pc_feats = self.extract_pc_feat(pc=[pc_queue_vi[i][:,0:4]])  # bs, 1, C, H, W
                #     pc_feats_list.append(pc_feats)
            pc_feats_list_combined = []
            for i in range(len(pc_feats_list[0])):
                pc_feats_combined = torch.cat([pc_feats_list[j][i] for j in range(len(pc_feats_list))], dim=1)
                pc_feats_list_combined.append(pc_feats_combined)
            return pc_feats_list_combined
        
        # pc_queue_veh = pc_queue[0]
        # pc_queue_inf = pc_queue[1]
        # len_queue = len(pc_queue_veh)
        # # bs, len_queue, num_cams, C, H, W = pc_queue.shape
        # # pc_queue = pc_queue.reshape(bs * len_queue, num_cams, C, H, W)
        # pc_feats_list_veh = []
        # pc_feats_list_inf = []
        # pc_feats_list = []
        # for i in range(len_queue):
        #     pc_feats = self.extract_pc_feat(pc=[pc_queue_veh[i][:,0:4]], len_queue=len_queue)
        #     pc_feats_list_veh.append(pc_feats)
        # for i in range(len_queue):
        #     pc_feats = self.extract_pc_feat(pc=[pc_queue_inf[i][:,0:4]], len_queue=len_queue)
        #     pc_feats_list_inf.append(pc_feats)
        # for i in range(len(pc_feats_list_inf)):
        #     tmp = []
        #     for j in range(len(pc_feats_list_inf[0])):
        #         tmp.append(torch.cat((pc_feats_list_veh[i][j],pc_feats_list_inf[i][j]),1))
        #     pc_feats_list.append(tmp)
        # return pc_feats_list
    
    def extract_pc_feat(self, pc):
        """Extract features of pointclouds."""
        voxels, num_points, coors = self.voxelize(pc)
        voxel_features = self.voxel_encoder(voxels, num_points, coors) # (10600,4)
        batch_size = coors[-1, 0].item() + 1  
        x = self.middle_encoder(voxel_features, coors, batch_size) #([1, 64, 512, 512])
        x = self.pc_backbone(x) # ([1, 64, 256, 256]),(1,128,128,128),(1,256,64,64)
        if self.with_neck:
            x = list(self.neck(x)) # ([1, 3*256, 256, 256])  NCHW
        for i in range(len(x)):
            x[i] = x[i].unsqueeze(0)
        return x
        # if pc is None:
        #     return None
        # assert pc.dim() == 5
        # B, N, C, H, W = pc.size()
        # pc = pc.reshape(B * N, C, H, W)
        # if self.use_grid_mask:
        #     pc = self.grid_mask(pc)
        # pc_feats = self.pc_backbone(pc)
        # if isinstance(pc_feats, dict):
        #     pc_feats = list(pc_feats.values())
        # if self.with_pc_neck:
        #     pc_feats = self.pc_neck(pc_feats)

        # pc_feats_reshaped = []
        # for pc_feat in pc_feats:
        #     _, c, h, w = pc_feat.size()
        #     if len_queue is not None:
        #         pc_feat_reshaped = pc_feat.view(B//len_queue, len_queue, N, c, h, w)
        #     else:
        #         pc_feat_reshaped = pc_feat.view(B, N, c, h, w)
        #     pc_feats_reshaped.append(pc_feat_reshaped)
        # return pc_feats_reshaped
    
    def extract_pc_feat_inf(self, pc):
        """Extract features of pointclouds."""
        voxels, num_points, coors = self.voxelize_inf(pc)
        voxel_features = self.voxel_encoder_inf(voxels, num_points, coors) # (10600,4)
        batch_size = coors[-1, 0].item() + 1  
        x = self.middle_encoder(voxel_features, coors, batch_size) #([1, 64, 512, 512])
        x = self.pc_backbone(x) # ([1, 64, 256, 256]),(1,128,128,128),(1,256,64,64)
        if self.with_neck:
            x = list(self.neck(x)) # ([1, 3*256, 256, 256])  NCHW
        for i in range(len(x)):
            x[i] = x[i].unsqueeze(0)
        return x

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('points_veh', 'points_inf'))
    def forward_train(self,
                    #   pc=None,
                    #   points=None,
                      points_veh = None,
                      points_inf = None,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                    #   timestamp_veh=None,
                    #   timestamp_inf=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,

                      #for coop
                      veh2inf_rt=None,

                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        if self.is_cooperation:
            pc = [points_veh,points_inf]
        else:
            pc = [points_veh]
        losses_track, outs_track = self.forward_track_train(pc, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    @auto_fp16(apply_to=("pc"))
    def forward_track_train(self,
                            pc,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp,
                            veh2inf_rt):
        """Forward funciton
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = len(pc[0][0])
        # init gt instances!
        gt_instances_list = []
        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(pc[0][0][0].device)
            # normalize gt bboxes here!
            boxes = normalize_bbox(boxes, self.pc_range)
            sd_boxes = gt_sdc_bbox[0][i].tensor.to(pc[0][0][0].device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i].float()
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()
        for i in range(num_frame):
            prev_img_metas = copy.deepcopy(img_metas)
            prev_pc = []
            pc_single = []
            for pc_vi in pc:
                prev_pc_vi = pc_vi[0][:i] if i != 0 else pc_vi[0][:1]
                pc_single_vi = torch.stack([pc_[i] for pc_ in pc_vi], dim=0)
                prev_pc.append(prev_pc_vi)
                pc_single.append(pc_single_vi)
            
            # pc_veh = pc[0]
            # # prev_pc = pc[:, :i, ...] if i != 0 else pc[:, :1, ...]
            # prev_pc_veh = pc_veh[0][:i] if i != 0 else pc_veh[0][:1]
            # prev_img_metas = copy.deepcopy(img_metas)
            # # TODO: Generate prev_bev in an RNN way.
            # pc_single_veh = torch.stack([pc_[i] for pc_ in pc_veh], dim=0)

            # pc_inf = pc[1]
            # # prev_pc = pc[:, :i, ...] if i != 0 else pc[:, :1, ...]
            # prev_pc_inf = pc_inf[0][:i] if i != 0 else pc_inf[0][:1]
            # # TODO: Generate prev_bev in an RNN way.
            # pc_single_inf = torch.stack([pc_[i] for pc_ in pc_inf], dim=0)


            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else: 
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            # frame_res = self._forward_single_frame_train_coop(
            frame_res = self._forward_single_frame_train(
                pc_single,
                # [pc_single_veh,pc_single_inf],
                img_metas_single,
                track_instances,
                prev_pc,
                # [prev_pc_veh,prev_pc_inf],
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
                veh2inf_rt
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]

        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        return losses, out
    
    def _forward_single_frame_train(
        self,
        pc,
        img_metas,
        track_instances,
        prev_pc,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
        veh2inf_rt=None
        ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        # import pdb;pdb.set_trace()
        bev_embed, bev_pos = self.get_bevs(
            pc, img_metas,
            prev_pc=prev_pc, prev_img_metas=prev_img_metas,
        )
        
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        # track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        # out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        out.update(self.select_sdc_track_query(track_instances[-1], img_metas))
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out
    
    def forward_test(self,
                     points_veh = None,
                     points_inf = None,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     #for coop
                     veh2inf_rt=None,
                     **kwargs
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        # img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        if self.is_cooperation:
            pc = [points_veh[0],points_inf[0]]
        else:
            pc = points_veh

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(pc, l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        # if self.with_seg_head:
        #     result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale, veh2inf_rt=veh2inf_rt)

        # if self.with_motion_head:
        #     result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
        #     outs_motion['bev_pos'] = result_track[0]['bev_pos']

        # outs_occ = dict()
        # if self.with_occ_head:
        #     occ_no_query = outs_motion['track_query'].shape[1] == 0
        #     outs_occ = self.occ_head.forward_test(
        #         bev_embed, 
        #         outs_motion,
        #         no_query = occ_no_query,
        #         gt_segmentation=gt_segmentation,
        #         gt_instance=gt_instance,
        #         gt_img_is_valid=gt_occ_img_is_valid,
        #     )
        #     result[0]['occ'] = outs_occ
        
        # if self.with_planning_head:
        #     planning_gt=dict(
        #         segmentation=gt_segmentation,
        #         sdc_planning=sdc_planning,
        #         sdc_planning_mask=sdc_planning_mask,
        #         command=command
        #     )
        #     result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
        #     result[0]['planning'] = dict(
        #         planning_gt=planning_gt,
        #         result_planning=result_planning,
        #     )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        # if self.with_seg_head:
        #     result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        # if self.with_motion_head:
        #     result_motion[0] = pop_elem_in_result(result_motion[0])
        # if self.with_occ_head:
        #     result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
        #         pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])

        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            # if self.with_motion_head:
            #     res.update(result_motion[i])
            # if self.with_seg_head:
            #     res.update(result_seg[i])

        return result
    
    def simple_test_track(
        self,
        pc,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        veh2inf_rt=None
        ):
        """only support bs=1 and sequential input"""

        # bs = img.size(0)
        bs = len(pc[0])

        # img_metas = img_metas[0]

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
        
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            pc,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances
                
        results = [dict()]
        get_keys = ["bev_embed", "bev_pos", 
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        # if self.with_motion_head:
        #     get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})

        ## UniV2X: inf_track_query
        if self.save_track_query:
            tensor_to_cpu = torch.zeros(1)
            save_path = os.path.join(self.save_track_query_file_root, img_metas[0]['sample_idx'] +'.pkl')
            track_instances = track_instances.to(tensor_to_cpu)
            mmcv.dump(track_instances, save_path)

        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _forward_single_frame_inference(
        self,
        pc,
        img_metas,
        track_instances,
        prev_pc=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None
        ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            # active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])                  
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(pc, img_metas, prev_pc=prev_pc)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed, 
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    @auto_fp16(apply_to=("pc", "prev_bev"))
    def get_bevs(self, pc, img_metas, prev_pc=None, prev_img_metas=None, prev_bev=None):
        if prev_pc is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_pc, prev_img_metas)
        pc_feats = self.extract_pc_feat_queue(pc_queue=pc)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(mlvl_feats=pc_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=pc_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos
    
    def get_history_bev(self, pc_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            len_queue = len(pc_queue[0])
            pc_feats_list = self.extract_pc_feat_queue(pc_queue, len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                pc_feats = [each_scale[:, i] for each_scale in pc_feats_list]
                # pc_feats = pc_feats_list[i]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(mlvl_feats=pc_feats, img_metas=img_metas, prev_bev=prev_bev)
        self.train()
        return prev_bev
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict
    
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """

        from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
        import mmcv
        from os import path as osp
        from mmcv.parallel import DataContainer as DC
        
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            # inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            # pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            inds = result[batch_id]['scores_3d'].numpy() > 0.1
            pred_bboxes = result[batch_id]['boxes_3d'][inds]

            if len(pred_bboxes) <= 0:
                continue

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for convertion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
