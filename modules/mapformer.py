import torch
import torch.nn as nn
from .base import BaseFormer
from .trackformer import TrackFormer, TrackFormerDecoderLayer
from .common import (
    ProtoSegModule, 
    RasterMapDetectionHead, 
    PosEmbedding1D, 
    SimpleMLP, 
    AddNorm
)
from .attentions import DeformableAttention, MultiHeadedAttention
from utils.img_utils import xyxy_to_xywh
from typing import Optional, Tuple, Union


class Box2ScaleStaticMethodMixin:
    @staticmethod
    def boxes2scale(dets: torch.Tensor, wh: Tuple[int, int]) -> torch.Tensor:
        """Expects boxes of shape (..., d) with d corresponding to [x, y, w, h, ...]"""
        box_max = torch.tensor([wh[0], wh[1], wh[0], wh[1]], device=dets.device)
        boxes   = box_max * dets[..., :4]
        return torch.concat([boxes, dets[..., 4:]], dim=-1)
    

class VectorMapKeypointDecoderLayer(TrackFormerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(VectorMapKeypointDecoderLayer, self).__init__(*args, **kwargs)


class PolyLineGeneratorLayer(nn.Module):
    def __init__(
            self,
            num_heads: int, 
            embed_dim: int,
            num_ref_points: int=4,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            bev_feature_hw: Tuple[int, int]=(200, 200),
            mask_out_ctx: bool=False
        ):
        super(PolyLineGeneratorLayer, self).__init__()

        self.num_heads       = num_heads
        self.embed_dim       = embed_dim
        self.num_ref_points  = num_ref_points
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.offset_scale    = offset_scale
        self.bev_feature_hw  = bev_feature_hw
        self.mask_out_ctx    = mask_out_ctx

        self.self_attention    = MultiHeadedAttention(
            self.num_heads, 
            self.embed_dim, 
            dropout=self.dropout,
        )
        self.addnorm1          = AddNorm(input_dim=self.embed_dim)

        self.deform_attention = DeformableAttention(
            self.num_heads,
            self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dropout=self.dropout, 
            offset_scale=self.offset_scale,
            num_fmap_levels=1,
            concat_vq_for_offset=False,
        )
        self.addnorm2 = AddNorm(input_dim=self.embed_dim)

        self.mlp      = SimpleMLP(self.embed_dim, self.embed_dim, self.dim_feedforward)

    def forward(
            self, 
            queries: torch.Tensor,
            bev_features: torch.Tensor,
            ref_points: torch.Tensor, 
            padding_mask: Optional[torch.Tensor]=None,
            attn_mask: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Input
        --------------------------------

        :queries: (N, num_elements, num_tokens, embed_dim)

        :bev_features: (N, W_bev * H_bev, (C_bev or embed_dim))

        :ref_points: (N, num_elements * num_tokens, 1, 2), reference points for the deformable attention

        :padding_mask: (N, num_elements, num_tokens, 1), padding / attention mask for queries (0 if to ignore else 1)

        :attn_mask: (N, num_elements, num_tokens, num_tokens), padding / attention mask for queries (0 if to ignore else 1)

        Returns
        --------------------------------
        :track_queries: (N, num_queries, embed_dim), output queries to be fed into the next layer
        """
        H_bev, W_bev = self.bev_feature_hw
        assert bev_features.shape[1] == H_bev * W_bev

        ctx_queries = self.self_attention(queries, queries, queries, padding_mask, attn_mask)
        
        query_mask = None
        if padding_mask is not None:
            query_mask   = ~padding_mask
            queries      = torch.masked_fill(queries, query_mask, value=0.0)
            # flatten to use as attention mask for the DeformableAttention
            deform_attn_mask = torch.flatten(padding_mask, start_dim=1, end_dim=2)

        ctx_queries = self.addnorm1(ctx_queries, queries)

        bev_spatial_shape = torch.tensor([[H_bev, W_bev]], device=bev_features.device, dtype=torch.int64)
        ctx_queries       = torch.flatten(ctx_queries, start_dim=1, end_dim=2)
        bev_ctx_queries   = self.deform_attention(
            ctx_queries, ref_points, bev_features, bev_spatial_shape, attention_mask=deform_attn_mask
        )
        ctx_queries       = self.addnorm2(ctx_queries, bev_ctx_queries)
        ctx_queries       = torch.unflatten(ctx_queries, dim=1, sizes=queries.shape[1:3])
        
        ctx_queries = self.mlp(ctx_queries)
        return ctx_queries


class VectorMapFormer(Box2ScaleStaticMethodMixin, BaseFormer):
    """
    This is an implementation of the transformer decoder in the VectorMapNet (https://arxiv.org/pdf/2206.08920), this
    architecture predicts polylines for segments rather than a rasterized map.

    Embeddings Used:
    --------------------------------
    box_kps_pos_emb: 
        used to initialize the kps element embeddings, it represents the position in an element 
        keypoint a given point belongs to.

     element_pos_emb: 
        used to initialize the kps element embeddings, it represents the position of each element.

     class_emb:
        used to create embeddings for each class index

     vertex_seq_emb:
        used to embed the location of a vertex point in the sequence of vertices

     xy_coord_emb:
        used to represent whether a given index in the flattened vertices is x or y

     grid_value_emb:
        used to embed the discrete grid of vertices along the BEV grid.
    """
    def __init__(
            self,
            num_classes: int,
            num_heads: int=4, 
            embed_dim: int=128,
            num_layers: int=4,
            num_ref_points: int=4,
            dim_feedforward: int=512, 
            dropout: float=0.1,
            offset_scale: float=1.0,
            max_elements: int=500,
            max_vertices: int=500,
            learnable_pe: bool=True,
            bev_feature_hw: Tuple[int, int]=(200, 200),
        ):
        super(VectorMapFormer, self).__init__()

        self.num_classes     = num_classes
        self.num_heads       = num_heads
        self.embed_dim       = embed_dim
        self.num_layers      = num_layers
        self.num_ref_points  = num_ref_points
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.offset_scale    = offset_scale
        self.max_elements    = max_elements
        self.learnable_pe    = learnable_pe
        self.bev_feature_hw  = bev_feature_hw

        # valid vertex coordinate indexes will range from 0 to max(bev_W, bev_H) - 1, so it is only natural
        # that the eos and pad tokens be an indexes outside this range. To avoid using up more memory than
        # necessary for sequence embedding, we use max((bev_W, bev_H)) + 2 as embedding size, so eos and pad
        # tokens are max(bev_W, bev_H) and max(bev_W, bev_H) + 1 respectively.
        grid_size            = max(self.bev_feature_hw)
        self.eos_token       = grid_size
        self.pad_token       = self.eos_token + 1
        self.max_vertices    = max_vertices
        self.num_bbox_kps    = 2
        self.num_coord       = 2

        self.box_kps_pos_emb   = PosEmbedding1D(self.num_bbox_kps, self.embed_dim, self.learnable_pe)
        self.element_pos_emb   = PosEmbedding1D( self.max_elements, self.embed_dim, self.learnable_pe)
        self.class_emb         = PosEmbedding1D(self.num_classes, self.embed_dim, self.learnable_pe)
        self.vertex_seq_emb    = PosEmbedding1D(self.max_vertices, self.embed_dim, self.learnable_pe)
        self.xy_coord_emb      = PosEmbedding1D(self.num_coord, self.embed_dim, self.learnable_pe)
        self.grid_value_emb    = PosEmbedding1D(grid_size + 2, self.embed_dim, self.learnable_pe, self.pad_token)
        self.kps_decoder       = VectorMapKeypointDecoderLayer(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim,
            num_ref_points=self.num_ref_points,
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            bev_feature_hw=self.bev_feature_hw,
        )
        self.bbox_kp_head       = SimpleMLP(self.embed_dim, 2, self.dim_feedforward, final_activation=nn.Sigmoid())
        self.bbox_class_head    = SimpleMLP(self.embed_dim*self.num_bbox_kps, self.num_classes, self.dim_feedforward)
        self.polyline_generator = self._create_polyline_pred_layers()

    def _get_ref_points_from_keypoints(self, kps: torch.Tensor) -> torch.Tensor:
        wh   = (kps[..., 1, :] - kps[..., 0, :]).abs()
        c_xy = kps[..., 0, :] + (wh / 2)
        c_xy = (2 * c_xy) - 1
        return c_xy[..., None, :]

    def _quantize_kps(self, kps: torch.Tensor, bev_wh: torch.Tensor) -> torch.Tensor:
        return (kps.detach() * (bev_wh - 1)).long()

    def _dequantize_kps(self, kps: torch.Tensor, bev_wh: torch.Tensor) -> torch.Tensor:
        return kps.detach() / (bev_wh - 1)
    
    def _make_casual_mask(self, *shape: int, device: Union[int, str, torch.device]):
        mask = torch.ones(*shape, dtype=torch.bool, device=device)
        mask = torch.tril(mask)
        return mask
    
    def _create_polyline_pred_layers(self) -> nn.ModuleList:
        layers = [PolyLineGeneratorLayer(
            num_heads=self.num_heads, 
            embed_dim=self.embed_dim, 
            num_ref_points=self.num_ref_points, 
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            offset_scale=self.offset_scale,
            bev_feature_hw=self.bev_feature_hw,
            mask_out_ctx=(i == (self.num_layers - 1))
        ) for i in range(0, self.num_layers)]
        
        layers.append(SimpleMLP(
            self.embed_dim, 
            max(self.bev_feature_hw) + 1, 
            self.dim_feedforward, 
            final_activation=nn.ReLU())
        )
        return nn.ModuleList(layers)
    
    def _generate_polylines(
        self, 
        input_queries: torch.Tensor,
        bev_features: torch.Tensor,
        ref_points: torch.Tensor, 
        padding_mask: Optional[torch.Tensor]=None,
        attn_mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        queries = input_queries
        for didx in range(0, len(self.polyline_generator)):
            if didx != len(self.polyline_generator) - 1:
                queries = self.polyline_generator[didx](
                    queries, bev_features, ref_points, padding_mask, attn_mask
                )
                continue
            # in inference one token is predicted at a time so we only take the last token prediction.
            # During training and eval, all tokens are predicted together, so we select all tokens, with
            # the exception of the starter tokens (i.e, the keypoints embedding token and the classification token)
            cutoff = -1 if self.inference_mode else (self.num_bbox_kps * 2) + 1
            queries = queries[..., cutoff:, :]

            vertex_logits = self.polyline_generator[didx](queries)
        return queries, vertex_logits

    def forward(
            self, 
            bev_features: torch.Tensor,
            tgt_kps: Optional[torch.Tensor]=None,
            tgt_classes: Optional[torch.Tensor]=None,
            tgt_vertices: Optional[torch.Tensor]=None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :tgt_kps: (N, max_elements, k, 2) Target keypoints for the map element box
            Expected to be quantized to range from x \in [0, W] and y \in [0, H]

        :tgt_classes: (N, max_elements), Target class labels for each map element

        :tgt_vertices: (N, max_elements, max_vertices, 2), the target vertices for each element along the grid, the targets
            are organized like so [[x_0, y_0], [x_1, y_1], [x_2, y2], ..., [EOS, PAD], [PAD, PAD], [PAD, PAD]]. Basically
            every sequnece ends with an EOS token and a PAD token as the x, y value, then subsequent values are [PAD, PAD]
            if not up to maximum sequence length.

        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder

        Returns
        --------------------------------
        :map_queries: (N, num_elements, embed_dim), computed context queries pooled along the vertices dimension.

        :polylines | polyline_logits: (N, num_elements, num_vertices, 2) | (N, num_elements, num_vertices, 2, num_grid),
            Contains either class labels of class logits depending on whether inference mode has been set or not 
            
        :boxes: (N, num_elements, (num_kp * 2) + num_classes), this is a combination of predicted box keypoints 
            and class logits
        """
        if tgt_kps is not None:
            assert tgt_kps.dtype == torch.int64

        if tgt_classes is not None:
            assert tgt_classes.dtype == torch.int64

        if tgt_vertices is not None:
            assert tgt_vertices.dtype == torch.int64
            
        batch_size = bev_features.shape[0]
        device     = bev_features.device

        element_emb = self.element_pos_emb()[:, None, :, :] + self.box_kps_pos_emb()[:, :, None, :]
        element_emb = torch.flatten(element_emb, start_dim=1, end_dim=2).tile(batch_size, 1, 1)

        ref_points  = DeformableAttention.generate_standard_ref_points(
            self.bev_feature_hw, batch_size=batch_size, device=device, n_sample=element_emb.shape[1]
        )
        ref_points  = ref_points[:, :, None, :]
        kps_queries = self.kps_decoder(element_emb, bev_features, ref_points)
        kps_queries = torch.unflatten(kps_queries, dim=1, sizes=(self.max_elements, self.num_bbox_kps))

        # these element keypoints are the x, y values of the top left and bottom right corner points of the box that covers 
        # the polyline to be generated. We can use the center of these boxes as reference points for the deformable attention.
        # element_kps shape:  (N, max_elements, k, 2), where k = 2 (two corner vertices)
        # class_logits shape: (N, max_elements, num_classes)
        bbox_kps     = self.bbox_kp_head(kps_queries)
        class_logits = self.bbox_class_head(torch.flatten(kps_queries, start_dim=2, end_dim=3))

        # class_emb        : (N, max_elements, 1, d)
        # kps_vertex_emb   : (N, max_elements, k, 2, d)
        # ref_points       : (N, max_elements, 1, 2)
        # coord_emb        : (1, 1, 1, 2, d)
        # box_kps_emb      : (1, 1, 2, 1, d)
        # seq_emb          : (N, 1, v, 1, d)
        # pline_vertex_emb : (N, max_elements, v, 2, d)

        bev_wh = torch.tensor([self.bev_feature_hw[1], self.bev_feature_hw[0]], dtype=bev_features.dtype, device=device)

        if tgt_kps is not None:
            ref_points     = self._get_ref_points_from_keypoints(self._dequantize_kps(tgt_kps))
            kps_vertex_emb = self.grid_value_emb(tgt_kps)
        else:
            ref_points     = self._get_ref_points_from_keypoints(bbox_kps)
            kps_vertex_emb = self.grid_value_emb(self._quantize_kps(bbox_kps, bev_wh))

        if tgt_classes is not None:
            class_embs = self.class_emb(tgt_classes)
        else:
            classes    = torch.argmax(class_logits, dim=-1)
            class_embs = self.class_emb(classes)

        class_embs = class_embs[:, :, None, :]         
            
        coord_idx = torch.arange(self.num_coord, device=bev_features.device, dtype=torch.int64)
        coord_idx = coord_idx[None, None, None, :]
        coord_emb = self.xy_coord_emb(coord_idx)

        box_kps_idx = torch.arange(self.num_bbox_kps, device=bev_features.device, dtype=torch.int64)
        box_kps_idx = box_kps_idx[None, None, :, None]
        box_kps_emb = self.box_kps_pos_emb(box_kps_idx)

        kps_emb        = torch.flatten(kps_vertex_emb + coord_emb + box_kps_emb, start_dim=2, end_dim=3)
        global_context = torch.concat([class_embs, kps_emb], dim=2)
        input_queries  = global_context

        bbox_kps = torch.flatten(bbox_kps, start_dim=2, end_dim=3)
        bbox_xywh = xyxy_to_xywh(bbox_kps)
        boxes = torch.concat([bbox_xywh, class_logits], dim=-1)
        if not self.inference_mode:
            assert tgt_vertices is not None
            map_queries, polyline_logits = self._train_forward(
                input_queries, tgt_vertices, bev_features, coord_emb, ref_points
            )
            return map_queries, polyline_logits, boxes
        else:
            map_queries, polylines = self._inference_forward(input_queries, bev_features, coord_emb, ref_points)
            return map_queries, polylines, boxes
        
    def _train_forward(
            self, 
            input_queries: torch.Tensor,
            tgt_vertices: torch.Tensor, 
            bev_features: torch.Tensor, 
            coord_emb: torch.Tensor, 
            ref_points: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            
        seq_idx = torch.arange(tgt_vertices.shape[2], device=bev_features.device, dtype=torch.int64)
        seq_idx = seq_idx[None, None, :, None]
        seq_emb = self.vertex_seq_emb(seq_idx)

        pline_vertex_emb = self.grid_value_emb(tgt_vertices)
        vertex_context   = torch.flatten(coord_emb + seq_emb + pline_vertex_emb, start_dim=2, end_dim=3)

        input_queries    = torch.concat([input_queries, vertex_context], dim=2)
        padding_mask     = torch.flatten((tgt_vertices != self.pad_token)[..., None], start_dim=2, end_dim=3)
        ones             = torch.ones(
            *tgt_vertices.shape[:2], 1 + (2 * self.num_bbox_kps), 1, dtype=torch.bool, device=bev_features.device
        )
        padding_mask     = torch.concat([ones, padding_mask], dim=2)
        attn_mask        = self._make_casual_mask(
            *input_queries.shape[:2], input_queries.shape[2], input_queries.shape[2], device=bev_features.device
        )
        attn_mask[..., :, :(1 + (2 * self.num_bbox_kps))] = 1
        # ref_points needs to be tiled this way because the deformable attention in the decoder requires
        # that the 4D query input be 3D, hence the the query is flattened along its second and third
        # dimensions to match
        ref_points = ref_points.tile(1, (tgt_vertices.shape[2] * 2) + (1 + (self.num_bbox_kps * 2)), 1, 1)
        map_queries, polyline_logits = self._generate_polylines(
            input_queries, bev_features, ref_points, padding_mask=padding_mask, attn_mask=attn_mask
        )
        map_queries     = torch.max(map_queries, dim=2).values
        polyline_logits = torch.unflatten(polyline_logits, dim=2, sizes=(-1, 2))
        return map_queries, polyline_logits
    
    def _inference_forward(
            self,
            input_queries: torch.Tensor,
            bev_features: torch.Tensor, 
            coord_emb: torch.Tensor, 
            ref_points: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        map_queries = []
        polylines = []
        num_iters = (2 * self.max_vertices) + 1

        is_finished  = torch.zeros(*input_queries.shape[:2], 1, dtype=torch.bool, device=bev_features.device)
        padding_mask = torch.ones(
            *input_queries.shape[:2], 1 + (2 * self.num_bbox_kps), 1, dtype=torch.bool, device=bev_features.device
        )

        for idx in range(0, num_iters):
            _ref_points = ref_points.tile(1, input_queries.shape[2], 1, 1)
            vertex_queries, vertex_logits = self._generate_polylines(
                input_queries, bev_features, _ref_points, padding_mask=padding_mask
            )
            next_vertex = torch.argmax(vertex_logits, dim=-1)
            
            map_queries.append(vertex_queries)
            polylines.append(next_vertex)

            if idx == (num_iters - 1):
                break

            if self.eos_token is not None:
                is_finished |= (next_vertex == self.eos_token)
                if torch.all(is_finished):
                    break

            seq_idx = torch.tensor([idx // self.num_coord], device=bev_features.device, dtype=torch.int64)
            seq_idx = seq_idx[None, None, :]
            seq_emb = self.vertex_seq_emb(seq_idx)

            coord_idx = torch.tensor([idx % self.num_coord], device=bev_features.device, dtype=torch.int64)
            coord_idx = coord_idx[None, None, :]
            coord_emb = self.xy_coord_emb(coord_idx)

            pline_vertex_emb = self.grid_value_emb(next_vertex)
            vertex_context   = coord_emb + seq_emb + pline_vertex_emb
            input_queries = torch.concat([input_queries, vertex_context], dim=2)
            padding_mask  = torch.concat([padding_mask, (next_vertex != self.pad_token)[..., None]], dim=2)

        map_queries = torch.max(torch.concat(map_queries[:-1], dim=2), dim=2).values
        polylines = torch.concat(polylines[:-1], axis=2)
        polylines = torch.unflatten(polylines, dim=2, sizes=(-1, 2))
        return map_queries, polylines


class RasterMapFormer(Box2ScaleStaticMethodMixin, TrackFormer):
    """
    This is not a standard panoptic segformer like the one in this Segformer paper (https://arxiv.org/pdf/2109.03814),
    this is a custom designed architecture that combines deformable DETR decoder (https://arxiv.org/pdf/2010.04159)
    with YOLACT (https://arxiv.org/pdf/1904.02689) for instance segmentation.
    """

    def __init__(self, *args, num_seg_coeffs: int=32, seg_c_h: int=256, **kwargs):
        super(RasterMapFormer, self).__init__(*args, **kwargs)

        assert num_seg_coeffs > 0
        
        self.num_seg_coeffs = num_seg_coeffs
        self.seg_c_h        = seg_c_h

        self.proto_seg_module = ProtoSegModule(
            in_channels=self.embed_dim, 
            out_channels=num_seg_coeffs, 
            c_h=seg_c_h, 
        )

        self.detection_module = RasterMapDetectionHead(
            embed_dim=self.embed_dim, 
            num_classes=self.num_classes, 
            num_coefs=self.num_seg_coeffs
        )
        del self.ego_query_emb

    def forward(self, bev_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input
        --------------------------------
        :bev_features: (N, H_bev*W_bev, (C_bev or embed_dim)), BEV features from the BevFormer encoder             

        Returns
        --------------------------------
        :queries: (N, num_queries, embed_dim) context queries for each map element from the last decoder layer

        :protos: (N, k, H_bev, W_bev), where k is the number of segmentation coefficients. 
            NOTE: This tensor is usually matrix multiplied with the k coefficient queries of n objects:
            (n, k) x (k, H_bev, W_bev) to produce an output of shape (n, H_bev, W_bev). The reason this
            multiplication is not done in this forward method is because a lot of predictions will be 
            dropped, both during training and inference, so only the predictions pertaining to valid 
            detections are used, hence no need for the wasted computation (probably sounds rich coming from me).

        :detections: (num_layers, N, num_queries, 4 + k + num_classes) | (N, num_queries, 4 + k + num_classes), tensor 
            contains confidence scores and classes logits for each element
        """

        assert bev_features.shape[-1] == self.embed_dim

        batch_size = bev_features.shape[0]
        queries    = self.detection_pos_emb().tile(batch_size, 1, 1)
        
        ref_points = DeformableAttention.generate_standard_ref_points(
            self.bev_feature_hw,
            batch_size=batch_size, 
            device=bev_features.device, 
            normalize=True,
            n_sample=queries.shape[1]
        )
        ref_points = ref_points[:, :, None, :]

        protos = self.proto_seg_module(
            bev_features.permute(0, 2, 1).reshape(batch_size, self.embed_dim, *self.bev_feature_hw)
        )
        
        layers_detections = []
        
        for decoder_idx in range(0, len(self.decoder_modules)):
            queries = self.decoder_modules[decoder_idx](
                queries=queries,
                bev_features=bev_features,
                ref_points=ref_points,
            )
            
            if self.inference_mode:
                if decoder_idx == self.num_layers - 1:
                    return queries, protos, self.detection_module(queries)
            else:
                layers_detections.append(self.detection_module(queries))
            
        layers_detections = torch.stack(layers_detections, dim=0)
        return queries, protos, layers_detections
