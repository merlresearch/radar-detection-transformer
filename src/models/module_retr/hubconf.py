# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 Microsoft.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from .backbone import Backbone, Joiner
from .detr import DETR, PostProcess
from .position_encoding import DepthPrioritizedPositionEmbeddingSine
from .transformer import (
    ConditionalTransformerDecoder,
    ConditionalTransformerDecoderLayer,
    ConditionalTransformerEncoder,
    ConditionalTransformerEncoderLayer,
)

dependencies = ["torch", "torchvision"]


def _make_detr(
    backbone_name: str,
    dilation=False,
    pretrained=False,
    num_classes=2,
    num_queries=2,
    return_intermediate_dec=False,
    ratio=0.5,
    topk=256,
    rw=128,
    rh=256,
    iw=240,
    ih=320,
    **kwargs
):
    hidden_dim = 256
    backbone = Backbone(
        backbone_name,
        train_backbone=True,
        return_interm_layers=True,
        dilation=dilation,
        pretrained=pretrained,
    )

    pos_enc = DepthPrioritizedPositionEmbeddingSine(hidden_dim, normalize=True, ratio=ratio)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels

    encoder_layer = ConditionalTransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    )
    encoder = ConditionalTransformerEncoder(encoder_layer, 6, None)

    decoder_layer = ConditionalTransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    )
    decoder_norm = nn.LayerNorm(hidden_dim)
    decoder = ConditionalTransformerDecoder(decoder_layer, 6, decoder_norm, return_intermediate=return_intermediate_dec)

    detr = DETR(
        backbone_with_pos_enc,
        encoder,
        decoder,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=True,
        topk=topk,
        rw=rw,
        rh=rh,
        iw=iw,
        ih=ih,
    )
    return detr


def detr_resnet18(
    pretrained=False,
    num_classes=91,
    topk=256,
    ratio=0.5,
    return_intermediate_dec=False,
    return_postprocessor=False,
    rw=128,
    rh=256,
    iw=240,
    ih=320,
    **kwargs
):
    model = _make_detr(
        "resnet18",
        dilation=False,
        pretrained=pretrained,
        num_classes=num_classes,
        topk=topk,
        ratio=ratio,
        return_intermediate_dec=return_intermediate_dec,
        rw=rw,
        rh=rh,
        iw=iw,
        ih=ih,
        **kwargs
    )
    pretrained_detr = False
    if pretrained_detr:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model
