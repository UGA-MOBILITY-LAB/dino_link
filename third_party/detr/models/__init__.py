# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .dinolink_token_detr import build_dinolink_token_model


def build_model(args):
    if getattr(args, "use_dinolink_tokens", False):
        return build_dinolink_token_model(args)
    return build(args)
