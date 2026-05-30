"""
PyTorch checkpoint demo for DEIMv2 wholebody49 instance segmentation.
"""

import argparse
import copy
import heapq
import importlib
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm


AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BODY_CLASS_ID = 0
BONE_CLASS_ID = 48
DEFAULT_CONFIG = 'configs/deimv2/deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center.yml'

BOX_COLORS = [
    ((216, 67, 21), 'Front'),
    ((255, 87, 34), 'Right-Front'),
    ((123, 31, 162), 'Right-Side'),
    ((255, 193, 7), 'Right-Back'),
    ((76, 175, 80), 'Back'),
    ((33, 150, 243), 'Left-Back'),
    ((156, 39, 176), 'Left-Side'),
    ((0, 188, 212), 'Left-Front'),
]

BONE_EDGE_PAIRS = (
    (21, 23),
    (21, 24),
    (21, 25),
    (25, 35),
    (23, 27),
    (27, 30),
    (24, 28),
    (28, 31),
    (35, 37),
    (37, 40),
    (40, 43),
    (39, 42),
    (35, 38),
    (38, 41),
    (41, 44),
)

EDGES = [
    (21, 22), (21, 22),
    (21, 25),
    (22, 26), (22, 26),
    (26, 29), (26, 29),
    (29, 32), (29, 32),
    (22, 36), (22, 36),
    (25, 35),
    (35, 36), (35, 36),
    (36, 39), (36, 39),
    (39, 42), (39, 42),
    (42, 45), (42, 45),
]

OBJECT_CLASS_IDS = {0, 5, 6, 7, 16, 17, 18, 19, 20, 32, 33, 34, 45, 46, 47, BONE_CLASS_ID}
ATTRIBUTE_CLASS_IDS = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15}
KEYPOINT_CLASS_IDS = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
}
KEYPOINT_CLASS_ID_ORDER = (21, 22, 25, 26, 29, 35, 36, 39, 42)
KEYPOINT_DRAW_CLASS_IDS = KEYPOINT_CLASS_IDS
KEYPOINT_NMS_CLASS_IDS = tuple(sorted(KEYPOINT_CLASS_IDS))
SKELETON_KEYPOINT_IDS = {21, 22, 25, 26, 29, 32, 35, 36, 39, 42, 45}
BONE_RENDER_KEYPOINT_IDS = {class_id for edge in BONE_EDGE_PAIRS for class_id in edge}
SKELETON_ASSIGNMENT_KEYPOINT_IDS = SKELETON_KEYPOINT_IDS | BONE_RENDER_KEYPOINT_IDS

LEFT_SIDE_CLASS_IDS = {23, 27, 30, 33, 37, 40, 43, 46}
RIGHT_SIDE_CLASS_IDS = {24, 28, 31, 34, 38, 41, 44, 47}
SIDE_ATTR_CLASS_IDS = LEFT_SIDE_CLASS_IDS | RIGHT_SIDE_CLASS_IDS
SIDE_PARENT_TO_CHILDREN = {
    22: (23, 24),
    26: (27, 28),
    29: (30, 31),
    32: (33, 34),
    36: (37, 38),
    39: (40, 41),
    42: (43, 44),
    45: (46, 47),
}
SIDE_AWARE_SKELETON_CLASS_IDS = set(SIDE_PARENT_TO_CHILDREN.keys())
SIDE_AWARE_OBJECT_CLASS_IDS = {32, 45}

LEFT_SIDE_COLOR = (0, 128, 0)
RIGHT_SIDE_COLOR = (255, 0, 255)
BONE_BBOX_COLOR = (255, 255, 0)
MASK_CLEANUP_PADDING = 1
MIXED_KEYPOINT_FOREIGN_SHARE_THRESHOLD = 0.10
MIXED_KEYPOINT_FOREIGN_PIXEL_THRESHOLD = 2
INSTANCE_EDGE_MIXED_ASSIGNED_SHARE_MIN = 0.75
INCLUDE_KEY = '__include__'
_CENTER_GRID_CACHE: dict[tuple[int, int, int, int, tuple[str, int], torch.dtype], torch.Tensor] = {}
_CENTER_INDEX_CACHE: dict[tuple[int, int, int, int, tuple[str, int]], tuple[torch.Tensor, torch.Tensor]] = {}
_ENGINE_CREATE: Optional[Callable[..., Any]] = None
_ENGINE_GLOBAL_CONFIG: Optional[Dict[str, Any]] = None


def _unique_undirected_edges(edges: Sequence[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]}


SKELETON_CONNECTION_DEGREE_LIMITS = Counter(
    class_id
    for edge in _unique_undirected_edges(tuple(EDGES) + tuple(BONE_EDGE_PAIRS))
    for class_id in edge
)
SKELETON_NATURAL_CONNECTION_KEYS = _unique_undirected_edges(tuple(EDGES) + tuple(BONE_EDGE_PAIRS))
SKELETON_NATURAL_NEIGHBOR_CLASS_IDS: Dict[int, set[int]] = {}
for _first_class_id, _second_class_id in SKELETON_NATURAL_CONNECTION_KEYS:
    SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.setdefault(_first_class_id, set()).add(_second_class_id)
    SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.setdefault(_second_class_id, set()).add(_first_class_id)


def merge_dict(dct: Dict[str, Any], another_dct: Dict[str, Any], inplace: bool = True) -> Dict[str, Any]:
    def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        for key in extra:
            if key in base and isinstance(base[key], dict) and isinstance(extra[key], dict):
                _merge(base[key], extra[key])
            else:
                base[key] = extra[key]
        return base

    target = dct if inplace else copy.deepcopy(dct)
    return _merge(target, another_dct)


def load_config(file_path: str | Path, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved_path = Path(file_path)
    if resolved_path.suffix.lower() not in {'.yml', '.yaml'}:
        raise ValueError(f'Only YAML configs are supported: {resolved_path}')

    config = {} if cfg is None else cfg
    with resolved_path.open('r', encoding='utf-8') as file:
        file_cfg = yaml.safe_load(file) or {}

    if INCLUDE_KEY in file_cfg:
        for base_yaml in list(file_cfg[INCLUDE_KEY]):
            base_path = Path(base_yaml).expanduser()
            if not base_path.is_absolute():
                base_path = resolved_path.parent / base_path
            base_cfg = load_config(base_path, config)
            merge_dict(config, base_cfg)

    return merge_dict(config, file_cfg)


def merge_config(
    cfg: Dict[str, Any],
    another_cfg: Optional[Dict[str, Any]] = None,
    inplace: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source = {} if another_cfg is None else another_cfg

    def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        for key in extra:
            if key not in base:
                base[key] = extra[key]
            elif isinstance(base[key], dict) and isinstance(extra[key], dict):
                _merge(base[key], extra[key])
            elif overwrite:
                base[key] = extra[key]
        return base

    target = cfg if inplace else copy.deepcopy(cfg)
    return _merge(target, source)


def load_engine_factory() -> tuple[Callable[..., Any], Dict[str, Any]]:
    global _ENGINE_CREATE, _ENGINE_GLOBAL_CONFIG
    if _ENGINE_CREATE is not None and _ENGINE_GLOBAL_CONFIG is not None:
        return _ENGINE_CREATE, _ENGINE_GLOBAL_CONFIG

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    importlib.import_module('engine')
    workspace = importlib.import_module('engine.core.workspace')
    _ENGINE_CREATE = workspace.create
    _ENGINE_GLOBAL_CONFIG = workspace.GLOBAL_CONFIG
    return _ENGINE_CREATE, _ENGINE_GLOBAL_CONFIG


class InferenceConfig:
    def __init__(self, cfg_path: str, **kwargs: Any) -> None:
        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)
        self.yaml_cfg = copy.deepcopy(cfg)
        self._model: Optional[nn.Module] = None
        self._postprocessor: Optional[nn.Module] = None

    @property
    def global_cfg(self) -> Dict[str, Any]:
        _, engine_global_config = load_engine_factory()
        return merge_config(self.yaml_cfg, another_cfg=engine_global_config, inplace=False, overwrite=False)

    @property
    def model(self) -> nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            create, _ = load_engine_factory()
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return self._model

    @property
    def postprocessor(self) -> nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            create, _ = load_engine_factory()
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return self._postprocessor


def compute_resized_mask_output_size(
    spatial_size: Sequence[int],
    size: int | Sequence[int],
    max_size: int | None = None,
) -> tuple[int, int]:
    if len(spatial_size) != 2:
        raise ValueError(f'Expected spatial_size=(height, width), got {spatial_size}')

    height, width = int(spatial_size[0]), int(spatial_size[1])
    if isinstance(size, int):
        target_size = int(size)
        if max_size is not None:
            min_original_size = float(min(width, height))
            max_original_size = float(max(width, height))
            if max_original_size / min_original_size * target_size > max_size:
                target_size = int(round(max_size * min_original_size / max_original_size))

        if (width <= height and width == target_size) or (height <= width and height == target_size):
            return height, width

        if width < height:
            out_width = target_size
            out_height = int(target_size * height / width)
        else:
            out_height = target_size
            out_width = int(target_size * width / height)
        return out_height, out_width

    if len(size) == 1:
        return compute_resized_mask_output_size(spatial_size, int(size[0]), max_size=max_size)

    if len(size) != 2:
        raise ValueError(f'Expected size to have length 1 or 2, got {size}')

    return int(size[0]), int(size[1])


def _cache_device_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else device.index


def _compute_center_source_coords(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_coords = (
        (torch.arange(out_height, device=device, dtype=dtype) + 0.5) - (out_height / 2.0)
    ) * (in_height / out_height) + (in_height / 2.0)
    x_coords = (
        (torch.arange(out_width, device=device, dtype=dtype) + 0.5) - (out_width / 2.0)
    ) * (in_width / out_width) + (in_width / 2.0)

    y_coords = y_coords.clamp(0.0, float(max(in_height - 1, 0)))
    x_coords = x_coords.clamp(0.0, float(max(in_width - 1, 0)))
    return y_coords, x_coords


def _build_center_resize_grid(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y_coords, x_coords = _compute_center_source_coords(
        in_height,
        in_width,
        out_height,
        out_width,
        device=device,
        dtype=dtype,
    )

    if in_height > 1:
        y_norm = (y_coords / (in_height - 1)) * 2.0 - 1.0
    else:
        y_norm = torch.zeros_like(y_coords)
    if in_width > 1:
        x_norm = (x_coords / (in_width - 1)) * 2.0 - 1.0
    else:
        x_norm = torch.zeros_like(x_coords)

    grid_y = y_norm[:, None].expand(out_height, out_width)
    grid_x = x_norm[None, :].expand(out_height, out_width)
    return torch.stack((grid_x, grid_y), dim=-1)


def _get_center_resize_grid(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (in_height, in_width, out_height, out_width, _cache_device_key(device), dtype)
    grid = _CENTER_GRID_CACHE.get(key)
    if grid is None:
        grid = _build_center_resize_grid(
            in_height,
            in_width,
            out_height,
            out_width,
            device=device,
            dtype=dtype,
        )
        _CENTER_GRID_CACHE[key] = grid
    return grid


def _get_center_resize_indices(
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (in_height, in_width, out_height, out_width, _cache_device_key(device))
    indices = _CENTER_INDEX_CACHE.get(key)
    if indices is None:
        y_coords, x_coords = _compute_center_source_coords(
            in_height,
            in_width,
            out_height,
            out_width,
            device=device,
            dtype=torch.float32,
        )
        indices = (y_coords.round().to(torch.int64), x_coords.round().to(torch.int64))
        _CENTER_INDEX_CACHE[key] = indices
    return indices


def resize_masks(
    masks: torch.Tensor,
    size: int | Sequence[int],
    *,
    max_size: int | None = None,
    mode: str = 'nearest',
    origin: str = 'center',
) -> torch.Tensor:
    if masks.ndim != 4:
        raise ValueError(f'Expected masks with shape [N, C, H, W], got {tuple(masks.shape)}')

    out_height, out_width = compute_resized_mask_output_size(masks.shape[-2:], size=size, max_size=max_size)
    in_height, in_width = masks.shape[-2:]
    if (in_height, in_width) == (out_height, out_width):
        return masks

    if origin == 'center' and mode in ('nearest', 'nearest-exact'):
        y_index, x_index = _get_center_resize_indices(
            in_height,
            in_width,
            out_height,
            out_width,
            device=masks.device,
        )
        return masks.index_select(-2, y_index).index_select(-1, x_index)

    original_dtype = masks.dtype
    needs_float = (not torch.is_floating_point(masks)) or mode == 'bilinear' or (
        masks.device.type == 'cpu' and masks.dtype in (torch.float16, torch.bfloat16)
    )
    work_masks = masks.float() if needs_float else masks

    if origin == 'topleft' or (origin == 'center' and mode == 'area'):
        align_corners = False if mode in ('bilinear', 'bicubic') else None
        resized = F.interpolate(
            work_masks,
            size=(out_height, out_width),
            mode=mode,
            align_corners=align_corners,
        )
    elif origin == 'center':
        grid = _get_center_resize_grid(
            in_height,
            in_width,
            out_height,
            out_width,
            device=work_masks.device,
            dtype=work_masks.dtype,
        )
        grid = grid.unsqueeze(0).expand(work_masks.shape[0], -1, -1, -1)
        resized = F.grid_sample(
            work_masks,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True,
        )
    else:
        raise ValueError(f'Unsupported mask resize origin: {origin}')

    if original_dtype == torch.bool:
        return resized > 0.5

    if torch.is_floating_point(torch.empty((), dtype=original_dtype)):
        return resized.to(dtype=original_dtype) if resized.dtype != original_dtype else resized

    if mode in ('nearest', 'nearest-exact'):
        return resized.round().to(dtype=original_dtype)

    return resized


@dataclass(frozen=False)
class Box:
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    source_idx: int = -1
    generation: int = -1
    gender: int = -1
    handedness: int = -1
    head_pose: int = -1
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1


@dataclass(frozen=True)
class KeypointInstanceQuality:
    is_mixed: bool
    assigned_pixel_share: float
    assigned_pixel_count: int
    foreign_pixel_count: int


class SkeletonLineRegistry:
    def __init__(self) -> None:
        self.line_keys: set[Tuple[int, int]] = set()
        self.endpoint_counts: Counter[int] = Counter()
        self.endpoint_neighbor_slot_counts: Counter[Tuple[int, int, int]] = Counter()

    @staticmethod
    def line_key(first_box: Box, second_box: Box) -> Tuple[int, int]:
        first_id = id(first_box)
        second_id = id(second_box)
        if first_id < second_id:
            return (first_id, second_id)
        return (second_id, first_id)

    @staticmethod
    def endpoint_limit(box: Box) -> int:
        return max(1, int(SKELETON_CONNECTION_DEGREE_LIMITS.get(box.classid, 1)))

    @staticmethod
    def endpoint_neighbor_slot_key(endpoint_box: Box, neighbor_box: Box) -> Optional[Tuple[int, int, int]]:
        if neighbor_box.classid not in SKELETON_NATURAL_NEIGHBOR_CLASS_IDS.get(endpoint_box.classid, ()):
            return None

        neighbor_side_slot = -1
        if endpoint_box.classid not in SIDE_AWARE_SKELETON_CLASS_IDS and neighbor_box.handedness >= 0:
            neighbor_side_slot = neighbor_box.handedness
        return (id(endpoint_box), neighbor_box.classid, neighbor_side_slot)

    @classmethod
    def endpoint_neighbor_slot_keys(cls, first_box: Box, second_box: Box) -> Tuple[Tuple[int, int, int], ...]:
        slot_keys = []
        first_slot_key = cls.endpoint_neighbor_slot_key(first_box, second_box)
        if first_slot_key is not None:
            slot_keys.append(first_slot_key)
        second_slot_key = cls.endpoint_neighbor_slot_key(second_box, first_box)
        if second_slot_key is not None:
            slot_keys.append(second_slot_key)
        return tuple(slot_keys)

    def has_line(self, first_box: Box, second_box: Box) -> bool:
        return self.line_key(first_box, second_box) in self.line_keys

    def has_neighbor_slot_capacity(self, first_box: Box, second_box: Box) -> bool:
        return all(
            self.endpoint_neighbor_slot_counts[slot_key] < 1
            for slot_key in self.endpoint_neighbor_slot_keys(first_box, second_box)
        )

    def connection_slot_priority(self, first_box: Box, second_box: Box) -> int:
        return 1 if self.has_neighbor_slot_capacity(first_box, second_box) else 0

    def can_add(self, first_box: Box, second_box: Box) -> bool:
        if self.has_line(first_box, second_box):
            return False
        return (
            self.endpoint_counts[id(first_box)] < self.endpoint_limit(first_box)
            and self.endpoint_counts[id(second_box)] < self.endpoint_limit(second_box)
            and self.has_neighbor_slot_capacity(first_box, second_box)
        )

    def add(self, first_box: Box, second_box: Box) -> bool:
        if not self.can_add(first_box, second_box):
            return False
        self.line_keys.add(self.line_key(first_box, second_box))
        self.endpoint_counts[id(first_box)] += 1
        self.endpoint_counts[id(second_box)] += 1
        for slot_key in self.endpoint_neighbor_slot_keys(first_box, second_box):
            self.endpoint_neighbor_slot_counts[slot_key] += 1
        return True


class SimpleSortTracker:
    """Single-person-biased tracker using IoU, gated reassociation, and smoothed boxes."""

    def __init__(
        self,
        iou_threshold: float = 0.20,
        max_age: int = 45,
        min_score: float = 0.45,
        center_gate: float = 0.25,
        area_ratio_threshold: float = 0.35,
        aspect_ratio_threshold: float = 0.45,
        smoothing_alpha: float = 0.80,
        confirmed_hit_streak: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_score = min_score
        self.center_gate = center_gate
        self.area_ratio_threshold = area_ratio_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.smoothing_alpha = smoothing_alpha
        self.confirmed_hit_streak = confirmed_hit_streak
        self.next_track_id = 1
        self.tracks: List[Dict[str, Any]] = []
        self.frame_index = 0

    @staticmethod
    def _iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    @staticmethod
    def _center_distance(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5
        return float(math.hypot(acx - bcx, acy - bcy))

    @staticmethod
    def _area_ratio(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        return float(min(area_a, area_b) / max(area_a, area_b))

    @staticmethod
    def _aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        return width / height

    def _adaptive_center_threshold(self, bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        diag_a = math.hypot(ax2 - ax1, ay2 - ay1)
        diag_b = math.hypot(bx2 - bx1, by2 - by1)
        return max(24.0, diag_a * self.center_gate, diag_b * self.center_gate)

    @staticmethod
    def _bbox_tuple_from_smoothed(smoothed_bbox: Sequence[float]) -> Tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in smoothed_bbox)

    def _blend_bbox(
        self,
        previous_bbox: Sequence[float],
        current_bbox: Tuple[int, int, int, int],
    ) -> List[float]:
        return [
            self.smoothing_alpha * float(previous_bbox[idx]) + (1.0 - self.smoothing_alpha) * float(current_bbox[idx])
            for idx in range(4)
        ]

    def _is_viable_reassociation(
        self,
        track_bbox: Tuple[int, int, int, int],
        det_bbox: Tuple[int, int, int, int],
    ) -> bool:
        if self._area_ratio(track_bbox, det_bbox) < self.area_ratio_threshold:
            return False
        track_ar = self._aspect_ratio(track_bbox)
        det_ar = self._aspect_ratio(det_bbox)
        ar_ratio = min(track_ar, det_ar) / max(track_ar, det_ar)
        if ar_ratio < self.aspect_ratio_threshold:
            return False
        center_distance = self._center_distance(track_bbox, det_bbox)
        if center_distance > self._adaptive_center_threshold(track_bbox, det_bbox):
            return False
        return True

    def _track_priority(self, track: Dict[str, Any]) -> Tuple[int, int, float, int]:
        confirmed = 1 if track.get('confirmed', False) else 0
        return (confirmed, track['hit_streak'], float(track['last_score']), -int(track['missed']))

    def update(self, boxes: List[Box]) -> None:
        self.frame_index += 1

        for box in boxes:
            box.track_id = -1

        detections = [box for box in boxes if box.score >= self.min_score]

        if not detections and not self.tracks:
            return

        iou_matrix = None
        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for track_idx, track in enumerate(self.tracks):
                track_bbox = self._bbox_tuple_from_smoothed(track['smoothed_bbox'])
                for det_idx, box in enumerate(detections):
                    det_bbox = (box.x1, box.y1, box.x2, box.y2)
                    iou_matrix[track_idx, det_idx] = self._iou(track_bbox, det_bbox)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if iou_matrix is not None and iou_matrix.size > 0:
            while True:
                best_track = -1
                best_det = -1
                best_iou = self.iou_threshold
                for track_idx in sorted(
                    range(len(self.tracks)),
                    key=lambda idx: self._track_priority(self.tracks[idx]),
                    reverse=True,
                ):
                    if track_idx in matched_tracks:
                        continue
                    for det_idx in range(len(detections)):
                        if det_idx in matched_detections:
                            continue
                        iou = float(iou_matrix[track_idx, det_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track_idx
                            best_det = det_idx
                if best_track == -1:
                    break
                matched_tracks.add(best_track)
                matched_detections.add(best_det)
                matches.append((best_track, best_det))

        # Recover from small frame-to-frame box jitter when IoU alone is too strict.
        fallback_candidates: List[Tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            if track_idx in matched_tracks:
                continue
            track_bbox = self._bbox_tuple_from_smoothed(track['smoothed_bbox'])
            for det_idx, box in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                det_bbox = (box.x1, box.y1, box.x2, box.y2)
                if self._is_viable_reassociation(track_bbox, det_bbox):
                    center_distance = self._center_distance(track_bbox, det_bbox)
                    fallback_candidates.append((center_distance, track_idx, det_idx))

        fallback_candidates.sort(key=lambda item: item[0])
        for _, track_idx, det_idx in fallback_candidates:
            if track_idx in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            matches.append((track_idx, det_idx))

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det_box = detections[det_idx]
            det_bbox = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            previous_smoothed = track['smoothed_bbox']
            track['bbox'] = det_bbox
            track['smoothed_bbox'] = self._blend_bbox(previous_smoothed, det_bbox)
            track['missed'] = 0
            track['age'] += 1
            track['hit_streak'] += 1
            track['confirmed'] = track.get('confirmed', False) or track['hit_streak'] >= self.confirmed_hit_streak
            track['last_score'] = float(det_box.score)
            track['last_seen'] = self.frame_index
            det_box.track_id = track['id']

        surviving_tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                surviving_tracks.append(track)
                continue
            track['missed'] += 1
            track['age'] += 1
            was_confirmed = track.get('confirmed', False) or track['hit_streak'] >= self.confirmed_hit_streak
            max_allowed_age = self.max_age + 15 if was_confirmed else self.max_age
            track['hit_streak'] = 0
            track['confirmed'] = was_confirmed
            if track['missed'] <= max_allowed_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        has_recent_confirmed_track = any(
            track['missed'] <= 5 and track.get('confirmed', False)
            for track in self.tracks
        )

        for det_idx, det_box in enumerate(detections):
            if det_idx in matched_detections:
                continue
            if has_recent_confirmed_track:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            det_box.track_id = track_id
            det_bbox = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            self.tracks.append(
                {
                    'id': track_id,
                    'bbox': det_bbox,
                    'smoothed_bbox': [float(v) for v in det_bbox],
                    'missed': 0,
                    'age': 1,
                    'hit_streak': 1,
                    'confirmed': False,
                    'last_score': float(det_box.score),
                    'last_seen': self.frame_index,
                }
            )

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()
        self.frame_index = 0


def make_instance_color(instance_idx: int) -> Tuple[int, int, int]:
    palette = [
        '#ff6b6b', '#4ecdc4', '#ffe66d', '#1a535c', '#ff9f1c',
        '#5f0f40', '#9a031e', '#fb8b24', '#0f4c5c', '#2ec4b6',
        '#3a86ff', '#8338ec', '#ff006e', '#8ac926', '#1982c4',
        '#6a4c93', '#e76f51', '#2a9d8f', '#e9c46a', '#264653',
    ]
    hex_color = palette[instance_idx % len(palette)].lstrip('#')
    rgb = tuple(int(hex_color[pos:pos + 2], 16) for pos in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def list_image_paths(images_dir: Path) -> List[Path]:
    image_paths = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name)


def load_checkpoint_state(resume_path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict) and 'module' in checkpoint['ema']:
        return checkpoint['ema']['module']
    if 'model' in checkpoint:
        return checkpoint['model']
    raise KeyError(f'Checkpoint {resume_path} does not contain `ema.module` or `model`.')


def tensor_state_only(state: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def matched_tensor_state(current_state: Dict[str, object], loaded_state: Dict[str, object]):
    current_tensors = tensor_state_only(current_state)
    loaded_tensors = tensor_state_only(loaded_state)

    matched_state: Dict[str, torch.Tensor] = {}
    missing_keys: List[str] = []
    mismatched_keys: List[str] = []

    for key, value in current_tensors.items():
        if key not in loaded_tensors:
            missing_keys.append(key)
            continue
        if value.shape != loaded_tensors[key].shape:
            mismatched_keys.append(key)
            continue
        matched_state[key] = loaded_tensors[key]

    unexpected_keys = sorted(set(loaded_tensors.keys()) - set(current_tensors.keys()))
    return matched_state, missing_keys, mismatched_keys, unexpected_keys


def move_to_device(data, device: torch.device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_onnx_model(model_path: Path) -> bool:
    return model_path.suffix.lower() == '.onnx'


def is_torch_checkpoint_model(model_path: Path) -> bool:
    return model_path.suffix.lower() in {'.pt', '.pth'}


def build_onnx_providers(device_arg: str | None, model_path: Path, inference_type: str):
    import onnxruntime as ort

    available_providers = set(ort.get_available_providers())
    requested = (device_arg or '').lower()
    inference_type = inference_type.lower()

    if requested.startswith('cuda'):
        if 'CUDAExecutionProvider' not in available_providers:
            raise RuntimeError('CUDAExecutionProvider is not available in this onnxruntime build.')
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if requested == 'tensorrt':
        if 'TensorrtExecutionProvider' not in available_providers:
            raise RuntimeError('TensorrtExecutionProvider is not available in this onnxruntime build.')
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = {
                'trt_fp16_enable': True,
            }
        elif inference_type == 'int8':
            ep_type_params = {
                'trt_fp16_enable': True,
                'trt_int8_enable': True,
                'trt_int8_calibration_table_name': 'calibration.flatbuffers',
            }
        else:
            raise ValueError(f'Unsupported inference type for TensorRT: {inference_type}')
        providers = [
            (
                'TensorrtExecutionProvider',
                {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(model_path.parent),
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                } | ep_type_params,
            )
        ]
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    if requested and requested != 'cpu':
        raise ValueError(f'Unsupported ONNX device: {device_arg}. Use cpu, cuda, cuda:0, or tensorrt.')

    if device_arg is None and 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def build_transform(image_size: Sequence[int], normalize: bool) -> Callable[[np.ndarray], torch.Tensor]:
    target_h, target_w = int(image_size[0]), int(image_size[1])
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def transform(image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).to(dtype=torch.float32) / 255.0
        if normalize:
            tensor = (tensor - mean) / std
        return tensor

    return transform


def build_onnx_transform(image_size: Sequence[int] | None) -> Callable[[np.ndarray], torch.Tensor]:
    def transform(image_bgr: np.ndarray) -> torch.Tensor:
        resized = image_bgr
        if image_size is not None:
            target_h = int(image_size[0])
            target_w = int(image_size[1])
            resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(chw)

    return transform


def build_onnx_transform_with_normalize(
    image_size: Sequence[int] | None,
    normalize: bool,
) -> Callable[[np.ndarray], torch.Tensor]:
    target_h = None if image_size is None else int(image_size[0])
    target_w = None if image_size is None else int(image_size[1])
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def transform(image_bgr: np.ndarray) -> torch.Tensor:
        resized = image_bgr
        if target_h is not None and target_w is not None:
            resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)) / 255.0
        if normalize:
            tensor = (tensor - mean) / std
        return tensor

    return transform


def infer_onnx_normalize_from_model_path(model_path: Path) -> bool:
    tokens = [token for token in re.split(r'[^a-z0-9]+', model_path.stem.lower()) if token]
    no_norm_tokens = {'atto', 'femto', 'pico', 'n'}
    if any(token in no_norm_tokens for token in tokens):
        return False
    return True


def binary_mask_bbox(mask: np.ndarray) -> List[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def calculate_iou(base_obj: Box, target_obj: Box) -> float:
    inter_xmin = max(base_obj.x1, target_obj.x1)
    inter_ymin = max(base_obj.y1, target_obj.y1)
    inter_xmax = min(base_obj.x2, target_obj.x2)
    inter_ymax = min(base_obj.y2, target_obj.y2)
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
    area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
    return inter_area / float(area1 + area2 - inter_area)


def find_most_relevant_obj(base_objs: List[Box], target_objs: List[Box]) -> None:
    for base_obj in base_objs:
        most_relevant_obj: Box | None = None
        best_score = 0.0
        best_iou = 0.0
        best_distance = float('inf')

        for target_obj in target_objs:
            distance = math.hypot(base_obj.cx - target_obj.cx, base_obj.cy - target_obj.cy)
            if not target_obj.is_used and distance <= 10.0:
                if target_obj.score >= best_score:
                    iou = calculate_iou(base_obj, target_obj)
                    if iou > best_iou:
                        most_relevant_obj = target_obj
                        best_iou = iou
                        best_distance = distance
                        best_score = target_obj.score
                    elif iou > 0.0 and iou == best_iou and distance < best_distance:
                        most_relevant_obj = target_obj
                        best_distance = distance
                        best_score = target_obj.score

        if most_relevant_obj is None:
            continue

        if most_relevant_obj.classid == 1:
            base_obj.generation = 0
        elif most_relevant_obj.classid == 2:
            base_obj.generation = 1
        elif most_relevant_obj.classid == 3:
            base_obj.gender = 0
        elif most_relevant_obj.classid == 4:
            base_obj.gender = 1
        elif most_relevant_obj.classid == 8:
            base_obj.head_pose = 0
        elif most_relevant_obj.classid == 9:
            base_obj.head_pose = 1
        elif most_relevant_obj.classid == 10:
            base_obj.head_pose = 2
        elif most_relevant_obj.classid == 11:
            base_obj.head_pose = 3
        elif most_relevant_obj.classid == 12:
            base_obj.head_pose = 4
        elif most_relevant_obj.classid == 13:
            base_obj.head_pose = 5
        elif most_relevant_obj.classid == 14:
            base_obj.head_pose = 6
        elif most_relevant_obj.classid == 15:
            base_obj.head_pose = 7
        elif most_relevant_obj.classid in LEFT_SIDE_CLASS_IDS:
            base_obj.handedness = 0
        elif most_relevant_obj.classid in RIGHT_SIDE_CLASS_IDS:
            base_obj.handedness = 1

        most_relevant_obj.is_used = True


def nms(target_objs: List[Box], iou_threshold: float) -> List[Box]:
    filtered_objs: List[Box] = []
    sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

    while sorted_objs:
        current_box = sorted_objs.pop(0)
        if current_box.is_used:
            continue

        filtered_objs.append(current_box)
        current_box.is_used = True

        remaining_boxes = []
        for box in sorted_objs:
            if not box.is_used:
                iou_value = calculate_iou(current_box, box)
                if iou_value >= iou_threshold:
                    box.is_used = True
                else:
                    remaining_boxes.append(box)
        sorted_objs = remaining_boxes

    return filtered_objs


def build_result_boxes(
    result: Dict[str, torch.Tensor],
    image_width: int,
    image_height: int,
    object_score_threshold: float,
    attribute_score_threshold: float,
    keypoint_threshold: float,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
    enable_bone_drawing_mode: bool,
) -> List[Box]:
    labels = result['labels'].detach().cpu()
    scores = result['scores'].detach().cpu()
    boxes = result['boxes'].detach().cpu()

    result_boxes: List[Box] = []
    box_score_threshold = min(object_score_threshold, attribute_score_threshold, keypoint_threshold)

    for idx in range(len(labels)):
        score = float(scores[idx].item())
        if score <= box_score_threshold:
            continue

        classid = int(labels[idx].item())
        x1_f, y1_f, x2_f, y2_f = [float(v) for v in boxes[idx].tolist()]
        x1 = max(0, min(int(round(x1_f)), image_width - 1))
        y1 = max(0, min(int(round(y1_f)), image_height - 1))
        x2 = max(0, min(int(round(x2_f)), image_width - 1))
        y2 = max(0, min(int(round(y2_f)), image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        result_boxes.append(
            Box(
                classid=classid,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) // 2,
                cy=(y1 + y2) // 2,
                source_idx=idx,
            )
        )

    result_boxes = [
        box for box in result_boxes
        if (box.classid in OBJECT_CLASS_IDS and box.score >= object_score_threshold) or box.classid not in OBJECT_CLASS_IDS
    ]
    result_boxes = [
        box for box in result_boxes
        if (box.classid in ATTRIBUTE_CLASS_IDS and box.score >= attribute_score_threshold) or box.classid not in ATTRIBUTE_CLASS_IDS
    ]
    result_boxes = [
        box for box in result_boxes
        if (box.classid in KEYPOINT_CLASS_IDS and box.score >= keypoint_threshold) or box.classid not in KEYPOINT_CLASS_IDS
    ]

    if not disable_generation_identification_mode:
        body_boxes = [box for box in result_boxes if box.classid == 0]
        generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
        find_most_relevant_obj(body_boxes, generation_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]

    if not disable_gender_identification_mode:
        body_boxes = [box for box in result_boxes if box.classid == 0]
        gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
        find_most_relevant_obj(body_boxes, gender_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]

    if not disable_headpose_identification_mode:
        head_boxes = [box for box in result_boxes if box.classid == 7]
        headpose_boxes = [box for box in result_boxes if box.classid in [8, 9, 10, 11, 12, 13, 14, 15]]
        find_most_relevant_obj(head_boxes, headpose_boxes)
    result_boxes = [box for box in result_boxes if box.classid not in [8, 9, 10, 11, 12, 13, 14, 15]]

    if not disable_left_and_right_hand_identification_mode:
        for parent_classid, child_classids in SIDE_PARENT_TO_CHILDREN.items():
            parent_boxes = [box for box in result_boxes if box.classid == parent_classid]
            side_boxes = [box for box in result_boxes if box.classid in child_classids]
            find_most_relevant_obj(parent_boxes, side_boxes)
    if not enable_bone_drawing_mode:
        result_boxes = [box for box in result_boxes if box.classid not in SIDE_ATTR_CLASS_IDS]

    for target_classid in KEYPOINT_NMS_CLASS_IDS:
        keypoint_boxes = [box for box in result_boxes if box.classid == target_classid]
        filtered_keypoint_boxes = nms(keypoint_boxes, iou_threshold=0.20)
        result_boxes = [box for box in result_boxes if box.classid != target_classid]
        result_boxes.extend(filtered_keypoint_boxes)

    return result_boxes


def prepare_prediction_payload(
    boxes: List[Box],
    result: Dict[str, torch.Tensor],
    mask_threshold: float,
    enable_masks: bool,
    enable_contours: bool,
    mask_bilateral_d: int = 0,
    mask_bilateral_sigma_color: float = 1.0,
    mask_bilateral_sigma_space: float = 3.0,
) -> List[Dict[str, object]]:
    masks = result.get('masks') if enable_masks else None
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu()
    contours = result.get('contours') if enable_contours else None
    if contours is not None and torch.is_tensor(contours):
        contours = contours.detach().cpu()

    records: List[Dict[str, object]] = []
    for box in boxes:
        record: Dict[str, object] = {
            'label': box.classid,
            'score': box.score,
            'box_xyxy': [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
            'generation': box.generation,
            'gender': box.gender,
            'handedness': box.handedness,
            'head_pose': box.head_pose,
            'track_id': box.track_id,
        }

        if masks is not None and box.classid == BODY_CLASS_ID and box.source_idx >= 0:
            mask_probs = lookup_mask_probs(masks, box.source_idx)
            if mask_probs is not None:
                mask_probs = postprocess_body_mask_probs(
                    mask_probs,
                    bilateral_d=mask_bilateral_d,
                    bilateral_sigma_color=mask_bilateral_sigma_color,
                    bilateral_sigma_space=mask_bilateral_sigma_space,
                )
                binary_mask = mask_probs >= mask_threshold
                binary_mask = clip_binary_mask_to_box(binary_mask, box)
                mask_bbox = binary_mask_bbox(binary_mask)
                if mask_bbox is not None:
                    record['mask_area'] = int(binary_mask.sum())
                    record['mask_bbox'] = mask_bbox

        if contours is not None and box.classid == BODY_CLASS_ID and box.source_idx >= 0:
            contour_probs = lookup_mask_probs(contours, box.source_idx)
            if contour_probs is not None:
                binary_contour = contour_probs >= mask_threshold
                binary_contour = clip_binary_mask_to_box(binary_contour, box)
                contour_bbox = binary_mask_bbox(binary_contour)
                if contour_bbox is not None:
                    record['contour_area'] = int(binary_contour.sum())
                    record['contour_bbox'] = contour_bbox

        records.append(record)
    return records


def lookup_mask_probs(
    mask_store: torch.Tensor | Dict[int, torch.Tensor] | None,
    source_idx: int,
) -> np.ndarray | None:
    if mask_store is None or source_idx < 0:
        return None

    if isinstance(mask_store, dict):
        mask_value = mask_store.get(source_idx)
    else:
        if source_idx >= len(mask_store):
            return None
        mask_value = mask_store[source_idx]

    if mask_value is None:
        return None

    if torch.is_tensor(mask_value):
        mask_array = mask_value.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_value)

    if mask_array.ndim == 3:
        return mask_array[0]
    return mask_array


def postprocess_body_mask_probs(
    mask_probs: np.ndarray,
    bilateral_d: int = 0,
    bilateral_sigma_color: float = 1.0,
    bilateral_sigma_space: float = 3.0,
) -> np.ndarray:
    if bilateral_d <= 1 or bilateral_sigma_color <= 0.0 or bilateral_sigma_space <= 0.0:
        return mask_probs

    filtered = cv2.bilateralFilter(
        np.ascontiguousarray(mask_probs, dtype=np.float32),
        bilateral_d,
        bilateral_sigma_color,
        bilateral_sigma_space,
    )
    return np.clip(filtered, 0.0, 1.0)


def clip_binary_mask_to_box(
    binary_mask: np.ndarray,
    box: Box,
    padding: int = MASK_CLEANUP_PADDING,
) -> np.ndarray:
    if binary_mask.size == 0:
        return binary_mask

    image_height, image_width = binary_mask.shape[:2]
    clip_x1 = max(0, box.x1 - padding)
    clip_y1 = max(0, box.y1 - padding)
    clip_x2 = min(image_width - 1, box.x2 + padding)
    clip_y2 = min(image_height - 1, box.y2 + padding)

    if (
        clip_x1 == 0 and clip_y1 == 0
        and clip_x2 == image_width - 1 and clip_y2 == image_height - 1
    ):
        return binary_mask

    has_outside_pixels = False
    if clip_y1 > 0 and binary_mask[:clip_y1, :].any():
        has_outside_pixels = True
    elif clip_y2 + 1 < image_height and binary_mask[clip_y2 + 1:, :].any():
        has_outside_pixels = True
    elif clip_x1 > 0 and binary_mask[:, :clip_x1].any():
        has_outside_pixels = True
    elif clip_x2 + 1 < image_width and binary_mask[:, clip_x2 + 1:].any():
        has_outside_pixels = True

    if not has_outside_pixels:
        return binary_mask

    clipped_mask = binary_mask.copy()
    if clip_y1 > 0:
        clipped_mask[:clip_y1, :] = False
    if clip_y2 + 1 < image_height:
        clipped_mask[clip_y2 + 1:, :] = False
    if clip_x1 > 0:
        clipped_mask[:, :clip_x1] = False
    if clip_x2 + 1 < image_width:
        clipped_mask[:, clip_x2 + 1:] = False
    return clipped_mask


def build_body_mask_entries(
    boxes: List[Box],
    result: Dict[str, torch.Tensor],
    args,
    mask_threshold: float,
) -> List[Tuple[Box, np.ndarray, np.ndarray]]:
    masks = result.get('masks')
    if masks is None:
        return []

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()

    body_masks: List[Tuple[Box, np.ndarray, np.ndarray]] = []
    for body_box in boxes:
        if body_box.classid != BODY_CLASS_ID or body_box.source_idx < 0:
            continue
        mask_probs = lookup_mask_probs(masks, body_box.source_idx)
        if mask_probs is None:
            continue
        mask_probs = postprocess_body_mask_probs(
            mask_probs,
            bilateral_d=args.mask_bilateral_d,
            bilateral_sigma_color=args.mask_bilateral_sigma_color,
            bilateral_sigma_space=args.mask_bilateral_sigma_space,
        )
        binary_mask = clip_binary_mask_to_box(mask_probs >= mask_threshold, body_box)
        if binary_mask.any():
            body_masks.append((body_box, mask_probs, binary_mask))
    return body_masks


def clipped_box_slice(box: Box, image_shape: Sequence[int]) -> Optional[Tuple[slice, slice]]:
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        return None

    x1 = max(0, min(box.x1, image_width - 1))
    y1 = max(0, min(box.y1, image_height - 1))
    x2 = max(0, min(box.x2, image_width - 1))
    y2 = max(0, min(box.y2, image_height - 1))
    if x2 < x1 or y2 < y1:
        return None
    return slice(y1, y2 + 1), slice(x1, x2 + 1)


def build_keypoint_mask_assignment_context(
    boxes: List[Box],
    result: Dict[str, torch.Tensor],
    args,
    mask_threshold: float,
) -> Tuple[Dict[int, int], Dict[int, KeypointInstanceQuality]]:
    body_masks = build_body_mask_entries(
        boxes=boxes,
        result=result,
        args=args,
        mask_threshold=mask_threshold,
    )
    if not body_masks:
        return {}, {}

    keypoint_mask_instance_map: Dict[int, int] = {}
    keypoint_instance_quality_map: Dict[int, KeypointInstanceQuality] = {}
    for keypoint_box in boxes:
        if keypoint_box.classid not in SKELETON_ASSIGNMENT_KEYPOINT_IDS:
            continue

        best_match: Tuple[float, float, int, int] | None = None
        instance_pixel_counts: Dict[int, int] = {}
        for body_box, mask_probs, binary_mask in body_masks:
            y, x = keypoint_box.cy, keypoint_box.cx
            if y < 0 or x < 0 or y >= binary_mask.shape[0] or x >= binary_mask.shape[1]:
                continue

            source_idx = int(body_box.source_idx)
            keypoint_slice = clipped_box_slice(keypoint_box, binary_mask.shape)
            if keypoint_slice is not None:
                pixel_count = int(binary_mask[keypoint_slice].sum())
                if pixel_count > 0:
                    instance_pixel_counts[source_idx] = pixel_count

            if not bool(binary_mask[y, x]):
                continue

            candidate = (float(mask_probs[y, x]), float(body_box.score), -source_idx, source_idx)
            if best_match is None or candidate > best_match:
                best_match = candidate

        if best_match is not None:
            assigned_instance = best_match[3]
            keypoint_mask_instance_map[id(keypoint_box)] = assigned_instance
            total_pixels = sum(instance_pixel_counts.values())
            assigned_pixels = instance_pixel_counts.get(assigned_instance, 0)
            foreign_pixels = max(0, total_pixels - assigned_pixels)
            assigned_pixel_share = float(assigned_pixels) / float(total_pixels) if total_pixels > 0 else 1.0
            foreign_pixel_share = float(foreign_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
            is_mixed = (
                foreign_pixels >= MIXED_KEYPOINT_FOREIGN_PIXEL_THRESHOLD
                and foreign_pixel_share >= MIXED_KEYPOINT_FOREIGN_SHARE_THRESHOLD
            )
            keypoint_instance_quality_map[id(keypoint_box)] = KeypointInstanceQuality(
                is_mixed=is_mixed,
                assigned_pixel_share=assigned_pixel_share,
                assigned_pixel_count=assigned_pixels,
                foreign_pixel_count=foreign_pixels,
            )

    return keypoint_mask_instance_map, keypoint_instance_quality_map


def build_keypoint_mask_instance_map(
    boxes: List[Box],
    result: Dict[str, torch.Tensor],
    args,
    mask_threshold: float,
) -> Dict[int, int]:
    keypoint_mask_instance_map, _ = build_keypoint_mask_assignment_context(
        boxes=boxes,
        result=result,
        args=args,
        mask_threshold=mask_threshold,
    )
    return keypoint_mask_instance_map


def overlay_body_masks(
    image: np.ndarray,
    result: Dict[str, torch.Tensor],
    boxes: List[Box],
    args,
    mask_threshold: float,
    mask_alpha: int,
    disable_render_classids: set[int],
    track_color_cache: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    masks = result.get('masks')
    if masks is None or BODY_CLASS_ID in disable_render_classids:
        return image

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
    overlay = np.zeros_like(image, dtype=np.uint8)
    mask_union = np.zeros(image.shape[:2], dtype=bool)
    body_instance_idx = 0

    for box in boxes:
        if box.classid != BODY_CLASS_ID or box.source_idx < 0:
            continue
        mask_probs = lookup_mask_probs(masks, box.source_idx)
        if mask_probs is None:
            continue
        mask_probs = postprocess_body_mask_probs(
            mask_probs,
            bilateral_d=args.mask_bilateral_d,
            bilateral_sigma_color=args.mask_bilateral_sigma_color,
            bilateral_sigma_space=args.mask_bilateral_sigma_space,
        )
        binary_mask = mask_probs >= mask_threshold
        if not binary_mask.any():
            continue
        binary_mask = clip_binary_mask_to_box(binary_mask, box)
        cached_color = track_color_cache.get(box.track_id) if track_color_cache is not None and box.track_id > 0 else None
        if isinstance(cached_color, np.ndarray):
            instance_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
        else:
            instance_color = make_instance_color(body_instance_idx)
        overlay[binary_mask] = np.array(instance_color, dtype=np.uint8)
        mask_union |= binary_mask
        body_instance_idx += 1

    if not mask_union.any():
        return image

    alpha = float(mask_alpha) / 255.0
    rendered = image.astype(np.float32, copy=True)
    rendered[mask_union] = rendered[mask_union] * (1.0 - alpha) + overlay[mask_union].astype(np.float32) * alpha
    return np.clip(rendered, 0, 255).astype(np.uint8)


def overlay_body_contours(
    image: np.ndarray,
    result: Dict[str, torch.Tensor],
    boxes: List[Box],
    contour_threshold: float,
    disable_render_classids: set[int],
    track_color_cache: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    contours = result.get('contours')
    if contours is None or BODY_CLASS_ID in disable_render_classids:
        return image

    if torch.is_tensor(contours):
        contours = contours.detach().cpu()
    rendered = image.copy()
    body_instance_idx = 0

    for box in boxes:
        if box.classid != BODY_CLASS_ID or box.source_idx < 0:
            continue
        contour_probs = lookup_mask_probs(contours, box.source_idx)
        if contour_probs is None:
            continue
        binary_contour = clip_binary_mask_to_box(contour_probs >= contour_threshold, box).astype(np.uint8)
        if not binary_contour.any():
            continue

        contour_segments, _ = cv2.findContours(binary_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contour_segments:
            continue

        cached_color = track_color_cache.get(box.track_id) if track_color_cache is not None and box.track_id > 0 else None
        if isinstance(cached_color, np.ndarray):
            instance_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
        else:
            instance_color = make_instance_color(body_instance_idx)
        cv2.drawContours(
            rendered,
            contour_segments,
            contourIdx=-1,
            color=instance_color,
            thickness=1,
        )
        body_instance_idx += 1

    return rendered


def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
    dashes = max(1, int(dist / dash_length))
    for idx in range(dashes):
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * idx / dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * idx / dashes),
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (idx + 0.5) / dashes),
            int(pt1[1] + (pt2[1] - pt1[1]) * (idx + 0.5) / dashes),
        )
        cv2.line(image, start, end, color, thickness)


def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, top_right, color, thickness, dash_length)
    draw_dashed_line(image, top_right, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bottom_left, color, thickness, dash_length)
    draw_dashed_line(image, bottom_left, top_left, color, thickness, dash_length)


def is_handedness_compatible(parent_box: Box, child_box: Box) -> bool:
    parent_side_aware = parent_box.classid in SIDE_AWARE_SKELETON_CLASS_IDS
    child_side_aware = child_box.classid in SIDE_AWARE_SKELETON_CLASS_IDS

    if parent_side_aware and parent_box.handedness < 0:
        return False
    if child_side_aware and child_box.handedness < 0:
        return False
    if parent_side_aware and child_side_aware:
        return parent_box.handedness == child_box.handedness
    return True


def box_area(box: Box) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def suppress_duplicate_skeleton_keypoints(
    boxes: List[Box],
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
) -> List[Box]:
    body_by_source_idx = {
        int(box.source_idx): box
        for box in boxes
        if box.classid == BODY_CLASS_ID and box.source_idx >= 0
    }
    body_by_person_id = {
        int(box.person_id): box
        for box in boxes
        if box.classid == BODY_CLASS_ID and box.person_id >= 0
    }
    selected_keypoints: Dict[Tuple[str, int, int, int], Tuple[Box, Tuple[int, float, float, float]]] = {}
    passthrough_keypoint_ids: set[int] = set()

    for box in boxes:
        if box.classid not in SKELETON_ASSIGNMENT_KEYPOINT_IDS:
            continue

        if keypoint_mask_instance_map is not None:
            instance_id = keypoint_mask_instance_map.get(id(box))
            if instance_id is None:
                passthrough_keypoint_ids.add(id(box))
                continue
            body_box = body_by_source_idx.get(int(instance_id))
            instance_key = ('mask', int(instance_id), box.classid, box.handedness)
        else:
            if box.person_id is None or box.person_id < 0:
                passthrough_keypoint_ids.add(id(box))
                continue
            body_box = body_by_person_id.get(int(box.person_id))
            instance_key = ('person', int(box.person_id), box.classid, box.handedness)

        body_area = box_area(body_box) if body_box is not None else 0
        if body_area <= 0:
            passthrough_keypoint_ids.add(id(box))
            continue

        area_ratio = box_area(box) / float(body_area)
        quality = keypoint_instance_quality_map.get(id(box)) if keypoint_instance_quality_map is not None else None
        clean_priority = 1 if quality is None or not quality.is_mixed else 0
        assigned_pixel_share = quality.assigned_pixel_share if quality is not None else 1.0
        priority = (clean_priority, assigned_pixel_share, area_ratio, float(box.score))
        candidate = (box, priority)
        current = selected_keypoints.get(instance_key)
        if current is None or priority > current[1]:
            selected_keypoints[instance_key] = candidate

    selected_keypoint_ids = {id(candidate[0]) for candidate in selected_keypoints.values()}
    return [
        box for box in boxes
        if (
            box.classid not in SKELETON_ASSIGNMENT_KEYPOINT_IDS
            or id(box) in selected_keypoint_ids
            or id(box) in passthrough_keypoint_ids
        )
    ]


def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color: Tuple[int, int, int] = (0, 255, 255),
    max_dist_threshold: Optional[float] = 500.0,
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
) -> None:
    person_boxes = [box for box in boxes if box.classid == 0]
    for person_id, person_box in enumerate(person_boxes):
        person_box.person_id = person_id

    for box in boxes:
        if box.classid in SKELETON_ASSIGNMENT_KEYPOINT_IDS:
            box.person_id = -1
            for person_box in person_boxes:
                if person_box.x1 <= box.cx <= person_box.x2 and person_box.y1 <= box.cy <= person_box.y2:
                    box.person_id = person_box.person_id
                    break

    skeleton_boxes = suppress_duplicate_skeleton_keypoints(
        boxes=boxes,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
    )
    person_skeleton_boxes = suppress_duplicate_skeleton_keypoints(
        boxes=boxes,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
    )
    line_registry = SkeletonLineRegistry()
    bone_boxes, classid_to_boxes = draw_bone_supported_skeleton(
        image=image,
        boxes=skeleton_boxes,
        color=color,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        line_registry=line_registry,
    )
    draw_bone_fallback_skeleton(
        image=image,
        color=color,
        bone_boxes=bone_boxes,
        classid_to_boxes=classid_to_boxes,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        line_registry=line_registry,
    )
    draw_instance_skeleton(
        image=image,
        boxes=skeleton_boxes,
        color=color,
        max_dist_threshold=max_dist_threshold,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        line_registry=line_registry,
    )
    draw_bone_mask_mismatch_rescue_skeleton(
        image=image,
        boxes=person_skeleton_boxes,
        color=color,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        line_registry=line_registry,
    )


def draw_instance_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color: Tuple[int, int, int],
    max_dist_threshold: Optional[float],
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
    line_registry: Optional[SkeletonLineRegistry] = None,
) -> SkeletonLineRegistry:
    if line_registry is None:
        line_registry = SkeletonLineRegistry()

    classid_to_boxes: Dict[int, List[Box]] = {}
    for box in boxes:
        classid_to_boxes.setdefault(box.classid, []).append(box)

    edge_counts = Counter(EDGES)
    lines_to_draw: List[Tuple[Box, Box]] = []

    for (parent_id, child_id), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(parent_id, [])
        child_list = classid_to_boxes.get(child_id, [])
        if not parent_list or not child_list:
            continue

        parent_capacity = [
            1 if parent_box.handedness >= 0 else repeat_count
            for parent_box in parent_list
        ]
        child_used = [False] * len(child_list)
        pair_candidates: List[Tuple[int, float, int, int]] = []

        for parent_idx, parent_box in enumerate(parent_list):
            for child_idx, child_box in enumerate(child_list):
                dx = parent_box.cx - child_box.cx
                dy = parent_box.cy - child_box.cy
                dist_sq = dx * dx + dy * dy
                if max_dist_threshold is not None and dist_sq > max_dist_threshold * max_dist_threshold:
                    continue
                if not is_handedness_compatible(parent_box, child_box):
                    continue
                if keypoint_mask_instance_map is not None and instance_edge_uses_mixed_keypoint(
                    parent_box,
                    child_box,
                    keypoint_instance_quality_map,
                ):
                    continue

                if keypoint_mask_instance_map is not None:
                    parent_mask_instance = keypoint_mask_instance_map.get(id(parent_box))
                    child_mask_instance = keypoint_mask_instance_map.get(id(child_box))
                    if (
                        parent_mask_instance is None
                        or child_mask_instance is None
                        or parent_mask_instance != child_mask_instance
                    ):
                        continue
                    pair_candidates.append((0, dist_sq, parent_idx, child_idx))
                elif parent_box.person_id == child_box.person_id and parent_box.person_id is not None:
                    pair_candidates.append((0, dist_sq, parent_idx, child_idx))

        pair_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        for _, _, parent_idx, child_idx in pair_candidates:
            if parent_capacity[parent_idx] > 0 and not child_used[child_idx]:
                parent_box = parent_list[parent_idx]
                child_box = child_list[child_idx]
                lines_to_draw.append((parent_box, child_box))
                parent_capacity[parent_idx] -= 1
                child_used[child_idx] = True

    for parent_box, child_box in lines_to_draw:
        if line_registry.add(parent_box, child_box):
            cv2.line(image, (parent_box.cx, parent_box.cy), (child_box.cx, child_box.cy), color, thickness=2)
    return line_registry


def instance_edge_uses_mixed_keypoint(
    parent_box: Box,
    child_box: Box,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]],
) -> bool:
    if keypoint_instance_quality_map is None:
        return False
    parent_quality = keypoint_instance_quality_map.get(id(parent_box))
    child_quality = keypoint_instance_quality_map.get(id(child_box))
    return is_low_quality_mixed_keypoint(parent_quality) or is_low_quality_mixed_keypoint(child_quality)


def is_low_quality_mixed_keypoint(quality: Optional[KeypointInstanceQuality]) -> bool:
    return bool(
        quality is not None
        and quality.is_mixed
        and quality.assigned_pixel_share < INSTANCE_EDGE_MIXED_ASSIGNED_SHARE_MIN
    )


def build_bone_inside_classid_to_boxes(
    bone_box: Box,
    classid_to_boxes: Dict[int, List[Box]],
    target_class_ids: set[int],
) -> Dict[int, List[Box]]:
    return {
        class_id: [
            box
            for box in classid_to_boxes.get(class_id, [])
            if keypoint_inside_box(box, bone_box)
        ]
        for class_id in target_class_ids
    }


def bone_candidate_heap_key(
    candidate: Tuple[int, int, float, int, Box, Box],
) -> Tuple[int, int, float, int]:
    slot_priority, clean_priority, score, candidate_order, _, _ = candidate
    return (-slot_priority, -clean_priority, -score, candidate_order)


def build_bone_candidate_lists(
    bone_boxes: List[Box],
    classid_to_boxes: Dict[int, List[Box]],
    edge_pairs: Sequence[Tuple[int, int]],
    keypoint_mask_instance_map: Optional[Dict[int, int]],
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]],
    line_registry: SkeletonLineRegistry,
) -> List[List[Tuple[int, int, float, int, Box, Box]]]:
    target_class_ids = {class_id for edge in edge_pairs for class_id in edge}
    candidate_order = 0
    candidate_lists: List[List[Tuple[int, int, float, int, Box, Box]]] = []

    for bone_box in bone_boxes:
        bone_candidates: List[Tuple[int, int, float, int, Box, Box]] = []
        inside_classid_to_boxes = build_bone_inside_classid_to_boxes(
            bone_box,
            classid_to_boxes,
            target_class_ids,
        )
        for first_id, second_id in edge_pairs:
            first_list = inside_classid_to_boxes.get(first_id, [])
            second_list = inside_classid_to_boxes.get(second_id, [])
            if not first_list or not second_list:
                continue

            for first_box in first_list:
                for second_box in second_list:
                    score = bone_supported_edge_score_after_inside_check(
                        bone_box,
                        first_box,
                        second_box,
                        keypoint_mask_instance_map=keypoint_mask_instance_map,
                        keypoint_instance_quality_map=keypoint_instance_quality_map,
                        allow_mixed_instance_override=True,
                        line_registry=line_registry,
                    )
                    if score is None:
                        candidate_order += 1
                        continue
                    clean_priority = bone_candidate_clean_priority(
                        first_box,
                        second_box,
                        keypoint_instance_quality_map,
                    )
                    slot_priority = line_registry.connection_slot_priority(first_box, second_box)
                    bone_candidates.append(
                        (slot_priority, clean_priority, score, candidate_order, first_box, second_box)
                    )
                    candidate_order += 1

        bone_candidates.sort(key=bone_candidate_heap_key)
        candidate_lists.append(bone_candidates)

    return candidate_lists


def draw_bone_candidate_lists(
    image: np.ndarray,
    color: Tuple[int, int, int],
    candidate_lists: List[List[Tuple[int, int, float, int, Box, Box]]],
    line_registry: SkeletonLineRegistry,
) -> None:
    candidate_heap: List[Tuple[int, int, float, int, int, int]] = []
    for bone_index, bone_candidates in enumerate(candidate_lists):
        if bone_candidates:
            heapq.heappush(candidate_heap, (*bone_candidate_heap_key(bone_candidates[0]), bone_index, 0))

    used_bone_indices: set[int] = set()
    while candidate_heap:
        _, _, _, _, bone_index, candidate_index = heapq.heappop(candidate_heap)
        if bone_index in used_bone_indices:
            continue

        bone_candidates = candidate_lists[bone_index]
        _, _, _, _, first_box, second_box = bone_candidates[candidate_index]
        if line_registry.add(first_box, second_box):
            used_bone_indices.add(bone_index)
            cv2.line(image, (first_box.cx, first_box.cy), (second_box.cx, second_box.cy), color, thickness=2)
            continue

        next_candidate_index = candidate_index + 1
        if next_candidate_index < len(bone_candidates):
            heapq.heappush(
                candidate_heap,
                (*bone_candidate_heap_key(bone_candidates[next_candidate_index]), bone_index, next_candidate_index),
            )


def draw_bone_supported_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color: Tuple[int, int, int],
    keypoint_mask_instance_map: Optional[Dict[int, int]],
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]],
    line_registry: SkeletonLineRegistry,
) -> Tuple[List[Box], Dict[int, List[Box]]]:
    bone_boxes = [box for box in boxes if box.classid == BONE_CLASS_ID]

    classid_to_boxes: Dict[int, List[Box]] = {}
    for box in boxes:
        classid_to_boxes.setdefault(box.classid, []).append(box)

    if not bone_boxes:
        return bone_boxes, classid_to_boxes

    candidate_lists = build_bone_candidate_lists(
        bone_boxes=bone_boxes,
        classid_to_boxes=classid_to_boxes,
        edge_pairs=BONE_EDGE_PAIRS,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        line_registry=line_registry,
    )
    draw_bone_candidate_lists(
        image=image,
        color=color,
        candidate_lists=candidate_lists,
        line_registry=line_registry,
    )

    return bone_boxes, classid_to_boxes


def bone_candidate_clean_priority(
    first_box: Box,
    second_box: Box,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]],
) -> int:
    if keypoint_instance_quality_map is None:
        return 1
    first_quality = keypoint_instance_quality_map.get(id(first_box))
    second_quality = keypoint_instance_quality_map.get(id(second_box))
    if (first_quality is not None and first_quality.is_mixed) or (
        second_quality is not None and second_quality.is_mixed
    ):
        return 0
    return 1


def draw_bone_fallback_skeleton(
    image: np.ndarray,
    color: Tuple[int, int, int],
    bone_boxes: List[Box],
    classid_to_boxes: Dict[int, List[Box]],
    keypoint_mask_instance_map: Optional[Dict[int, int]],
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]],
    line_registry: SkeletonLineRegistry,
) -> None:
    fallback_edge_pairs = tuple(dict.fromkeys(EDGES))
    candidate_lists = build_bone_candidate_lists(
        bone_boxes=bone_boxes,
        classid_to_boxes=classid_to_boxes,
        edge_pairs=fallback_edge_pairs,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        line_registry=line_registry,
    )
    draw_bone_candidate_lists(
        image=image,
        color=color,
        candidate_lists=candidate_lists,
        line_registry=line_registry,
    )


def draw_bone_mask_mismatch_rescue_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color: Tuple[int, int, int],
    keypoint_mask_instance_map: Optional[Dict[int, int]],
    line_registry: SkeletonLineRegistry,
) -> None:
    if keypoint_mask_instance_map is None:
        return

    bone_boxes = [box for box in boxes if box.classid == BONE_CLASS_ID]
    if not bone_boxes:
        return

    classid_to_boxes: Dict[int, List[Box]] = {}
    for box in boxes:
        classid_to_boxes.setdefault(box.classid, []).append(box)

    rescue_edge_pairs = tuple(dict.fromkeys(EDGES))
    selected_lines: Dict[Tuple[int, int], Tuple[float, Box, Box]] = {}

    for bone_box in bone_boxes:
        best_candidate: Optional[Tuple[float, Box, Box]] = None
        for first_id, second_id in rescue_edge_pairs:
            first_list = classid_to_boxes.get(first_id, [])
            second_list = classid_to_boxes.get(second_id, [])
            if not first_list or not second_list:
                continue

            for first_box in first_list:
                for second_box in second_list:
                    score = bone_mask_mismatch_rescue_edge_score(
                        bone_box,
                        first_box,
                        second_box,
                        keypoint_mask_instance_map=keypoint_mask_instance_map,
                        line_registry=line_registry,
                    )
                    if score is None:
                        continue
                    candidate = (score, first_box, second_box)
                    if best_candidate is None or candidate[0] > best_candidate[0]:
                        best_candidate = candidate

        if best_candidate is None:
            continue

        score, first_box, second_box = best_candidate
        pair_key = line_registry.line_key(first_box, second_box)
        if pair_key not in selected_lines or score > selected_lines[pair_key][0]:
            selected_lines[pair_key] = (score, first_box, second_box)

    for _, first_box, second_box in sorted(selected_lines.values(), key=lambda item: item[0], reverse=True):
        if line_registry.add(first_box, second_box):
            cv2.line(image, (first_box.cx, first_box.cy), (second_box.cx, second_box.cy), color, thickness=2)


def bone_mask_mismatch_rescue_edge_score(
    bone_box: Box,
    first_box: Box,
    second_box: Box,
    keypoint_mask_instance_map: Dict[int, int],
    line_registry: SkeletonLineRegistry,
) -> Optional[float]:
    if line_registry.has_line(first_box, second_box):
        return None
    if not is_handedness_compatible(first_box, second_box):
        return None
    if first_box.person_id != second_box.person_id or first_box.person_id < 0:
        return None

    first_mask_instance = keypoint_mask_instance_map.get(id(first_box))
    second_mask_instance = keypoint_mask_instance_map.get(id(second_box))
    if first_mask_instance is None or second_mask_instance is None or first_mask_instance == second_mask_instance:
        return None

    return bone_supported_edge_score(
        bone_box,
        first_box,
        second_box,
        keypoint_mask_instance_map=None,
        line_registry=line_registry,
    )


def bone_supported_edge_score_after_inside_check(
    bone_box: Box,
    first_box: Box,
    second_box: Box,
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
    allow_mixed_instance_override: bool = False,
    line_registry: Optional[SkeletonLineRegistry] = None,
) -> Optional[float]:
    if line_registry is not None and not line_registry.can_add(first_box, second_box):
        return None
    if not keypoints_share_instance_or_person(
        first_box,
        second_box,
        keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        allow_mixed_instance_override=allow_mixed_instance_override,
    ):
        return None

    width = max(1, bone_box.x2 - bone_box.x1)
    height = max(1, bone_box.y2 - bone_box.y1)
    dx = abs(first_box.cx - second_box.cx)
    dy = abs(first_box.cy - second_box.cy)
    long_axis = max(width, height)
    short_axis = max(1, min(width, height))
    long_axis_separation = max(dx, dy) / float(long_axis)
    if long_axis_separation < 0.45:
        return None

    aspect_ratio = long_axis / float(short_axis)
    if aspect_ratio < 1.8:
        if dx / float(width) < 0.30 or dy / float(height) < 0.30:
            return None

    midpoint_x = (first_box.cx + second_box.cx) * 0.5
    midpoint_y = (first_box.cy + second_box.cy) * 0.5
    bone_center_x = (bone_box.x1 + bone_box.x2) * 0.5
    bone_center_y = (bone_box.y1 + bone_box.y2) * 0.5
    center_offset = math.hypot(
        (midpoint_x - bone_center_x) / float(width),
        (midpoint_y - bone_center_y) / float(height),
    )
    if center_offset > 0.35:
        return None

    return float(bone_box.score) * 1000.0 + long_axis_separation - center_offset


def bone_supported_edge_score(
    bone_box: Box,
    first_box: Box,
    second_box: Box,
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
    allow_mixed_instance_override: bool = False,
    line_registry: Optional[SkeletonLineRegistry] = None,
) -> Optional[float]:
    if not keypoint_inside_box(first_box, bone_box) or not keypoint_inside_box(second_box, bone_box):
        return None
    return bone_supported_edge_score_after_inside_check(
        bone_box,
        first_box,
        second_box,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
        allow_mixed_instance_override=allow_mixed_instance_override,
        line_registry=line_registry,
    )


def keypoints_share_instance_or_person(
    first_box: Box,
    second_box: Box,
    keypoint_mask_instance_map: Optional[Dict[int, int]],
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
    allow_mixed_instance_override: bool = False,
) -> bool:
    if keypoint_mask_instance_map is not None:
        first_mask_instance = keypoint_mask_instance_map.get(id(first_box))
        second_mask_instance = keypoint_mask_instance_map.get(id(second_box))
        if first_mask_instance is None or second_mask_instance is None or first_mask_instance == second_mask_instance:
            return True
        return bool(
            allow_mixed_instance_override
            and instance_edge_uses_mixed_keypoint(first_box, second_box, keypoint_instance_quality_map)
        )
    return first_box.person_id == second_box.person_id and first_box.person_id >= 0


def keypoint_inside_box(keypoint_box: Box, container_box: Box, padding: int = 1) -> bool:
    return (
        container_box.x1 - padding <= keypoint_box.cx <= container_box.x2 + padding
        and container_box.y1 - padding <= keypoint_box.cy <= container_box.y2 + padding
    )


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.7,
    outline_color: Tuple[int, int, int] = (255, 255, 255),
    outline_thickness: int = 2,
    text_thickness: int = 1,
) -> None:
    if not text:
        return
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)


def apply_face_mosaic(image: np.ndarray, box: Box) -> None:
    crop = image[box.y1:box.y2, box.x1:box.x2, :]
    if crop.size == 0:
        return
    width = max(1, abs(box.x2 - box.x1))
    height = max(1, abs(box.y2 - box.y1))
    small_box = cv2.resize(crop, (3, 3))
    normal_box = cv2.resize(small_box, (width, height))
    if normal_box.shape[0] != height or normal_box.shape[1] != width:
        normal_box = cv2.resize(small_box, (width, height))
    image[box.y1:box.y2, box.x1:box.x2, :] = normal_box


def get_render_color(
    box: Box,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
) -> Tuple[int, int, int]:
    classid = box.classid
    color = (255, 255, 255)

    if classid == 0:
        if not disable_gender_identification_mode:
            if box.gender == 0:
                color = (255, 0, 0)
            elif box.gender == 1:
                color = (139, 116, 225)
            else:
                color = (0, 200, 255)
        else:
            color = (0, 200, 255)
    elif classid == 5:
        color = (0, 200, 255)
    elif classid == 6:
        color = (83, 36, 179)
    elif classid == 7:
        if not disable_headpose_identification_mode:
            color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216, 67, 21)
        else:
            color = (0, 0, 255)
    elif classid == 16:
        color = (0, 200, 255)
    elif classid == 17:
        color = (255, 0, 0)
    elif classid == 18:
        color = (0, 255, 0)
    elif classid == 19:
        color = (0, 0, 255)
    elif classid == 20:
        color = (203, 192, 255)
    elif classid == 21:
        color = (0, 0, 255)
    elif classid == 22:
        color = (255, 0, 0)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 23:
        color = LEFT_SIDE_COLOR
    elif classid == 24:
        color = RIGHT_SIDE_COLOR
    elif classid == 25:
        color = (252, 189, 107)
    elif classid == 26:
        color = (0, 255, 0)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 27:
        color = LEFT_SIDE_COLOR
    elif classid == 28:
        color = RIGHT_SIDE_COLOR
    elif classid == 29:
        color = (0, 0, 255)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 30:
        color = LEFT_SIDE_COLOR
    elif classid == 31:
        color = RIGHT_SIDE_COLOR
    elif classid == 32:
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
            else:
                color = (0, 255, 0)
        else:
            color = (0, 255, 0)
    elif classid == 33:
        color = LEFT_SIDE_COLOR
    elif classid == 34:
        color = RIGHT_SIDE_COLOR
    elif classid == 35:
        color = (0, 0, 255)
    elif classid == 36:
        color = (255, 0, 0)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 37:
        color = LEFT_SIDE_COLOR
    elif classid == 38:
        color = RIGHT_SIDE_COLOR
    elif classid == 39:
        color = (250, 0, 136)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 40:
        color = LEFT_SIDE_COLOR
    elif classid == 41:
        color = RIGHT_SIDE_COLOR
    elif classid == 42:
        color = (252, 189, 107)
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
    elif classid == 43:
        color = LEFT_SIDE_COLOR
    elif classid == 44:
        color = RIGHT_SIDE_COLOR
    elif classid == 45:
        if not disable_left_and_right_hand_identification_mode:
            if box.handedness == 0:
                color = LEFT_SIDE_COLOR
            elif box.handedness == 1:
                color = RIGHT_SIDE_COLOR
            else:
                color = (0, 255, 0)
        else:
            color = (0, 255, 0)
    elif classid == 46:
        color = LEFT_SIDE_COLOR
    elif classid == 47:
        color = RIGHT_SIDE_COLOR
    elif classid == BONE_CLASS_ID:
        color = BONE_BBOX_COLOR

    return color


def draw_detections(
    image: np.ndarray,
    boxes: List[Box],
    disable_render_classids: set[int],
    enable_bone_bbox_drawing_mode: bool,
    keypoint_drawing_mode: str,
    enable_bone_drawing_mode: bool,
    enable_face_mosaic: bool,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_left_and_right_label: bool,
    disable_headpose_identification_mode: bool,
    bounding_box_line_width: int,
    keypoint_dot_radius: int,
    enable_head_distance_measurement: bool,
    camera_horizontal_fov: int,
    enable_trackid_overlay: bool = False,
    track_color_cache: Optional[Dict[int, np.ndarray]] = None,
    keypoint_mask_instance_map: Optional[Dict[int, int]] = None,
    keypoint_instance_quality_map: Optional[Dict[int, KeypointInstanceQuality]] = None,
) -> np.ndarray:
    debug_image = image.copy()
    debug_image_h, debug_image_w = debug_image.shape[:2]
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1

    for box in boxes:
        classid = box.classid
        if classid in disable_render_classids:
            continue
        if classid == BONE_CLASS_ID and not enable_bone_bbox_drawing_mode:
            continue

        color = get_render_color(
            box,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )

        if (
            (classid == 0 and not disable_gender_identification_mode)
            or (classid == 7 and not disable_headpose_identification_mode)
            or (classid in SIDE_AWARE_OBJECT_CLASS_IDS and not disable_left_and_right_hand_identification_mode)
            or classid == 16
            or classid in KEYPOINT_DRAW_CLASS_IDS
        ):
            if classid == 0:
                if box.gender == -1:
                    draw_dashed_rectangle(
                        image=debug_image,
                        top_left=(box.x1, box.y1),
                        bottom_right=(box.x2, box.y2),
                        color=color,
                        thickness=2,
                        dash_length=10,
                    )
                else:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid == 7:
                if box.head_pose == -1:
                    draw_dashed_rectangle(
                        image=debug_image,
                        top_left=(box.x1, box.y1),
                        bottom_right=(box.x2, box.y2),
                        color=color,
                        thickness=2,
                        dash_length=10,
                    )
                else:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid == 16:
                if enable_face_mosaic:
                    apply_face_mosaic(debug_image, box)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid in SIDE_AWARE_OBJECT_CLASS_IDS:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)
            elif classid in KEYPOINT_DRAW_CLASS_IDS:
                if keypoint_drawing_mode in ['dot', 'both']:
                    cv2.circle(debug_image, (box.cx, box.cy), keypoint_dot_radius + 1, (255, 255, 255), -1)
                    cv2.circle(debug_image, (box.cx, box.cy), keypoint_dot_radius, color, -1)
                if keypoint_drawing_mode in ['box', 'both']:
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), 2)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
        else:
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), white_line_width)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

        if enable_trackid_overlay and classid == BODY_CLASS_ID and box.track_id > 0:
            track_text = f'ID: {box.track_id}'
            track_x = max(box.x1 - 5, 0)
            track_y = box.y1 - 30
            if track_y < 20:
                track_y = min(box.y2 + 25, debug_image_h - 10)
            cached_color = track_color_cache.get(box.track_id) if track_color_cache is not None else None
            if isinstance(cached_color, np.ndarray):
                text_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
            else:
                text_color = color
            cv2.putText(
                debug_image,
                track_text,
                (track_x, track_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                track_text,
                (track_x, track_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                1,
                cv2.LINE_AA,
            )

        generation_txt = ''
        if box.generation == 0:
            generation_txt = 'Adult'
        elif box.generation == 1:
            generation_txt = 'Child'

        gender_txt = ''
        if box.gender == 0:
            gender_txt = 'M'
        elif box.gender == 1:
            gender_txt = 'F'

        attr_txt = f'{generation_txt}({gender_txt})' if gender_txt else generation_txt
        headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
        attr_txt = f'{attr_txt} {headpose_txt}'.strip() if headpose_txt else attr_txt

        text_org = (
            box.x1 if box.x1 + 50 < debug_image_w else debug_image_w - 50,
            box.y1 - 10 if box.y1 - 25 > 0 else 20,
        )
        draw_text_with_outline(debug_image, attr_txt, text_org, color)

        if not disable_left_and_right_label:
            handedness_txt = ''
            if classid in LEFT_SIDE_CLASS_IDS:
                handedness_txt = 'L'
            elif classid in RIGHT_SIDE_CLASS_IDS:
                handedness_txt = 'R'
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            draw_text_with_outline(debug_image, handedness_txt, text_org, color)

        if enable_head_distance_measurement and classid == 7 and abs(box.x2 - box.x1) > 0:
            if camera_horizontal_fov > 90:
                focal_length = debug_image_w / (camera_horizontal_fov * (math.pi / 180))
            else:
                focal_length = debug_image_w / (2 * math.tan((camera_horizontal_fov / 2) * (math.pi / 180)))
            distance = (AVERAGE_HEAD_WIDTH * focal_length) / abs(box.x2 - box.x1)
            distance_org = (
                box.x1 + 5 if box.x1 < debug_image_w else debug_image_w - 50,
                box.y1 + 20 if box.y1 - 5 > 0 else 20,
            )
            cv2.putText(debug_image, f'{distance:.3f} m', distance_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{distance:.3f} m', distance_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 1, cv2.LINE_AA)

    if enable_bone_drawing_mode:
        draw_skeleton(
            image=debug_image,
            boxes=boxes,
            color=(0, 255, 255),
            max_dist_threshold=None,
            keypoint_mask_instance_map=keypoint_mask_instance_map,
            keypoint_instance_quality_map=keypoint_instance_quality_map,
        )

    return debug_image


class InferenceModel(nn.Module):
    def __init__(
        self,
        cfg: InferenceConfig,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device,
        mask_resize_origin: str = 'topleft',
    ):
        super().__init__()
        matched_state, missing_keys, mismatched_keys, unexpected_keys = matched_tensor_state(
            cfg.model.state_dict(),
            state_dict,
        )
        load_info = cfg.model.load_state_dict(matched_state, strict=False)
        if missing_keys or mismatched_keys or unexpected_keys:
            print(
                'Partially loaded checkpoint for inference: '
                f'matched={len(matched_state)}, '
                f'missing={len(missing_keys)}, '
                f'shape_mismatch={len(mismatched_keys)}, '
                f'unexpected={len(unexpected_keys)}'
            )
            if missing_keys:
                print(f'  Missing keys (first 10): {missing_keys[:10]}')
            if mismatched_keys:
                print(f'  Shape mismatches (first 10): {mismatched_keys[:10]}')
            if unexpected_keys:
                print(f'  Unexpected keys (first 10): {unexpected_keys[:10]}')
            if load_info.missing_keys:
                print(f'  load_state_dict missing keys (first 10): {load_info.missing_keys[:10]}')
            if load_info.unexpected_keys:
                print(f'  load_state_dict unexpected keys (first 10): {load_info.unexpected_keys[:10]}')
        self.model = cfg.model.eval().to(device)
        self.postprocessor = cfg.postprocessor.eval()
        self.postprocessor.mask_resize_origin = mask_resize_origin
        self.device = device
        self.last_inference_time = 0.0

    @torch.inference_mode()
    def forward(
        self,
        image_tensor: torch.Tensor,
        orig_target_sizes: torch.Tensor,
        return_masks: bool=False,
        return_contours: bool=False,
    ):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        inference_start_time = time.perf_counter()
        outputs = self.model(
            image_tensor,
            return_masks=return_masks,
            return_contours=return_contours,
        )
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_inference_time = time.perf_counter() - inference_start_time
        outputs = move_to_device(outputs, torch.device('cpu'))
        orig_target_sizes = orig_target_sizes.to('cpu')
        return self.postprocessor(
            outputs,
            orig_target_sizes,
            return_masks=return_masks,
            return_contours=return_contours,
        )


class OnnxInferenceModel:
    def __init__(
        self,
        model_path: Path,
        device_arg: str | None,
        inference_type: str,
        mask_resize_origin: str = 'topleft',
        mask_resize_mode: str = 'bilinear',
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError('onnxruntime is required for ONNX inference.') from exc

        providers = build_onnx_providers(device_arg, model_path, inference_type)
        session_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.mask_resize_origin = mask_resize_origin
        self.mask_resize_mode = mask_resize_mode
        self.providers = self.session.get_providers()
        self.last_inference_time = 0.0
        image_input = self.session.get_inputs()[0]
        image_shape = image_input.shape
        self.image_size = tuple(int(v) for v in image_shape[2:4]) if len(image_shape) >= 4 and all(isinstance(v, int) for v in image_shape[2:4]) else None

    def _decode_label_xyxy_score(
        self,
        label_xyxy_score: np.ndarray,
        orig_target_sizes: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        batch_results: List[Dict[str, torch.Tensor]] = []
        for batch_idx in range(label_xyxy_score.shape[0]):
            batch_pred = label_xyxy_score[batch_idx]
            boxes = torch.from_numpy(batch_pred[:, 1:5].astype(np.float32, copy=False))
            if 'orig_target_sizes' not in self.input_names:
                orig_w = float(orig_target_sizes[batch_idx, 0].item())
                orig_h = float(orig_target_sizes[batch_idx, 1].item())
                boxes[:, 0::2] *= orig_w
                boxes[:, 1::2] *= orig_h
            batch_results.append(
                {
                    'labels': torch.from_numpy(batch_pred[:, 0].astype(np.int64, copy=False)),
                    'boxes': boxes,
                    'scores': torch.from_numpy(batch_pred[:, 5].astype(np.float32, copy=False)),
                }
            )
        return batch_results

    def _resize_mask_batch(
        self,
        masks: np.ndarray,
        orig_target_sizes: torch.Tensor,
    ) -> List[torch.Tensor]:
        resized_batches: List[torch.Tensor] = []
        for batch_idx in range(masks.shape[0]):
            batch_masks = torch.from_numpy(masks[batch_idx]).to(dtype=torch.float32)
            batch_masks = resize_masks(
                batch_masks.unsqueeze(1),
                size=tuple(int(v) for v in orig_target_sizes[batch_idx, [1, 0]].tolist()),
                mode=self.mask_resize_mode,
                origin=self.mask_resize_origin,
            )
            if self.mask_resize_mode == 'bicubic':
                batch_masks = batch_masks.clamp(0.0, 1.0)
            resized_batches.append(batch_masks)
        return resized_batches

    def _resize_selected_mask_map(
        self,
        masks: np.ndarray,
        orig_target_size: torch.Tensor,
        selected_indices: Sequence[int],
    ) -> Dict[int, torch.Tensor]:
        unique_indices = sorted({int(idx) for idx in selected_indices if idx >= 0})
        if not unique_indices:
            return {}

        batch_masks = torch.from_numpy(masks[unique_indices]).to(dtype=torch.float32)
        resized_masks = resize_masks(
            batch_masks.unsqueeze(1),
            size=tuple(int(v) for v in orig_target_size[[1, 0]].tolist()),
            mode=self.mask_resize_mode,
            origin=self.mask_resize_origin,
        )
        if self.mask_resize_mode == 'bicubic':
            resized_masks = resized_masks.clamp(0.0, 1.0)
        return {
            source_idx: resized_masks[pos]
            for pos, source_idx in enumerate(unique_indices)
        }

    @torch.inference_mode()
    def __call__(
        self,
        image_tensor: torch.Tensor,
        orig_target_sizes: torch.Tensor,
        return_masks: bool = False,
        return_contours: bool = False,
    ):
        input_feed = {'images': image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)}
        if 'orig_target_sizes' in self.input_names:
            input_feed['orig_target_sizes'] = orig_target_sizes.detach().cpu().numpy().astype(np.float32, copy=False)

        requested_outputs = ['label_xyxy_score']
        if return_masks:
            requested_outputs.append('masks')
        if return_contours:
            requested_outputs.append('contours')
        inference_start_time = time.perf_counter()
        output_values = self.session.run(requested_outputs, input_feed)
        self.last_inference_time = time.perf_counter() - inference_start_time
        outputs = dict(zip(requested_outputs, output_values))

        if 'label_xyxy_score' not in outputs:
            raise KeyError('ONNX model output `label_xyxy_score` is required.')

        results = self._decode_label_xyxy_score(outputs['label_xyxy_score'], orig_target_sizes)

        if return_masks:
            if 'masks' not in outputs:
                raise RuntimeError('ONNX model does not provide `masks`, but --enable-masks was requested.')
            for batch_idx, result in enumerate(results):
                result['_onnx_masks'] = outputs['masks'][batch_idx]

        if return_contours:
            if 'contours' not in outputs:
                raise RuntimeError('ONNX model does not provide `contours`, but --enable-contours was requested.')
            for batch_idx, result in enumerate(results):
                result['_onnx_contours'] = outputs['contours'][batch_idx]

        return results


def save_predictions_json(output_dir: Path, image_path: Path, records: List[Dict[str, object]]) -> None:
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'image': image_path.name,
        'predictions': records,
    }
    with (pred_dir / f'{image_path.stem}.json').open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_predictions_json_by_name(output_dir: Path, item_name: str, item_stem: str, records: List[Dict[str, object]]) -> None:
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'image': item_name,
        'predictions': records,
    }
    with (pred_dir / f'{item_stem}.json').open('w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def initialize_model(args, config_path: Path, resume_path: Path):
    use_onnx = is_onnx_model(resume_path)
    use_torch_checkpoint = is_torch_checkpoint_model(resume_path)
    device: Optional[torch.device] = None

    if use_onnx:
        model = OnnxInferenceModel(
            resume_path,
            args.device,
            args.inference_type,
            mask_resize_origin=args.mask_resize_origin,
            mask_resize_mode=args.mask_resize_mode,
        )
        transform = build_onnx_transform_with_normalize(
            model.image_size,
            normalize=infer_onnx_normalize_from_model_path(resume_path),
        )
    elif use_torch_checkpoint:
        if (args.device or '').lower() == 'tensorrt':
            raise ValueError('TensorRT inference is only supported when --resume points to an ONNX model.')
        cfg = InferenceConfig(str(config_path), resume=str(resume_path))
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        if 'DINOv3STAs' in cfg.yaml_cfg:
            cfg.yaml_cfg['DINOv3STAs']['weights_path'] = None

        state_dict = load_checkpoint_state(resume_path)
        device = resolve_device(args.device)
        model = InferenceModel(cfg, state_dict, device, mask_resize_origin=args.mask_resize_origin)
        image_size = cfg.yaml_cfg['eval_spatial_size']
        normalize = bool(cfg.yaml_cfg.get('DINOv3STAs', False))
        transform = build_transform(image_size, normalize)
    else:
        raise ValueError(
            f'Unsupported model file extension: {resume_path.suffix or "<none>"}. '
            'Supported extensions are .onnx, .pt, and .pth.'
        )

    return model, transform, use_onnx, device


def prepare_runtime_settings(args) -> Dict[str, object]:
    return {
        'object_score_threshold': args.object_score_threshold if args.object_score_threshold is not None else args.score_threshold,
        'attribute_score_threshold': args.attribute_score_threshold if args.attribute_score_threshold is not None else args.score_threshold,
        'keypoint_threshold': args.keypoint_threshold if args.keypoint_threshold is not None else args.score_threshold,
        'tracking_iou_threshold': args.tracking_iou_threshold,
        'tracking_max_age': args.tracking_max_age,
        'tracking_min_score': args.tracking_min_score,
        'tracking_center_gate': args.tracking_center_gate,
        'disable_render_classids': set(args.disable_render_classids),
        'enable_bone_bbox_drawing_mode': args.enable_bone_bbox_drawing_mode,
        'enable_tracking': not args.disable_tracking,
        'enable_trackid_overlay': not args.disable_trackid_overlay,
        'enable_head_distance_measurement': not args.disable_head_distance_measurement,
        'keypoint_drawing_mode': args.keypoint_drawing_mode,
        'enable_bone_drawing_mode': args.enable_bone_drawing_mode,
        'disable_generation_identification_mode': args.disable_generation_identification_mode,
        'disable_gender_identification_mode': args.disable_gender_identification_mode,
        'disable_left_and_right_hand_identification_mode': args.disable_left_and_right_hand_identification_mode,
        'disable_left_and_right_label': args.disable_left_and_right_label,
        'disable_headpose_identification_mode': args.disable_headpose_identification_mode,
        'enable_face_mosaic': args.enable_face_mosaic,
    }


def run_model_inference(
    image: np.ndarray,
    model,
    transform: Callable[[np.ndarray], torch.Tensor],
    use_onnx: bool,
    device: Optional[torch.device],
    args,
    runtime_settings: Dict[str, object],
) -> Tuple[Dict[str, torch.Tensor], List[Box], float]:
    orig_h, orig_w = image.shape[:2]
    orig_target_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.float32)
    image_tensor = transform(image).unsqueeze(0)
    if not use_onnx and device is not None:
        image_tensor = image_tensor.to(device)

    fallback_inference_start_time = time.perf_counter()
    results = model(
        image_tensor,
        orig_target_sizes,
        return_masks=args.enable_masks,
        return_contours=args.enable_contours,
    )
    fallback_inference_time = time.perf_counter() - fallback_inference_start_time
    inference_time = float(getattr(model, 'last_inference_time', fallback_inference_time))
    result = results[0]

    boxes = build_result_boxes(
        result=result,
        image_width=orig_w,
        image_height=orig_h,
        object_score_threshold=runtime_settings['object_score_threshold'],
        attribute_score_threshold=runtime_settings['attribute_score_threshold'],
        keypoint_threshold=runtime_settings['keypoint_threshold'],
        disable_generation_identification_mode=runtime_settings['disable_generation_identification_mode'],
        disable_gender_identification_mode=runtime_settings['disable_gender_identification_mode'],
        disable_left_and_right_hand_identification_mode=runtime_settings['disable_left_and_right_hand_identification_mode'],
        disable_headpose_identification_mode=runtime_settings['disable_headpose_identification_mode'],
        enable_bone_drawing_mode=runtime_settings['enable_bone_drawing_mode'],
    )

    body_source_indices = [box.source_idx for box in boxes if box.classid == BODY_CLASS_ID and box.source_idx >= 0]
    if use_onnx and args.enable_masks:
        raw_masks = result.pop('_onnx_masks', None)
        if raw_masks is not None:
            result['masks'] = model._resize_selected_mask_map(
                raw_masks,
                orig_target_sizes[0],
                body_source_indices,
            )
    if use_onnx and args.enable_contours:
        raw_contours = result.pop('_onnx_contours', None)
        if raw_contours is not None:
            result['contours'] = model._resize_selected_mask_map(
                raw_contours,
                orig_target_sizes[0],
                body_source_indices,
            )

    return result, boxes, inference_time


def apply_tracking_to_boxes(
    boxes: List[Box],
    enable_tracking: bool,
    tracker: SimpleSortTracker,
    track_color_cache: Dict[int, np.ndarray],
    tracking_enabled_prev: bool,
) -> bool:
    body_boxes = [box for box in boxes if box.classid == BODY_CLASS_ID]
    if enable_tracking:
        if not tracking_enabled_prev:
            tracker.reset()
            track_color_cache.clear()
        tracker.update(body_boxes)
        for box in body_boxes:
            if box.track_id > 0 and box.track_id not in track_color_cache:
                track_color_cache[box.track_id] = np.array(make_instance_color(box.track_id - 1), dtype=np.uint8)
        active_track_ids = {track['id'] for track in tracker.tracks}
        stale_ids = [track_id for track_id in track_color_cache.keys() if track_id not in active_track_ids]
        for track_id in stale_ids:
            track_color_cache.pop(track_id, None)
    else:
        if tracking_enabled_prev:
            tracker.reset()
            track_color_cache.clear()
        for box in boxes:
            box.track_id = -1
    return enable_tracking


def render_frame(
    image: np.ndarray,
    result: Dict[str, torch.Tensor],
    boxes: List[Box],
    args,
    runtime_settings: Dict[str, object],
    track_color_cache: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    keypoint_mask_instance_map = None
    keypoint_instance_quality_map = None
    if (
        args.enable_masks
        and runtime_settings['enable_bone_drawing_mode']
        and result.get('masks') is not None
    ):
        keypoint_mask_instance_map, keypoint_instance_quality_map = build_keypoint_mask_assignment_context(
            boxes=boxes,
            result=result,
            args=args,
            mask_threshold=args.mask_threshold,
        )

    rendered = overlay_body_masks(
        image=image.copy(),
        result=result,
        boxes=boxes,
        args=args,
        mask_threshold=args.mask_threshold,
        mask_alpha=args.mask_alpha,
        disable_render_classids=runtime_settings['disable_render_classids'],
        track_color_cache=track_color_cache,
    )
    rendered = overlay_body_contours(
        image=rendered,
        result=result,
        boxes=boxes,
        contour_threshold=args.mask_threshold,
        disable_render_classids=runtime_settings['disable_render_classids'],
        track_color_cache=track_color_cache,
    )
    return draw_detections(
        image=rendered,
        boxes=boxes,
        disable_render_classids=runtime_settings['disable_render_classids'],
        enable_bone_bbox_drawing_mode=runtime_settings['enable_bone_bbox_drawing_mode'],
        keypoint_drawing_mode=runtime_settings['keypoint_drawing_mode'],
        enable_bone_drawing_mode=runtime_settings['enable_bone_drawing_mode'],
        enable_face_mosaic=runtime_settings['enable_face_mosaic'],
        disable_generation_identification_mode=runtime_settings['disable_generation_identification_mode'],
        disable_gender_identification_mode=runtime_settings['disable_gender_identification_mode'],
        disable_left_and_right_hand_identification_mode=runtime_settings['disable_left_and_right_hand_identification_mode'],
        disable_left_and_right_label=runtime_settings['disable_left_and_right_label'],
        disable_headpose_identification_mode=runtime_settings['disable_headpose_identification_mode'],
        bounding_box_line_width=args.bounding_box_line_width,
        keypoint_dot_radius=args.keypoint_dot_radius,
        enable_head_distance_measurement=runtime_settings['enable_head_distance_measurement'],
        camera_horizontal_fov=args.camera_horizontal_fov,
        enable_trackid_overlay=runtime_settings['enable_trackid_overlay'],
        track_color_cache=track_color_cache,
        keypoint_mask_instance_map=keypoint_mask_instance_map,
        keypoint_instance_quality_map=keypoint_instance_quality_map,
    )


def render_frame_from_input(
    image: np.ndarray,
    model,
    transform: Callable[[np.ndarray], torch.Tensor],
    use_onnx: bool,
    device: Optional[torch.device],
    args,
    runtime_settings: Dict[str, object],
    tracker: Optional[SimpleSortTracker] = None,
    track_color_cache: Optional[Dict[int, np.ndarray]] = None,
    tracking_enabled_prev: bool = False,
) -> Tuple[np.ndarray, Dict[str, torch.Tensor], List[Box], float, float, bool]:
    start_time = time.perf_counter()
    result, boxes, inference_time = run_model_inference(
        image=image,
        model=model,
        transform=transform,
        use_onnx=use_onnx,
        device=device,
        args=args,
        runtime_settings=runtime_settings,
    )
    total_time = time.perf_counter() - start_time

    if tracker is not None and track_color_cache is not None:
        tracking_enabled_prev = apply_tracking_to_boxes(
            boxes=boxes,
            enable_tracking=runtime_settings['enable_tracking'],
            tracker=tracker,
            track_color_cache=track_color_cache,
            tracking_enabled_prev=tracking_enabled_prev,
        )

    rendered = render_frame(
        image=image,
        result=result,
        boxes=boxes,
        args=args,
        runtime_settings=runtime_settings,
        track_color_cache=track_color_cache,
    )
    return rendered, result, boxes, inference_time, total_time, tracking_enabled_prev


def save_stream_predictions(output_dir: Path, frame_index: int, records: List[Dict[str, object]]) -> None:
    frame_name = f'{frame_index:08d}.png'
    frame_stem = f'{frame_index:08d}'
    save_predictions_json_by_name(output_dir, frame_name, frame_stem, records)


def is_parsable_to_int(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def create_video_writer(output_dir: Path, frame_width: int, frame_height: int, fps: float) -> cv2.VideoWriter:
    safe_fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    return cv2.VideoWriter(
        filename=str(output_dir / 'output.mp4'),
        fourcc=fourcc,
        fps=safe_fps,
        frameSize=(frame_width, frame_height),
    )


def process_images(args) -> None:
    config_path = Path(args.config)
    resume_path = Path(args.resume)
    images_dir = Path(args.images_dir) if args.images_dir is not None else None
    output_dir = Path(args.output_dir)
    video = args.video
    use_onnx = is_onnx_model(resume_path)
    use_torch_checkpoint = is_torch_checkpoint_model(resume_path)

    if not resume_path.exists():
        raise FileNotFoundError(f'Model file not found: {resume_path}')
    if use_torch_checkpoint and not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    if images_dir is not None:
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f'Image directory not found: {images_dir}')
        image_paths = list_image_paths(images_dir)
        if not image_paths:
            raise FileNotFoundError(f'No image files found in {images_dir}')
    else:
        image_paths = None

    output_dir.mkdir(parents=True, exist_ok=True)

    model, transform, use_onnx, device = initialize_model(args, config_path, resume_path)
    runtime_settings = prepare_runtime_settings(args)

    print(f'Using model: {resume_path}')
    if use_onnx:
        print(f'ONNX providers: {model.providers}')
    else:
        print(f'Device: {device}')
    print(f'Output directory: {output_dir}')
    print(f'Mask resize origin: {args.mask_resize_origin}')
    print(f'Enable masks: {args.enable_masks}')
    print(f'Enable contours: {args.enable_contours}')
    if args.mask_bilateral_d > 1 and args.mask_bilateral_sigma_color > 0.0 and args.mask_bilateral_sigma_space > 0.0:
        print(
            'Body mask bilateral filter: '
            f'd={args.mask_bilateral_d}, '
            f'sigma_color={args.mask_bilateral_sigma_color}, '
            f'sigma_space={args.mask_bilateral_sigma_space}'
        )
    if image_paths is not None:
        tracker = SimpleSortTracker(
            iou_threshold=runtime_settings['tracking_iou_threshold'],
            max_age=runtime_settings['tracking_max_age'],
            min_score=runtime_settings['tracking_min_score'],
            center_gate=runtime_settings['tracking_center_gate'],
        )
        track_color_cache: Dict[int, np.ndarray] = {}
        tracking_enabled_prev = runtime_settings['enable_tracking']
        print(f'Processing {len(image_paths)} images from {images_dir}')
        for image_path in tqdm(
            image_paths,
            desc='Processing images',
            dynamic_ncols=True,
            unit='image',
        ):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Failed to read image: {image_path}')
            rendered, result, boxes, _, _, tracking_enabled_prev = render_frame_from_input(
                image=image,
                model=model,
                transform=transform,
                use_onnx=use_onnx,
                device=device,
                args=args,
                runtime_settings=runtime_settings,
                tracker=tracker,
                track_color_cache=track_color_cache,
                tracking_enabled_prev=tracking_enabled_prev,
            )
            cv2.imwrite(str(output_dir / image_path.name), rendered)

            if args.save_raw_predictions:
                records = prepare_prediction_payload(
                    boxes=boxes,
                    result=result,
                    mask_threshold=args.mask_threshold,
                    enable_masks=args.enable_masks,
                    enable_contours=args.enable_contours,
                    mask_bilateral_d=args.mask_bilateral_d,
                    mask_bilateral_sigma_color=args.mask_bilateral_sigma_color,
                    mask_bilateral_sigma_space=args.mask_bilateral_sigma_space,
                )
                save_predictions_json(output_dir, image_path, records)
        return

    print(f'Processing video source: {video}')
    cap = cv2.VideoCapture(int(video) if is_parsable_to_int(video) else video)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video source: {video}')

    tracker = SimpleSortTracker(
        iou_threshold=runtime_settings['tracking_iou_threshold'],
        max_age=runtime_settings['tracking_max_age'],
        min_score=runtime_settings['tracking_min_score'],
        center_gate=runtime_settings['tracking_center_gate'],
    )
    track_color_cache: Dict[int, np.ndarray] = {}
    tracking_enabled_prev = runtime_settings['enable_tracking']
    video_writer = None
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            image = frame
            rendered, result, boxes, inference_time, total_time, tracking_enabled_prev = render_frame_from_input(
                image=image,
                model=model,
                transform=transform,
                use_onnx=use_onnx,
                device=device,
                args=args,
                runtime_settings=runtime_settings,
                tracker=tracker,
                track_color_cache=track_color_cache,
                tracking_enabled_prev=tracking_enabled_prev,
            )

            debug_image = rendered.copy()
            cv2.putText(debug_image, f'infer: {inference_time * 1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'infer: {inference_time * 1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_image, f'total: {total_time * 1000:.2f} ms', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'total: {total_time * 1000:.2f} ms', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            if video_writer is None and not args.disable_video_writer:
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_writer = create_video_writer(output_dir, debug_image.shape[1], debug_image.shape[0], fps)

            if video_writer is not None:
                video_writer.write(debug_image)

            if args.save_raw_predictions:
                records = prepare_prediction_payload(
                    boxes=boxes,
                    result=result,
                    mask_threshold=args.mask_threshold,
                    enable_masks=args.enable_masks,
                    enable_contours=args.enable_contours,
                    mask_bilateral_d=args.mask_bilateral_d,
                    mask_bilateral_sigma_color=args.mask_bilateral_sigma_color,
                    mask_bilateral_sigma_space=args.mask_bilateral_sigma_space,
                )
                save_stream_predictions(output_dir, frame_index, records)

            cv2.imshow('DEIMv2 WholeBody49', debug_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('\x1b'):
                break
            if key == ord('b'):
                runtime_settings['enable_bone_drawing_mode'] = not runtime_settings['enable_bone_drawing_mode']
            elif key == ord('n'):
                runtime_settings['disable_generation_identification_mode'] = not runtime_settings['disable_generation_identification_mode']
            elif key == ord('g'):
                runtime_settings['disable_gender_identification_mode'] = not runtime_settings['disable_gender_identification_mode']
            elif key == ord('p'):
                runtime_settings['disable_headpose_identification_mode'] = not runtime_settings['disable_headpose_identification_mode']
            elif key == ord('h'):
                runtime_settings['disable_left_and_right_hand_identification_mode'] = not runtime_settings['disable_left_and_right_hand_identification_mode']
            elif key == ord('k'):
                if runtime_settings['keypoint_drawing_mode'] == 'dot':
                    runtime_settings['keypoint_drawing_mode'] = 'box'
                elif runtime_settings['keypoint_drawing_mode'] == 'box':
                    runtime_settings['keypoint_drawing_mode'] = 'both'
                else:
                    runtime_settings['keypoint_drawing_mode'] = 'dot'
            elif key == ord('r'):
                runtime_settings['enable_tracking'] = not runtime_settings['enable_tracking']
                if runtime_settings['enable_tracking'] and not runtime_settings['enable_trackid_overlay']:
                    runtime_settings['enable_trackid_overlay'] = True
            elif key == ord('t'):
                runtime_settings['enable_trackid_overlay'] = not runtime_settings['enable_trackid_overlay']
                if not runtime_settings['enable_tracking']:
                    runtime_settings['enable_trackid_overlay'] = False
            elif key == ord('m'):
                runtime_settings['enable_head_distance_measurement'] = not runtime_settings['enable_head_distance_measurement']
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


def parse_args():
    def check_positive(value: str) -> int:
        ivalue = int(value)
        if ivalue < 2:
            raise argparse.ArgumentTypeError(f'Invalid value: {ivalue}. Please specify an integer of 2 or greater.')
        return ivalue

    def check_positive_radius(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f'Invalid value: {ivalue}. Please specify an integer of 1 or greater.')
        return ivalue

    def check_alpha(value: str) -> int:
        ivalue = int(value)
        if not 0 <= ivalue <= 255:
            raise argparse.ArgumentTypeError(f'Invalid value: {ivalue}. Please specify an integer in the range 0-255.')
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('-r', '--resume', type=str, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--images_dir', type=str)
    input_group.add_argument('-v', '--video', type=str, help='Video file path or camera index.')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('--inference_type', type=str, choices=['fp16', 'int8'], default='fp16')
    parser.add_argument('--disable_video_writer', action='store_true')
    parser.add_argument('--disable_waitKey', action='store_true')
    parser.add_argument('--score_threshold', type=float, default=0.50)
    parser.add_argument('--object_score_threshold', '--object_socre_threshold', dest='object_score_threshold', type=float, default=None)
    parser.add_argument('--attribute_score_threshold', '--attribute_socre_threshold', dest='attribute_score_threshold', type=float, default=0.75)
    parser.add_argument('--keypoint_threshold', type=float, default=None)
    parser.add_argument('--mask_threshold', type=float, default=0.4)
    parser.add_argument('--mask_alpha', type=check_alpha, default=160)
    parser.add_argument('--mask_bilateral_d', type=int, default=0)
    parser.add_argument('--mask_bilateral_sigma_color', type=float, default=1.0)
    parser.add_argument('--mask_bilateral_sigma_space', type=float, default=3.0)
    parser.add_argument('--mask_resize_origin', type=str, choices=['topleft', 'center'], default='topleft')
    parser.add_argument(
        '--mask_resize_mode',
        type=str,
        choices=['nearest', 'nearest-exact', 'bilinear', 'bicubic', 'area'],
        default='bilinear',
    )
    parser.add_argument('--enable-masks', action='store_true')
    parser.add_argument('--enable-contours', action='store_true')
    parser.add_argument('--keypoint_drawing_mode', type=str, choices=['dot', 'box', 'both'], default='dot')
    parser.add_argument('--keypoint_dot_radius', type=check_positive_radius, default=2)
    parser.add_argument('--enable_bone_drawing_mode', action='store_true')
    parser.add_argument('--disable_generation_identification_mode', action='store_true')
    parser.add_argument('--disable_gender_identification_mode', action='store_true')
    parser.add_argument('--disable_left_and_right_hand_identification_mode', action='store_true')
    parser.add_argument('--disable_left_and_right_label', action='store_true')
    parser.add_argument('--disable_headpose_identification_mode', action='store_true')
    parser.add_argument('--disable_render_classids', type=int, nargs='*', default=[])
    parser.add_argument(
        '--enable_bone_bbox_drawing_mode',
        dest='enable_bone_bbox_drawing_mode',
        action='store_true',
    )
    parser.add_argument('--enable_face_mosaic', action='store_true')
    parser.add_argument('--disable_tracking', action='store_true')
    parser.add_argument('--disable_trackid_overlay', action='store_true')
    parser.add_argument('--tracking_iou_threshold', type=float, default=0.20)
    parser.add_argument('--tracking_max_age', type=int, default=45)
    parser.add_argument('--tracking_min_score', type=float, default=0.45)
    parser.add_argument('--tracking_center_gate', type=float, default=0.25)
    parser.add_argument('--disable_head_distance_measurement', action='store_true')
    parser.add_argument('--bounding_box_line_width', type=check_positive, default=2)
    parser.add_argument('--camera_horizontal_fov', type=int, default=90)
    parser.add_argument('--save_raw_predictions', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    process_images(parse_args())
