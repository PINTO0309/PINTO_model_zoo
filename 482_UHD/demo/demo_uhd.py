#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper


def preprocess(img_bgr: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return chw[np.newaxis, ...]


def postprocess(detections: np.ndarray, orig_shape: Tuple[int, int], conf_thresh: float) -> List[Tuple[float, int, float, float, float, float]]:
    h, w = orig_shape
    out: List[Tuple[float, int, float, float, float, float]] = []
    for det in detections:
        if len(det) < 6:
            continue
        score, cls_id, cx, cy, bw, bh = det[:6]
        if score < conf_thresh:
            continue
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        # clamp to valid range to avoid NaN/inf impacting drawing
        x1 = max(0.0, min(x1, w))
        x2 = max(0.0, min(x2, w))
        y1 = max(0.0, min(y1, h))
        y2 = max(0.0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((float(score), int(cls_id), float(x1), float(y1), float(x2), float(y2)))
    return out


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[float, int, float, float, float, float]], color: Tuple[int, int, int]) -> np.ndarray:
    out = img_bgr.copy()
    for score, cls_id, x1, y1, x2, y2 in boxes:
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def softplus_np(x: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    # numerically stable softplus without truncating large positives (softplus(x) ~= x when x>>0)
    sp = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    if cap is not None:
        sp = np.minimum(sp, cap)
    return sp


def _is_decoded_shape(shape) -> bool:
    return bool(shape) and len(shape) >= 3 and shape[-1] == 6


def _parse_anchor_hint_from_path(onnx_path: str) -> Optional[int]:
    """Infer anchor count from filename pattern like '*_anc8_*'."""
    name = os.path.basename(onnx_path).lower()
    m = re.search(r"anc(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _infer_anchor_count_from_channels(c: int) -> Optional[int]:
    """Heuristic to guess anchor count from channel size."""
    candidates = [3, 4, 5, 6, 8, 9, 12, 16]
    for na in candidates:
        if c % na == 0 and (c // na) >= 6:
            return na
    return None


def load_anchors_from_onnx(onnx_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Extract anchors/wh_scale (if present) from the ONNX initializers."""
    anchors = None
    wh_scale = None
    has_quality = False
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        print(f"[WARN] Failed to load ONNX for anchor lookup: {exc}")
        return anchors, wh_scale, has_quality

    for init in model.graph.initializer:
        name_l = init.name.lower()
        arr = numpy_helper.to_array(init)
        if arr.ndim == 2 and arr.shape[1] == 2:
            if "anchor" in name_l and anchors is None:
                anchors = arr.astype(np.float32)
                continue
            if "wh_scale" in name_l and wh_scale is None:
                wh_scale = arr.astype(np.float32)
                has_quality = True
                continue
            if anchors is None and arr.shape[0] <= 16:
                anchors = arr.astype(np.float32)
        if "quality" in name_l:
            has_quality = True
    return anchors, wh_scale, has_quality


def decode_ultratinyod_raw(
    raw_out: np.ndarray,
    anchors: np.ndarray,
    conf_thresh: float,
    has_quality: bool = False,
    wh_scale: Optional[np.ndarray] = None,
    topk: int = 100,
) -> np.ndarray:
    """
    Decode raw UltraTinyOD output [B, C, H, W] -> [N, 6] (score, cls, cx, cy, bw, bh), normalized coords.
    """
    if raw_out.ndim == 3:
        raw_out = raw_out[None, ...]
    if raw_out.ndim != 4:
        raise ValueError(f"Unexpected raw output ndim={raw_out.ndim}; expected 4D map.")
    b, c, h, w = raw_out.shape
    anchors = np.asarray(anchors, dtype=np.float32)
    na = anchors.shape[0]
    if c % na != 0:
        raise ValueError(f"Channel/anchor mismatch: C={c}, anchors={na}")
    per_anchor = c // na
    quality_extra = 1 if has_quality and per_anchor >= 6 else 0

    pred = raw_out.reshape(b, na, per_anchor, h, w)
    tx = pred[:, :, 0]
    ty = pred[:, :, 1]
    tw = pred[:, :, 2]
    th = pred[:, :, 3]
    obj = pred[:, :, 4]
    quality = pred[:, :, 5] if quality_extra else None
    cls_logits = pred[:, :, (5 + quality_extra) :]

    obj_sig = sigmoid_np(obj)
    cls_sig = sigmoid_np(cls_logits)
    score_base = obj_sig
    if quality is not None:
        score_base = score_base * sigmoid_np(quality)
    scores = score_base[..., None] * cls_sig  # [B, A, H, W, C]

    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    gx = gx.reshape(1, 1, h, w)
    gy = gy.reshape(1, 1, h, w)

    anchor_use = anchors
    if wh_scale is not None and wh_scale.shape == anchors.shape:
        anchor_use = anchor_use * wh_scale
    pw = anchor_use[:, 0].reshape(1, na, 1, 1)
    ph = anchor_use[:, 1].reshape(1, na, 1, 1)

    cx = (sigmoid_np(tx) + gx) / float(w)
    cy = (sigmoid_np(ty) + gy) / float(h)
    bw = pw * softplus_np(tw)  # no cap; allow large scaling for tiny anchors
    bh = ph * softplus_np(th)

    best_cls = scores.argmax(axis=-1)  # [B, A, H, W]
    best_scores = scores.max(axis=-1)

    cx_flat = cx.reshape(b, -1)
    cy_flat = cy.reshape(b, -1)
    bw_flat = bw.reshape(b, -1)
    bh_flat = bh.reshape(b, -1)
    scores_flat = best_scores.reshape(b, -1)
    cls_flat = best_cls.reshape(b, -1)

    k = min(int(topk), scores_flat.shape[1])
    top_idx = np.argsort(-scores_flat, axis=1)[:, :k]

    def _gather(t: np.ndarray) -> np.ndarray:
        return np.take_along_axis(t, top_idx, axis=1)

    top_scores = _gather(scores_flat)
    top_cls = _gather(cls_flat)
    top_cx = _gather(cx_flat)
    top_cy = _gather(cy_flat)
    top_bw = _gather(bw_flat)
    top_bh = _gather(bh_flat)

    dets = []
    for i in range(b):
        mask = (top_scores[i] > 0.0)
        if not np.any(mask):
            continue
        stacked = np.stack(
            [
                top_scores[i][mask],
                top_cls[i][mask].astype(np.float32),
                top_cx[i][mask],
                top_cy[i][mask],
                top_bw[i][mask],
                top_bh[i][mask],
            ],
            axis=-1,
        )
        finite_mask = np.all(np.isfinite(stacked), axis=-1)
        stacked = stacked[finite_mask]
        dets.append(stacked)
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    return dets[0]


def load_session(onnx_path: str, img_size: Tuple[int, int]):
    """Load ONNX session and infer whether outputs already include post-process."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    outputs_info = session.get_outputs()
    anchor_hint = _parse_anchor_hint_from_path(onnx_path)

    decoded_output = None
    for o in outputs_info:
        if _is_decoded_shape(o.shape):
            decoded_output = o.name
            break

    anchors = None
    wh_scale = None
    has_quality = False
    raw_channels = None
    raw_output = None

    if decoded_output is None:
        # Probe with a dummy forward to inspect actual shapes and capture anchors/wh_scale outputs if present.
        _, c_in, h_in, w_in = input_info.shape
        h_probe = int(img_size[0] if h_in in (None, "None") else h_in)
        w_probe = int(img_size[1] if w_in in (None, "None") else w_in)
        dummy = np.zeros((1, int(c_in or 3), h_probe, w_probe), dtype=np.float32)
        outs = session.run(None, {input_info.name: dummy})
        for meta, val in zip(outputs_info, outs):
            if val.ndim == 4 and raw_output is None:
                raw_output = meta.name
                raw_channels = val.shape[1] if val.ndim == 4 else val.shape[-1]
            elif val.ndim == 2 and val.shape[1] == 2:
                name_l = meta.name.lower()
                if anchors is None and ("anchor" in name_l or raw_output is None):
                    anchors = val.astype(np.float32)
                if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                    wh_scale = val.astype(np.float32)
                    has_quality = True
        if raw_output is None and outputs_info:
            raw_output = outputs_info[0].name
            raw_shape = outs[0].shape if outs else outputs_info[0].shape
            raw_channels = raw_shape[1] if isinstance(raw_shape, tuple) and len(raw_shape) >= 2 else None
        if anchors is None or wh_scale is None:
            anchors_f, wh_scale_f, has_quality_f = load_anchors_from_onnx(onnx_path)
            anchors = anchors if anchors is not None else anchors_f
            wh_scale = wh_scale if wh_scale is not None else wh_scale_f
            has_quality = has_quality or has_quality_f
        if anchors is None and anchor_hint:
            anchors = np.stack(
                [
                    np.linspace(0.08, 0.32, anchor_hint, dtype=np.float32),
                    np.linspace(0.10, 0.40, anchor_hint, dtype=np.float32),
                ],
                axis=1,
            )
            print(f"[INFO] Using anchors inferred from filename (anc{anchor_hint}).")
        if anchors is None:
            print("[WARN] Could not find anchors in ONNX; raw decode may fail.")
        decoded = False
        output_shape = outs[0].shape if outs else outputs_info[0].shape
    else:
        decoded = True
        output_shape = next(o.shape for o in outputs_info if o.name == decoded_output)

    kind = "decoded output" if decoded else "raw output + demo post-process"
    print(f"[INFO] Detected {kind} (output shape: {output_shape})")
    return session, {
        "decoded": decoded,
        "anchors": anchors,
        "wh_scale": wh_scale,
        "has_quality": has_quality or wh_scale is not None,
        "raw_channels": raw_channels if not decoded else None,
        "anchor_hint": anchor_hint,
        "input_name": input_info.name,
        "decoded_output": decoded_output,
        "raw_output": raw_output,
    }


def run_and_decode(
    session: ort.InferenceSession,
    session_info: dict,
    inp: np.ndarray,
    conf_thresh: float,
) -> np.ndarray:
    if session_info.get("decoded", False):
        dets = session.run([session_info["decoded_output"]], {session_info["input_name"]: inp})[0]
        return dets[0] if dets.ndim >= 3 else dets

    # Raw path: prefer cached anchors/wh_scale from ONNX outputs
    anchors = session_info.get("anchors")
    wh_scale = session_info.get("wh_scale")
    outputs = [session_info.get("raw_output") or session.get_outputs()[0].name]
    if anchors is None or wh_scale is None:
        # If anchors/wh_scale were not cached, fetch them from ONNX outputs
        outputs = [o.name for o in session.get_outputs()]
    run_outs = session.run(outputs, {session_info["input_name"]: inp})

    # Identify raw / anchors / wh_scale in the returned list
    raw = None
    for name, val in zip(outputs, run_outs):
        name_l = name.lower()
        if raw is None and val.ndim == 4:
            raw = val
        elif val.ndim == 2 and val.shape[1] == 2:
            if anchors is None and ("anchor" in name_l or True):
                anchors = val.astype(np.float32)
            if wh_scale is None and ("wh_scale" in name_l or "scale" in name_l):
                wh_scale = val.astype(np.float32)
                session_info["has_quality"] = True

    if raw is None:
        raw = run_outs[0]

    def _build_fallback_anchors(na: int) -> np.ndarray:
        return np.stack(
            [
                np.linspace(0.08, 0.32, na, dtype=np.float32),
                np.linspace(0.10, 0.40, na, dtype=np.float32),
            ],
            axis=1,
        )

    if anchors is None:
        c = raw.shape[1] if raw.ndim == 4 else raw.shape[-1]
        na_hint = session_info.get("anchor_hint")
        na_guess = _infer_anchor_count_from_channels(int(c))
        na = na_hint if na_hint is not None else (na_guess if na_guess is not None else 3)
        anchors = _build_fallback_anchors(na)
        print(f"[WARN] Anchors not found in ONNX; using fallback anchors (A={na}).")

    session_info["anchors"] = anchors
    if wh_scale is not None:
        session_info["wh_scale"] = wh_scale

    return decode_ultratinyod_raw(
        raw,
        anchors=anchors,
        conf_thresh=0.0,  # avoid double-thresholding; postprocess will apply user conf
        has_quality=session_info.get("has_quality", False),
        wh_scale=wh_scale,
    )


def run_images(
    session: ort.InferenceSession,
    session_info: dict,
    img_dir: Path,
    out_dir: Path,
    img_size: Tuple[int, int],
    conf_thresh: float,
    actual_size: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"No images found under {img_dir}")
        return

    for img_path in images:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable file: {img_path}")
            continue
        h, w = img_bgr.shape[:2]
        target_h, target_w = img_size if actual_size else (h, w)
        inp = preprocess(img_bgr, img_size)
        dets = run_and_decode(session, session_info, inp, conf_thresh)
        boxes = postprocess(dets, (target_h, target_w), conf_thresh)
        if not boxes and dets.size > 0 and conf_thresh > 0.05:
            fallback_thresh = max(0.05, conf_thresh * 0.5)
            boxes = postprocess(dets, (target_h, target_w), fallback_thresh)
        base = cv2.resize(img_bgr, (target_w, target_h)) if actual_size else img_bgr
        vis_out = draw_boxes(base, boxes, (0, 0, 255))
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis_out)
        print(f"Saved {save_path} (detections: {len(boxes)})")


def run_camera(
    session: ort.InferenceSession,
    session_info: dict,
    camera_id: int,
    img_size: Tuple[int, int],
    conf_thresh: float,
    record_path: Optional[Path] = None,
    actual_size: bool = False,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id {camera_id}")

    writer = None
    last_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        target_h, target_w = img_size if actual_size else (h, w)
        inp = preprocess(frame, img_size)
        dets = run_and_decode(session, session_info, inp, conf_thresh)
        boxes = postprocess(dets, (target_h, target_w), conf_thresh)
        if not boxes and dets.size > 0 and conf_thresh > 0.05:
            fallback_thresh = max(0.05, conf_thresh * 0.5)
            boxes = postprocess(dets, (target_h, target_w), fallback_thresh)
        base = cv2.resize(frame, (target_w, target_h)) if actual_size else frame
        vis = draw_boxes(base, boxes, (255, 0, 0))

        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        last_time = t1
        if not actual_size:
            label = f"{ms:.2f} ms"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 6
            x, y = 10, 30
            cv2.rectangle(
                vis,
                (x - pad, y - th - pad),
                (x + tw + pad, y + baseline + pad),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                vis,
                label,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

        vis_out = cv2.resize(vis, img_size) if actual_size else vis

        if record_path:
            if writer is None:
                h, w = vis_out.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                if fps <= 0:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                record_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(record_path), fourcc, fps, (w, h))
            writer.write(vis_out)

        cv2.imshow("UHD ONNX (press q to quit)", vis_out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if writer is not None:
        writer.release()
        print(f"Saved recording to {record_path}")
    cap.release()
    cv2.destroyAllWindows()


def parse_size(arg: str) -> Tuple[int, int]:
    s = str(arg).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(float(h)), int(float(w))
    v = int(float(s))
    return v, v


def build_args():
    parser = argparse.ArgumentParser(description="UltraTinyOD ONNX demo (CPU).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--images", type=str, help="Directory with images to run batch inference.")
    mode.add_argument("--camera", type=int, help="USB camera id for realtime inference.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (CPU).")
    parser.add_argument("--output", type=str, default="demo_output", help="Output directory for image mode.")
    parser.add_argument("--img-size", type=str, default="64x64", help="Input size HxW, e.g., 64x64.")
    parser.add_argument("--conf-thresh", type=float, default=0.90, help="Confidence threshold.")
    parser.add_argument(
        "--record",
        type=str,
        default="camera_record.mp4",
        help="MP4 path for automatic recording when --camera is used.",
    )
    parser.add_argument(
        "--actual-size",
        action="store_true",
        help="Display and recording use the model input resolution instead of the original frame size.",
    )
    return parser


def main():
    args = build_args().parse_args()
    img_size = parse_size(args.img_size)
    session, session_info = load_session(args.onnx, img_size)

    if args.images:
        run_images(
            session,
            session_info,
            Path(args.images),
            Path(args.output),
            img_size,
            args.conf_thresh,
            args.actual_size,
        )
    else:
        record_path = Path(args.record) if args.record else None
        run_camera(session, session_info, int(args.camera), img_size, args.conf_thresh, record_path, args.actual_size)


if __name__ == "__main__":
    main()
