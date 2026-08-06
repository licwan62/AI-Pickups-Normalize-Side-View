from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from .geometry import Bounds


class VehicleDetector(ABC):
    """Extension point for a future YOLO/SAM automatic detector."""

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        expected_aspect: float | None = None,
    ) -> Bounds:
        """Return vehicle bounds in source-image pixels or raise on failure."""
        raise NotImplementedError


class OpenCVVehicleDetector(VehicleDetector):
    """Foreground-based detector intended for clean side-profile product images."""

    def __init__(self, max_working_size: int = 1400, iterations: int = 5):
        self.max_working_size = max_working_size
        self.iterations = iterations
        self.last_attempt_bounds: Bounds | None = None

    def _detect_complex_scene(
        self,
        working: np.ndarray,
        original_width: int,
        original_height: int,
        scale: float,
        expected_aspect: float,
    ) -> Bounds:
        """Recover a vehicle box in a natural scene using its known proportions."""
        height, width = working.shape[:2]
        if expected_aspect <= 1.2:
            raise RuntimeError("Expected vehicle aspect ratio is not plausible")

        bgr = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)
        mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(2, round(min(width, height) * 0.02))
        mask[:round(height * 0.25), :] = cv2.GC_BGD
        mask[-round(height * 0.12):, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Side-profile source photos consistently put the body through the lower
        # centre of the frame. A small certain-foreground seed lets GrabCut learn
        # the paint/body colours without labelling every non-border colour as a
        # vehicle (which is what joins trees, pavement, and utility poles).
        seed_top = round(height * 0.525)
        seed_bottom = max(seed_top + 1, round(height * 0.575))
        seed_left = round(width * 0.42)
        seed_right = max(seed_left + 1, round(width * 0.58))
        mask[seed_top:seed_bottom, seed_left:seed_right] = cv2.GC_FGD

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                bgr,
                mask,
                None,
                bg_model,
                fg_model,
                self.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as exc:
            raise RuntimeError("OpenCV guided foreground segmentation failed") from exc

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel_size = max(3, round(min(width, height) * 0.008))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground,
            8,
        )
        candidates: list[tuple[float, int]] = []
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            center_x, center_y = centroids[label]
            if component_width < width * 0.5 or component_height < height * 0.12:
                continue
            if not width * 0.25 <= center_x <= width * 0.75:
                continue
            if not height * 0.3 <= center_y <= height * 0.75:
                continue
            candidates.append((float(area), label))
        if not candidates:
            raise RuntimeError(
                "Automatic detection confidence is too low; use --manual for this image"
            )

        _, best_label = max(candidates)
        x, y, box_width, box_height, _ = (
            int(value) for value in stats[best_label]
        )
        component_aspect = box_width / max(box_height, 1)
        component_aspect_error = abs(component_aspect / expected_aspect - 1.0)

        # GrabCut tends to stop at painted panels and miss narrow chrome bumpers,
        # tow hardware, or mud flaps. Preserve a small horizontal safety margin,
        # but reduce it when the vehicle already fills most of the frame.
        padding_fraction = 0.01 if box_width >= width * 0.90 else 0.025
        horizontal_padding = round(width * padding_fraction)
        padded_left = max(0, x - horizontal_padding)
        padded_right = min(width, x + box_width + horizontal_padding)
        x = padded_left
        box_width = padded_right - padded_left

        # The component normally captures the long body but may omit either the
        # roof or tyres. Reconstruct that axis from the catalogued length/height
        # ratio, retaining 2% vertical breathing room for roof lights and tyres.
        crop_height = max(1, round((box_width / expected_aspect) * 1.02))
        if box_height < crop_height * 0.70:
            crop_top = y - round(crop_height * 0.03)
        else:
            crop_top = round(y + box_height / 2.0 - crop_height / 2.0)
        crop_top = min(max(0, crop_top), max(0, height - crop_height))
        crop_bottom = min(height, crop_top + crop_height)

        guided_aspect = box_width / max(crop_bottom - crop_top, 1)
        aspect_error = abs(guided_aspect / expected_aspect - 1.0)
        inverse_scale = 1.0 / scale
        bounds = Bounds(
            left=max(0, int(np.floor(x * inverse_scale))),
            right=min(
                original_width,
                int(np.ceil((x + box_width) * inverse_scale)),
            ),
            roof=max(0, int(np.floor(crop_top * inverse_scale))),
            ground=min(
                original_height,
                int(np.ceil(crop_bottom * inverse_scale)),
            ),
        )
        bounds.validate(original_width, original_height)
        self.last_attempt_bounds = bounds

        # A near-full-width component is valid when its raw proportions already
        # resemble the catalogued vehicle. This distinguishes a tightly framed
        # pickup from an unrelated full-canvas foreground component.
        full_width_is_plausible_vehicle = (
            box_width > width * 0.995
            and component_aspect_error <= 0.20
        )
        if (
            (box_width > width * 0.995 and not full_width_is_plausible_vehicle)
            or crop_bottom - crop_top > height * 0.75
            or aspect_error > 0.08
        ):
            raise RuntimeError(
                "Automatic detection confidence is too low; use --manual for this image"
            )

        return bounds

    @staticmethod
    def _wheel_contact_ground(
        grayscale: np.ndarray,
        edges: np.ndarray,
        roof: int,
        detected_ground: int,
    ) -> int | None:
        """Infer the tyre contact row from a plausible pair of circular wheels."""
        height, width = grayscale.shape
        blurred = cv2.GaussianBlur(grayscale, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=width * 0.2,
            param1=100,
            param2=35,
            minRadius=max(8, round(height * 0.04)),
            maxRadius=max(10, round(height * 0.25)),
        )
        if circles is None:
            return None

        box_height = detected_ground - roof
        candidates = [
            tuple(int(value) for value in circle)
            for circle in np.round(circles[0]).astype(int)
            if circle[1] >= roof + box_height * 0.45
        ]
        pairs: list[tuple[float, tuple[int, int, int], tuple[int, int, int]]] = []
        for index, first in enumerate(candidates):
            for second in candidates[index + 1:]:
                left, right = sorted((first, second), key=lambda circle: circle[0])
                separation = right[0] - left[0]
                largest_radius = max(left[2], right[2])
                if not width * 0.25 <= separation <= width * 0.85:
                    continue
                if abs(left[2] - right[2]) > largest_radius * 0.5:
                    continue
                if abs(left[1] - right[1]) > largest_radius * 0.4:
                    continue
                score = (
                    separation
                    - abs(left[2] - right[2]) * 2
                    - abs(left[1] - right[1]) * 2
                )
                pairs.append((score, left, right))
        if not pairs:
            return None

        _, left_wheel, right_wheel = max(pairs, key=lambda item: item[0])
        contact_rows: list[int] = []
        for center_x, center_y, radius in (left_wheel, right_wheel):
            zone_left = max(0, round(center_x - radius * 1.1))
            zone_right = min(width, round(center_x + radius * 1.1))
            # Hough can lock onto the bright rim instead of the outer tyre. Search
            # as far as two detected radii and use the last sustained dark tyre row.
            search_bottom = min(height, round(center_y + radius * 2.0))
            dark_counts = np.count_nonzero(
                grayscale[center_y:search_bottom, zone_left:zone_right] < 100,
                axis=1,
            )
            active_rows = np.flatnonzero(dark_counts >= max(4, round(radius * 0.2)))
            if not active_rows.size:
                edge_counts = np.count_nonzero(
                    edges[center_y:search_bottom, zone_left:zone_right],
                    axis=1,
                )
                active_rows = np.flatnonzero(
                    edge_counts >= max(4, round(radius * 0.08))
                )
            if active_rows.size:
                contact_rows.append(center_y + int(active_rows[-1]) + 1)
        if len(contact_rows) != 2:
            return None
        return min(height, round(float(np.median(contact_rows))))

    def detect(
        self,
        image: Image.Image,
        expected_aspect: float | None = None,
    ) -> Bounds:
        self.last_attempt_bounds = None
        rgb = np.asarray(image.convert("RGB"))
        original_height, original_width = rgb.shape[:2]
        scale = min(1.0, self.max_working_size / max(original_width, original_height))
        if scale < 1.0:
            working = cv2.resize(
                rgb,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = rgb

        height, width = working.shape[:2]
        if width < 40 or height < 40:
            raise RuntimeError("Image is too small for automatic vehicle detection")

        bgr = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)
        mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(2, round(min(width, height) * 0.02))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Border colour provides a strong prior for white/solid studio backgrounds.
        border_pixels = np.concatenate(
            (working[:border].reshape(-1, 3), working[-border:].reshape(-1, 3),
             working[:, :border].reshape(-1, 3), working[:, -border:].reshape(-1, 3)),
            axis=0,
        )
        background_colour = np.median(border_pixels, axis=0)
        colour_distance = np.linalg.norm(working.astype(np.float32) - background_colour, axis=2)
        mask[colour_distance > 32.0] = cv2.GC_PR_FGD

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                bgr,
                mask,
                None,
                bg_model,
                fg_model,
                self.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as exc:
            raise RuntimeError("OpenCV foreground segmentation failed") from exc

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel_size = max(3, round(min(width, height) * 0.008))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
        candidates: list[tuple[float, int]] = []
        image_area = width * height
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if area < image_area * 0.01 or component_width < width * 0.2:
                continue
            aspect = component_width / max(component_height, 1)
            center_x, center_y = centroids[label]
            centrality = 1.0 - min(
                0.8,
                abs(center_x / width - 0.5) + 0.5 * abs(center_y / height - 0.55),
            )
            wide_bonus = min(aspect, 4.0) / 2.0
            candidates.append((area * centrality * wide_bonus, label))

        if not candidates:
            raise RuntimeError("No plausible vehicle foreground was detected")
        _, best_label = max(candidates)
        selected = np.where(labels == best_label, 255, 0).astype(np.uint8)
        x, y, box_width, box_height = cv2.boundingRect(selected)
        initial_span_ratio = box_width / width

        grayscale = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grayscale, 50, 140)

        # A pair of wheels gives a stronger ground cue than the foreground mask:
        # it trims broad studio shadows and recovers dark tyres on black backgrounds.
        wheel_ground = self._wheel_contact_ground(
            grayscale,
            edges,
            roof=y,
            detected_ground=y + box_height,
        )
        if wheel_ground is not None and wheel_ground > y:
            box_height = wheel_ground - y

        # Shadows and gradient floors can be selected across the whole canvas.
        # Re-project stable vehicle edges through the body band to recover tight
        # bumper-to-bumper bounds without relying on foreground fill colour.
        body_band_bottom = y + max(1, round(box_height * 0.88))
        body_band = edges[y:body_band_bottom, :]
        column_counts = np.count_nonzero(body_band, axis=0)
        side_width = max(1, round(width * 0.05))
        side_edge_density = float(np.mean(np.concatenate(
            (edges[:, :side_width].reshape(-1), edges[:, -side_width:].reshape(-1))
        ) > 0))
        complex_full_width_scene = (
            initial_span_ratio > 0.9 and side_edge_density > 0.05
        )
        projection_fraction = 0.10 if complex_full_width_scene else 0.02
        minimum_column_pixels = max(
            4,
            round(body_band.shape[0] * projection_fraction),
        )
        active_columns = np.flatnonzero(column_counts >= minimum_column_pixels)
        if active_columns.size >= 2:
            if complex_full_width_scene and active_columns.size >= 20:
                padding = round(width * 0.02)
                projected_left = max(
                    0,
                    int(np.quantile(active_columns, 0.01)) - padding,
                )
                projected_right = min(
                    width,
                    int(np.quantile(active_columns, 0.99)) + padding + 1,
                )
            else:
                projected_left = int(active_columns[0])
                projected_right = int(active_columns[-1] + 1)
            if projected_right - projected_left >= width * 0.2:
                x = projected_left
                box_width = projected_right - projected_left

        # Place the crop's ground edge on the lowest tyre contact pixel. The two
        # side zones cover the usual rear/front wheel positions while excluding
        # most of a broad centre shadow.
        wheel_zones = (
            (x + round(box_width * 0.08), x + round(box_width * 0.40)),
            (x + round(box_width * 0.60), x + round(box_width * 0.92)),
        )
        tyre_search_top = y + round(box_height * 0.55)
        minimum_tyre_pixels = max(5, round(box_width * 0.008))
        tyre_bottom_rows: list[int] = []
        for zone_left, zone_right in wheel_zones:
            dark_counts = np.count_nonzero(
                grayscale[tyre_search_top:y + box_height, zone_left:zone_right] < 100,
                axis=1,
            )
            contact_rows = np.flatnonzero(dark_counts >= minimum_tyre_pixels)
            if contact_rows.size:
                tyre_bottom_rows.append(tyre_search_top + int(contact_rows[-1]) + 1)
        if tyre_bottom_rows and wheel_ground is None:
            tyre_ground = max(tyre_bottom_rows)
            maximum_safe_trim = round(box_height * 0.06)
            if 0 <= y + box_height - tyre_ground <= maximum_safe_trim:
                box_height = tyre_ground - y

        selected_area = np.count_nonzero(
            selected[y:y + box_height, x:x + box_width]
        )
        fill_ratio = selected_area / (box_width * box_height)
        box_area_ratio = box_width * box_height / image_area
        aspect = box_width / max(box_height, 1)
        inverse_scale = 1.0 / scale
        initial_bounds = Bounds(
            left=max(0, int(np.floor(x * inverse_scale))),
            right=min(
                original_width,
                int(np.ceil((x + box_width) * inverse_scale)),
            ),
            roof=max(0, int(np.floor(y * inverse_scale))),
            ground=min(
                original_height,
                int(np.ceil((y + box_height) * inverse_scale)),
            ),
        )
        initial_bounds.validate(original_width, original_height)
        self.last_attempt_bounds = initial_bounds
        low_confidence = (
            box_area_ratio > 0.92
            or fill_ratio > 0.92
            or aspect < 1.2
        )
        aspect_mismatch = (
            expected_aspect is not None
            and abs(aspect / expected_aspect - 1.0) > 0.35
        )
        if expected_aspect is not None and (low_confidence or aspect_mismatch):
            return self._detect_complex_scene(
                working=working,
                original_width=original_width,
                original_height=original_height,
                scale=scale,
                expected_aspect=expected_aspect,
            )
        if low_confidence:
            raise RuntimeError(
                "Automatic detection confidence is too low; use --manual for this image"
            )

        return initial_bounds
