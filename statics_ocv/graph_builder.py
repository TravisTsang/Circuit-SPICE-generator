"""Skeletonize trace masks and recover electrical topology as a NetworkX graph."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from skimage.morphology import skeletonize

from .config import AppConfig
from .detection import BoundingBox, ComponentDetection

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TerminalCandidate:
    """A nearby wire-net candidate for one component terminal."""

    net_label: int
    point: tuple[int, int]
    distance_to_anchor: float
    pixel_count: int
    points: list[tuple[int, int]] = field(default_factory=list)


@dataclass(slots=True)
class CircuitTopology:
    """Recovered topology and debug data."""

    graph: nx.Graph
    components: list[ComponentDetection]
    net_names: dict[int, str]
    warnings: list[str] = field(default_factory=list)
    skeleton: np.ndarray | None = None
    net_label_image: np.ndarray | None = None


class CircuitGraphBuilder:
    """Build a circuit graph from a binary trace mask and component detections."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.warnings: list[str] = []

    def build(
        self,
        binary_mask: np.ndarray,
        components: list[ComponentDetection],
    ) -> CircuitTopology:
        """Recover schematic topology and map component terminals to SPICE nodes."""

        self.warnings = []
        if binary_mask.ndim != 2:
            raise ValueError("Expected a single-channel binary mask.")

        skeleton = self._skeletonize(binary_mask)
        label_image, net_points = self._label_wire_nets(skeleton)
        self._warn_for_open_circuits(skeleton, label_image)

        graph = nx.Graph()
        net_names = self._initial_net_names(net_points)
        self._map_ground_symbols(components, label_image, net_names)

        for label, name in net_names.items():
            graph.add_node(
                name,
                kind="net",
                label=int(label),
                pixel_count=len(net_points.get(label, [])),
            )

        for component in components:
            self._attach_component(component, graph, label_image, net_points, net_names)

        for warning in self.warnings:
            LOGGER.warning(warning)

        return CircuitTopology(
            graph=graph,
            components=components,
            net_names=net_names,
            warnings=list(self.warnings),
            skeleton=skeleton,
            net_label_image=label_image,
        )

    def _skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """Convert a trace mask into a one-pixel-wide wire skeleton."""

        mask_bool = binary_mask.astype(bool)
        skel = skeletonize(mask_bool)
        return skel.astype(np.uint8)

    def _label_wire_nets(
        self,
        skeleton: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
        """Label connected wire skeleton components and remove tiny specks."""

        labels = measure.label(skeleton > 0, connectivity=2).astype(np.int32)
        net_points: dict[int, list[tuple[int, int]]] = {}
        next_label = 1
        cleaned = np.zeros_like(labels, dtype=np.int32)

        for region in measure.regionprops(labels):
            coords_yx = region.coords
            if len(coords_yx) < self.config.graph.min_wire_pixels:
                continue
            points = [(int(x), int(y)) for y, x in coords_yx]
            for x, y in points:
                cleaned[y, x] = next_label
            net_points[next_label] = points
            next_label += 1

        LOGGER.info("Recovered %d wire-net skeleton components.", len(net_points))
        return cleaned, net_points

    def _initial_net_names(self, net_points: dict[int, list[tuple[int, int]]]) -> dict[int, str]:
        return {label: f"N{label:03d}" for label in sorted(net_points)}

    def _map_ground_symbols(
        self,
        components: list[ComponentDetection],
        label_image: np.ndarray,
        net_names: dict[int, str],
    ) -> None:
        """Map nets touching ground symbols to SPICE ground node 0."""

        for component in components:
            if not component.is_ground:
                continue
            candidates = self._terminal_candidates(component, label_image, expected_count=1)
            if not candidates:
                self._warn(
                    f"{component.uid}: ground symbol is not connected to any detected wire."
                )
                continue
            best = candidates[0]
            net_names[best.net_label] = "0"
            component.terminal_nets = ["0"]
            component.terminal_points = [best.point]

    def _attach_component(
        self,
        component: ComponentDetection,
        graph: nx.Graph,
        label_image: np.ndarray,
        net_points: dict[int, list[tuple[int, int]]],
        net_names: dict[int, str],
    ) -> None:
        """Add a component node and connect it to inferred electrical nets."""

        if component.is_ground:
            graph.add_node(
                component.uid,
                kind="component",
                class_name=component.class_name,
                prefix=component.spice_prefix,
                value=component.value,
                bbox=component.bbox,
            )
            if component.terminal_nets:
                graph.add_edge(component.uid, component.terminal_nets[0], terminal=1)
            return

        expected = self.config.terminal_count_for_prefix(component.spice_prefix)
        candidates = self._terminal_candidates(component, label_image, expected)
        anchors = self._terminal_anchors(component.bbox, expected)
        assignments = self._assign_candidates_to_anchors(component, candidates, anchors, expected)

        component.terminal_nets = []
        component.terminal_points = []

        graph.add_node(
            component.element_name,
            kind="component",
            class_name=component.class_name,
            prefix=component.spice_prefix,
            value=component.value,
            bbox=component.bbox,
            confidence=component.confidence,
        )

        for idx in range(expected):
            candidate = assignments[idx] if idx < len(assignments) else None
            if candidate is None:
                nc_name = f"NC_{component.element_name}_{idx + 1}"
                graph.add_node(nc_name, kind="net", label=None, pixel_count=0, open=True)
                graph.add_edge(component.element_name, nc_name, terminal=idx + 1)
                component.terminal_nets.append(nc_name)
                component.terminal_points.append(None)
                self._warn(
                    f"{component.element_name}: terminal {idx + 1} has no nearby wire; "
                    f"assigned open node {nc_name}."
                )
                continue

            net_name = net_names.get(candidate.net_label, f"N{candidate.net_label:03d}")
            if net_name not in graph:
                graph.add_node(
                    net_name,
                    kind="net",
                    label=candidate.net_label,
                    pixel_count=len(net_points.get(candidate.net_label, [])),
                )
            graph.add_edge(component.element_name, net_name, terminal=idx + 1)
            component.terminal_nets.append(net_name)
            component.terminal_points.append(candidate.point)

        if len(set(component.terminal_nets)) < len(component.terminal_nets):
            self._warn(
                f"{component.element_name}: multiple terminals map to the same net "
                f"({', '.join(component.terminal_nets)}); component may be shorted."
            )

    def _terminal_candidates(
        self,
        component: ComponentDetection,
        label_image: np.ndarray,
        expected_count: int,
    ) -> list[TerminalCandidate]:
        """Find distinct wire nets that touch or approach a component's bounding box."""

        h, w = label_image.shape
        search = component.bbox.expanded(self.config.graph.terminal_search_radius_px)
        x1 = max(0, int(math.floor(search.x1)))
        y1 = max(0, int(math.floor(search.y1)))
        x2 = min(w - 1, int(math.ceil(search.x2)))
        y2 = min(h - 1, int(math.ceil(search.y2)))

        labels_to_points: dict[int, list[tuple[int, int]]] = {}
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                label = int(label_image[y, x])
                if label <= 0:
                    continue
                if not self._point_is_terminal_contact(x, y, component.bbox):
                    continue
                labels_to_points.setdefault(label, []).append((x, y))

        anchors = self._terminal_anchors(component.bbox, expected_count)
        candidates: list[TerminalCandidate] = []
        for label, points in labels_to_points.items():
            best_point, best_distance = self._best_point_for_anchors(points, anchors)
            candidates.append(
                TerminalCandidate(
                    net_label=label,
                    point=best_point,
                    distance_to_anchor=best_distance,
                    pixel_count=len(points),
                    points=points,
                )
            )

        candidates.sort(key=lambda c: (c.distance_to_anchor, -c.pixel_count))
        return candidates

    def _point_is_terminal_contact(self, x: int, y: int, bbox: BoundingBox) -> bool:
        """Keep skeleton pixels near a component edge, rejecting deep interior noise."""

        radius = self.config.graph.terminal_search_radius_px
        margin = self.config.graph.terminal_inner_margin_px

        inside_x = bbox.x1 <= x <= bbox.x2
        inside_y = bbox.y1 <= y <= bbox.y2
        if inside_x and inside_y:
            distance_to_edge = min(x - bbox.x1, bbox.x2 - x, y - bbox.y1, bbox.y2 - y)
            return distance_to_edge <= margin

        dx = max(bbox.x1 - x, 0.0, x - bbox.x2)
        dy = max(bbox.y1 - y, 0.0, y - bbox.y2)
        return math.hypot(dx, dy) <= radius

    @staticmethod
    def _best_point_for_anchors(
        points: Iterable[tuple[int, int]],
        anchors: list[tuple[float, float]],
    ) -> tuple[tuple[int, int], float]:
        """Return the point with minimum distance to any expected terminal anchor."""

        best_point = (0, 0)
        best_distance = float("inf")
        for point in points:
            px, py = point
            distance = min(math.hypot(px - ax, py - ay) for ax, ay in anchors)
            if distance < best_distance:
                best_point = point
                best_distance = distance
        return best_point, best_distance

    def _assign_candidates_to_anchors(
        self,
        component: ComponentDetection,
        candidates: list[TerminalCandidate],
        anchors: list[tuple[float, float]],
        expected: int,
    ) -> list[TerminalCandidate | None]:
        """Greedily assign distinct nearby net candidates to expected terminals."""

        if not candidates:
            return [None] * expected

        assignments: list[TerminalCandidate | None] = [None] * expected
        used_points: set[tuple[int, int]] = set()
        max_anchor_distance = float(self.config.graph.terminal_search_radius_px + 2)

        ranked_pairs: list[tuple[float, int, TerminalCandidate, tuple[int, int]]] = []
        for anchor_idx, anchor in enumerate(anchors):
            for candidate in candidates:
                point, distance = self._best_point_for_anchors(candidate.points, [anchor])
                if distance <= max_anchor_distance:
                    ranked_pairs.append((distance, anchor_idx, candidate, point))

        for distance, anchor_idx, candidate, point in sorted(ranked_pairs, key=lambda item: item[0]):
            if assignments[anchor_idx] is not None:
                continue
            if point in used_points:
                continue
            assignments[anchor_idx] = TerminalCandidate(
                net_label=candidate.net_label,
                point=point,
                distance_to_anchor=distance,
                pixel_count=candidate.pixel_count,
                points=candidate.points,
            )
            used_points.add(point)
            if all(item is not None for item in assignments):
                break

        missing = sum(item is None for item in assignments)
        if missing:
            self._warn(
                f"{component.element_name}: found {expected - missing}/{expected} expected "
                "wire contacts near the symbol."
            )

        if len(candidates) > expected:
            extra = len(candidates) - expected
            self._warn(
                f"{component.element_name}: {extra} extra nearby wire net(s) were ignored; "
                "check for crossing wires or an oversized detection box."
            )

        return assignments

    @staticmethod
    def _terminal_anchors(bbox: BoundingBox, expected: int) -> list[tuple[float, float]]:
        """Estimate terminal anchor locations from a detector bounding box."""

        cx, cy = bbox.center
        if expected <= 1:
            return [(cx, cy)]

        horizontal = bbox.width >= bbox.height
        if expected == 2:
            if horizontal:
                return [(bbox.x1, cy), (bbox.x2, cy)]
            return [(cx, bbox.y1), (cx, bbox.y2)]

        if expected == 3:
            if horizontal:
                return [(bbox.x1, cy), (bbox.x2, bbox.y1), (bbox.x2, bbox.y2)]
            return [(bbox.x1, bbox.y2), (cx, bbox.y1), (bbox.x2, bbox.y2)]

        if expected == 4:
            return [
                (bbox.x1, cy),
                (cx, bbox.y1),
                (bbox.x2, cy),
                (cx, bbox.y2),
            ]

        return [
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y1),
            (bbox.x1, bbox.y2),
            (bbox.x2, bbox.y2),
            (cx, cy),
        ][:expected]

    def _warn_for_open_circuits(self, skeleton: np.ndarray, label_image: np.ndarray) -> None:
        """Find skeleton endpoints that almost touch but belong to different nets."""

        endpoints = self._skeleton_endpoints(skeleton, label_image)
        if len(endpoints) < 2:
            return

        coords = np.array([(x, y) for x, y, _ in endpoints], dtype=np.float32)
        tree = cKDTree(coords)
        radius = self.config.graph.endpoint_gap_warning_px
        seen: set[tuple[int, int]] = set()

        for idx, (x, y, label) in enumerate(endpoints):
            for neighbor_idx in tree.query_ball_point((x, y), r=radius):
                if neighbor_idx == idx:
                    continue
                nx_, ny_, neighbor_label = endpoints[neighbor_idx]
                if neighbor_label == label:
                    continue
                pair = tuple(sorted((idx, neighbor_idx)))
                if pair in seen:
                    continue
                seen.add(pair)
                distance = math.hypot(x - nx_, y - ny_)
                self._warn(
                    "Possible broken wire: endpoints "
                    f"({x},{y}) on N{label:03d} and ({nx_},{ny_}) on N{neighbor_label:03d} "
                    f"are {distance:.1f}px apart."
                )

    @staticmethod
    def _skeleton_endpoints(
        skeleton: np.ndarray,
        label_image: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Return skeleton pixels with one or zero neighboring skeleton pixels."""

        points: list[tuple[int, int, int]] = []
        ys, xs = np.nonzero(skeleton)
        h, w = skeleton.shape
        for y, x in zip(ys, xs):
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            x1 = max(0, x - 1)
            x2 = min(w, x + 2)
            degree = int(skeleton[y1:y2, x1:x2].sum()) - 1
            label = int(label_image[y, x])
            if label > 0 and degree <= 1:
                points.append((int(x), int(y), label))
        return points

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
