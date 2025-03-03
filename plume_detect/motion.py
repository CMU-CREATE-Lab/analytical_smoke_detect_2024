import cv2 as cv
import numpy as np

from collections import defaultdict, namedtuple
from itertools import product
from typing import Callable, Literal, Union

from cmn import DisjointSet
from view import View

FramePadding = Union[int, tuple[int, int]]
PixelPadding = Union[int, tuple[int, int], tuple[int, int, int, int]]


class TemporalContour:
    """A collection of coordinates in a video corresponding to motion."""

    def __init__(self, points: list[tuple[int, int, int]]):
        """Creates a new contour from a list of points

        Parameters
        ----------
        * points - a collection of (frame, row, column) triples
        """
        self.points = np.array(points)
        self.frames = np.sort(np.unique(self.points[:, 0]))
        self.number_of_frames = len(self.frames)
        self.number_of_points = len(points)
        max_y, max_x, min_y, min_x = 0, 0, 2147483647, 2147483647

        for frame in self.frames:
            mask_candidates = self.points[:, 0] == frame
            points = self.points[mask_candidates, 1:]
            y_values, x_values = points[:, 0], points[:, 1]

            max_y = max(max_y, np.max(y_values))
            max_x = max(max_x, np.max(x_values))
            min_y = min(min_y, np.min(y_values))
            min_x = min(min_x, np.min(x_values))

        self.region = View(min_x, min_y, max_x, max_y)
        self.width = self.region.width
        self.height = self.region.height
    
    def density(self, digits: int = 3) -> float:
        return round(self.number_of_points / (len(self.frames) * self.region.width * self.region.height), digits)
    
class MotionAnalysis:
    """Temporal motion analysis of a video."""

    _KERNEL_ = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    _SPATIAL4_ = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    _SPATIAL8_ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, video: np.ndarray, *, background_subtractor):
        self.video = video
        self.number_of_frames, self.height, self.width = video.shape[:3]

        masks = [None] * self.number_of_frames

        for i in range(self.number_of_frames):
            masks[i] = cv.morphologyEx(background_subtractor.apply(video[i]), cv.MORPH_OPEN, MotionAnalysis._KERNEL_)

        self.masks = np.array(masks)

    def contours(self, neighbors: Literal[4, 8] = 8, depth: int = 3, *, threshold: int = 127) -> list[TemporalContour]:
        """Analyzes the video producing temporal contours.
        
        Parameters
        ----------
        * neighbors - number of neighboring pixels to consider
        * depth - number of frames to look backward/forward
        * threshold - value in range [0, 255], any value greater than `threshold` will be considered.

        Returns
        -------
        
        A collection of events, where each event visualizes a list of 3D contours over
        the input video.
        """
        vid = self.masks > threshold
        spatial_neighbors = MotionAnalysis._SPATIAL8_ if neighbors == 8 else MotionAnalysis._SPATIAL4_
        labels, label_set, next_label = np.zeros_like(vid, dtype=np.int32), DisjointSet(), 1

        for f, r, c in product(range(self.number_of_frames), range(self.height), range(self.width)):
            if not vid[f, r, c]:
                continue

            nbrs = []

            for y, x in [(r + dy, c + dx) for dy, dx in spatial_neighbors]:
                if 0 <= f < self.number_of_frames and 0 <= y < self.height and 0 <= x < self.width and vid[f, y, x]:
                    nbrs.append((f, y, x))

            nbrs += [(t, r, c) for t in [f - dt for dt in range(1, depth + 1)] if t >= 0 and vid[t, r, c]]

            if nbrs:
                if non_zero_labels := [labels[t, y, x] for t, y, x in nbrs if labels[t, y, x] > 0]:
                    labels[f, r, c] = np.min(non_zero_labels)
                else:
                    labels[f, r, c] = next_label
                    label_set.add(next_label)
                    next_label += 1
 
                for non_zero_label in non_zero_labels:
                    label_set.union(labels[f, r, c], non_zero_label)
            else:
                labels[f, r, c] = next_label
                label_set.add(next_label)
                next_label += 1

        components = defaultdict(list)

        for f, r, c in product(range(self.number_of_frames), range(self.height), range(self.width)):
            if labels[f, r, c] > 0:
                label = label_set.find(labels[f, r, c])
                components[label].append((f, r, c))

        return [TemporalContour(p) for p in components.values()]

    def count_white_pixels(self, 
                           contour: TemporalContour, 
                           hls_video: np.ndarray, *,
                           nlevels: int = 1,
                           lightness_lower_bound: int = 200) -> int:
            """Counts the number of pixels in a video considered white.
            
            Parameters
            ----------
            * contour - The collection of pixels to check
            * hls_video - A video in HSL/HLS format
            * nlevels - The resolution to use for the pixels in `contour`
            * lightness_lower_bound - minimum (L)ightness necessary to be considered white, in range [0, 255]

            Returns
            -------
            The number of points in `contour` considered white in `hsl_video`.
            """
            white_pixels = 0

            for frame in contour.frames:
                points = contour.points[:, 0] == frame
                coords = contour.points[points,:,1:]
                lightness_channel = hls_video[frame,:,:,1]
                mask = cv.inRange(lightness_channel, lightness_lower_bound, 255)

                for y, x in coords:
                    if mask[y * nlevels, x * nlevels] == 255:
                        white_pixels += 1

            return white_pixels

    def get_contour(self,
                    contour: TemporalContour,
                    video: np.ndarray, *, 
                    nlevels: int = 1, 
                    pad_frames: FramePadding = 0, 
                    pad_region: PixelPadding = 0) -> np.ndarray:
        """Pulls the part of a video corresponding to a contour.
        
        Parameters
            ----------
            * contour - collection of pixels
            * video - source video
            * nlevels - resolution to use for the bounding rectangle of `contour`
            * pad_frames - number of extra frames to add to start and/or end
            * pad_region - number of pixels to add to bounding rectangle dimensions

            Returns
            -------
            An array of frames from `video` trimmed to the region containing the pixels in `contour`.
        """
        region = contour.region.upsample(nlevels)
        l, t, r, b = region.left, region.top, region.right, region.bottom
        n, h, w = video.shape[:3]
        frame_start_padding, frame_end_padding = (pad_frames, pad_frames) if isinstance(pad_frames, int) else pad_frames

        if isinstance(pad_region, int):
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region, pad_region, pad_region, pad_region
        elif len(pad_region) == 2:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region[0], pad_region[1], pad_region[0], pad_region[1]
        else:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region

        l = max(l - pixels_l, 0)
        t = max(t - pixels_t, 0)
        r = min(r + pixels_r, w - 1)
        b = min(b + pixels_b, h - 1)
        start_frame = max(contour.frames[0] - frame_start_padding, 0)
        end_frame = min(contour.frames[-1] + frame_end_padding, n - 1)
        frames = [*range(start_frame, contour.frames[0]), *contour.frames, *range(n, end_frame + 1)]

        return np.array([video[f][t:(b + 1), l:(r + 1), :] for f in frames])
    
    def has_white_pixel(self, 
                           contour: TemporalContour, 
                           hls_video: np.ndarray, *,
                           nlevels: int = 1,
                           lightness_lower_bound: int = 200) -> int:
            """Counts the number of pixels in a video considered white.
            
            Parameters
            ----------
            * contour - The collection of pixels to check
            * hls_video - A video in HSL/HLS format
            * nlevels - The resolution to use for the pixels in `contour`
            * lightness_lower_bound - minimum (L)ightness necessary to be considered white, in range [0, 255]

            Returns
            -------
            The number of points in `contour` considered white in `hsl_video`.
            """
            for frame in contour.frames:
                points = contour.points[:, 0] == frame
                coords = contour.points[points,1:]
                lightness_channel = hls_video[frame,:,:,1]
                mask = cv.inRange(lightness_channel, lightness_lower_bound, 255)

                for y, x in coords:
                    if mask[y * nlevels, x * nlevels] == 255:
                        return True

            return False
    