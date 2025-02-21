import cv2 as cv
import math
import numpy as np

from collections import defaultdict, namedtuple
from itertools import product
from typing import Literal, Union

from .cmn import DisjointSet, video_to_html, videos_to_html, videos_to_html_stack
from .view import View

Color = namedtuple('Color', ["red", "green", "blue"])
Point = namedtuple("Point", ["frame", "row", "column"])

BLACK = Color(0, 0, 0)
BLUE = Color(0, 0, 255)
GRAY = Color(128, 128, 128)
GREEN = Color(0, 255, 0)
ORANGE = Color(255, 128, 0)
PURPLE = Color(128, 0, 128)
RED = Color(255, 0, 0)
WHITE = Color(255, 255, 255)
YELLOW = Color(255, 255, 0)


class Contour:
    """A collection of triples, where each triple is the frame, row, and column of a pixel."""

    def __init__(self, points: list[Union[tuple[int, int, int], Point]]):
        """Creates a new contour from a list of points, which are either (frame, row, column) or `Point` triples."""
        self.array = np.array(points)
        self.frames = np.sort(np.unique(self.array[:, 0]))
        self.points = [point if isinstance(point, Point) else Point(*point) for point in points]

    def __getitem__(self, index):
        """Indexes the underlying NumPy array of points."""
        return self.array[index]

    def __len__(self):
        """Number of points in the contour."""
        return len(self.points)

    def bounding_rect(self) -> View:
        """The smallest rectangle containing every point in the contour."""
        max_y, max_x = 0, 0
        min_y, min_x = 2147483647, 2147483647

        for frame in self.frames:
            mask_candidates = self.array[:, 0] == frame
            points = self.array[mask_candidates][:, 1:]
            y_values = points[:, 0]
            x_values = points[:, 1]

            max_y = np.max([max_y, np.max(y_values)])
            max_x = np.max([max_x, np.max(x_values)])
            min_y = np.min([min_y, np.min(y_values)])
            min_x = np.min([min_x, np.min(x_values)])

        return View(min_x, min_y, max_x, max_y)


class Event:
    """Visualizes a contour on a background."""

    def __init__(self, video: np.ndarray, contour: Contour, *, bgcolor: Color = BLACK, fgcolor: Color = GRAY):
        self.height, self.width = video.shape[1:3]
        self.contour = contour
        self.region = contour.bounding_rect()
        self.frames = contour.frames
        self.background = bgcolor
        self.foreground = fgcolor

        video_frames = []
        masks = []
        masked_frames = []

        for frame in contour.frames:
            mask = np.full((self.height, self.width, 3), bgcolor, dtype=np.uint8)
            masked_frame = np.full((self.height, self.width, 3), bgcolor, dtype=video.dtype)

            mask_candidates = contour[:, 0] == frame
            points = contour[mask_candidates][:, 1:]

            for y, x in points:
                mask[y, x] = fgcolor
                masked_frame[y, x] = video[frame, y, x]

            video_frames.append(video[frame])
            masks.append(mask)
            masked_frames.append(masked_frame)

        self.video = np.array(video_frames)
        self.masks = np.array(masks)
        self.masked_frames = np.array(masked_frames)

    @property
    def size(self) -> tuple[int, int]:
        return self.region.width, self.region.height
    
    def __len__(self):
        """Number of frames in the event."""
        return len(self.frames)

    def density(self, digits: Union[int, None] = 3) -> float:
        n, w, h = self.video.shape[:-1]

        return round(len(self.contour) / (n * w * h), digits)
    
    def region_density(self, digits: int = 3) -> float:
        return round(len(self.contour) / (self.video.shape[0] * self.region.width * self.region.height), digits)

        return np.array(upsampled_video)
    
    def get_masked_video(self) -> np.ndarray:
        """HTML for masked video."""
        l, t, r, b = self.region.left, self.region.top, self.region.right, self.region.bottom
        video = [f[t:(b + 1), l:(r + 1), :] for f in self.masked_frames]

        return np.array(video)
    
    def get_video(self) -> np.ndarray:
        """HTML for the input video, contour masks, and the masked video."""
        l, t, r, b = self.region.left, self.region.top, self.region.right, self.region.bottom
        video = [f[t:(b + 1), l:(r + 1), :] for f in self.video]

        return np.array(video)
    
    def get_upsampled_video(self, 
                            nlevels: int, 
                            video: np.ndarray, *,
                            pad_frames: Union[int, tuple[int, int]] = 0, 
                            pad_pixels: Union[int, tuple[int, int], tuple[int, int, int, int]] = 0) -> np.ndarray:
        """HTML for the input video, contour masks, and the masked video."""
        region = self.region.upsample(nlevels)
        l, t, r, b = region.left, region.top, region.right, region.bottom
        n, h, w = video.shape[:3]
        frame_padding_start, frame_padding_end = (pad_frames, pad_frames) if isinstance(pad_frames, int) else pad_frames

        if isinstance(pad_pixels, int):
            pixels_l, pixels_t, pixels_r, pixels_b = pad_pixels, pad_pixels, pad_pixels, pad_pixels
        elif len(pad_pixels) == 2:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_pixels[0], pad_pixels[1], pad_pixels[0], pad_pixels[1]
        else:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_pixels

        l = max(l - pixels_l, 0)
        t = max(t - pixels_t, 0)
        r = min(r + pixels_r, w - 1)
        b = min(b + pixels_b, h - 1)
        start_frame = max(self.frames[0] - frame_padding_start, 0)
        end_frame = min(self.frames[-1] + frame_padding_end, n - 1)
        frame_nos = [
            *range(start_frame, self.frames[0]),
            *self.frames,
            *range(n, end_frame + 1)
        ]

        return np.array([video[f][t:(b + 1), l:(r + 1), :] for f in frame_nos])

    def with_background_color(self, color: Color):
        """Changes the background color for contour masks and masked video."""
        return Event(self.video, self.contour, bgcolor=color, fgcolor=self.foreground)

    def with_colors(self, bgcolor: Color, fgcolor: Color):
        """Changes the background color for contour masks and masked video, and the foreground color of the mask."""
        return Event(self.video, self.contour, bgcolor=bgcolor, fgcolor=fgcolor)

    def with_foreground_color(self, color: Color):
        """Changes the foreground color of the mask."""
        return Event(self.video, self.contour, bgcolor=self.background, fgcolor=color)


class EventSpace:
    """Collection of events based on a list of contours."""

    def __init__(self, events: list[Event], neighborhood_size: int, temporal_window: int):
        self.contours = [e.contour for e in events]
        self.events = events
        self.neighborhood_size = neighborhood_size
        self.temporal_window = temporal_window

    @classmethod
    def from_contours(cls, video: np.ndarray, contours: list[Contour], neighborhood_size: int, temporal_window: int):
        return cls([Event(video, c) for c in contours], neighborhood_size, temporal_window)

    def __getitem__(self, index):
        """Gets the event at `index`."""
        return self.events[index]

    def __len__(self):
        """The number of events in the space."""
        return len(self.events)
    
    def filter_events(self, *, points: int = 0, frames: int = 0, width: int = 0, height: int = 0) -> "EventSpace":
        if points < 1 and frames < 1 and width < 1 and height < 1:
            return self
        
        def event_filter(event: Event) -> bool:
            return (len(event.contour) >= points and 
                    len(event.contour.frames) >= frames and
                    event.region.width >= width and
                    event.region.height >= height)
        
        return EventSpace(list(filter(event_filter, self.events)), self.neighborhood_size, self.temporal_window)

    def masks(self, trim: bool = False):
        """HTML for the contour masks."""
        if trim:
            return [e.trimmed_masks() for e in self.events]
        else:
            return [e.masks for e in self.events]

    def masked_videos(self, padding: int = 0, trim: bool = False):
        """HTML for masked video."""
        if trim:
            return [e.get_masked_video() for e in self.events]
        else:
            return [e.masked_frames for e in self.events]
    
    def sorted_by_area(self, *, descending: bool = False) -> "EventSpace":
        return EventSpace(
            sorted(self.events, key=lambda e: e.region.width * e.region.height, reverse=descending), 
            self.neighborhood_size, 
            self.temporal_window
        )
    
    def sorted_by_points(self, *, descending: bool = False) -> "EventSpace":
        return EventSpace(
            sorted(self.events, key=lambda e: len(e.contour), reverse=descending), 
            self.neighborhood_size, 
            self.temporal_window
        )
    
    def upsampled_videos(self, 
                         nlevels: int, 
                         video: np.ndarray, *,
                         pad_frames: Union[int, tuple[int, int]] = 0, 
                         pad_pixels: Union[int, tuple[int, int], tuple[int, int, int, int]] = 0):
        return [e.get_upsampled_video(nlevels, video, pad_frames=pad_frames, pad_pixels=pad_pixels) for e in self.events]

    def videos(self):
        return [e.get_video() for e in self.events]
        
class Motion:
    """Temporal motion analysis of a video."""

    _KERNEL_ = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    _SPATIAL4_ = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    _SPATIAL8_ = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, video: np.ndarray, *, detectShadows: bool = True, varThreshold: int = 16):
        self.depth, self.height, self.width = video.shape[:3]
        self.depth -= 1
        self.video = video[1:].copy()

        bgsub = cv.createBackgroundSubtractorMOG2(detectShadows=detectShadows, varThreshold=varThreshold)
        bgsub.apply(video[0])

        self.masks = np.array([
            cv.morphologyEx(bgsub.apply(frame), cv.MORPH_OPEN, Motion._KERNEL_) for frame in video[1:]
        ])

    def event_space(
            self, 
            neighborhood: Literal[4, 8] = 8, 
            span: int = 3, *, 
            points: int = 0,
            frames: int = 0,
            width: int = 0,
            height: int = 0,
            threshold: int = 127) -> EventSpace:
        """Searches the video for temporal contours.
        
        Parameters
        ----------
        * neighborhood - number of neighboring pixels to check
        * span - number of frames to look back/forward
        * points - minimum number of points required in the contour of an event
        * frames - minimum number of frames required for an event
        * width - minimum width of an event
        * height - minimum height of an event

        Returns
        -------
        
        A collection of events, where each event visualizes a list of 3D contours over
        the input video.
        """
        def is_valid(frame, row, col):
            return 0 <= frame < self.depth and 0 <= row < self.height and 0 <= col < self.width and vid[frame, row, col]

        vid = self.masks > threshold
        spatial_neighbors = Motion._SPATIAL8_ if neighborhood == 8 else Motion._SPATIAL4_
        labels, label_set, next_label = np.zeros_like(vid, dtype=np.int32), DisjointSet(), 1

        for f, r, c in product(range(self.depth), range(self.height), range(self.width)):
            if not vid[f, r, c]:
                continue

            nbrs = [(f, y, x) for y, x in [(r + dy, c + dx) for dy, dx in spatial_neighbors] if is_valid(f, y, x)]
            nbrs += [(t, r, c) for t in [f - dt for dt in range(1, span + 1)] if t >= 0 and vid[t, r, c]]

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

        for f, r, c in product(range(self.depth), range(self.height), range(self.width)):
            if labels[f, r, c] > 0:
                label = label_set.find(labels[f, r, c])
                components[label].append((f, r, c))

        if points < 1 and frames < 1 and width < 1 and height < 1:
            return EventSpace.from_contours(self.video, [Contour(p) for p in components.values()], neighborhood, span)
        
        filtered_events = []

        for p in components.values():
            if len(p) >= points:
                e = Event(self.video, Contour(p))

                if len(e.frames) >= frames and e.region.width >= width and e.region.height >= height:
                    filtered_events.append(e)

        return EventSpace(filtered_events, neighborhood, span)
