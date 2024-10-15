import dataclasses
import datetime
import numpy as np
import requests
import urllib.parse

from collections import namedtuple
from io import BytesIO
from PIL import Image
from typing import Any, Final


FRAME_T_OFFSETS_1 = (0.02, 0.11, 0.19, 0.27, 0.36, 0.44, 0.52, 0.61, 0.69, 0.77, 0.86, 0.94)
FRAME_T_OFFSETS_2 = (0.03, 0.11, 0.19, 0.28, 0.36, 0.44, 0.53, 0.61, 0.69, 0.78, 0.86, 0.94)


BoundingBox = namedtuple("BoundingBox", ["l", "t", "r", "b"])
PixelCenter = namedtuple("PixelCenter", ["x", "y", "scale"])


@dataclasses.dataclass(kw_only=True)
class BreatheCamView:
    KEYS = ["v", "t", "bt", "et", "startDwell", "endDwell", "d", "s", "fps", "ps"]

    v: BoundingBox | PixelCenter
    t: float
    bt: datetime.datetime
    et: datetime.datetime
    d: str
    s: str
    fps: int
    startDwell: float = 0
    endDwell: float = 0
    ps: float = 0

    def as_params(self):
        d = dataclasses.asdict(self)

        def norm(val: float):
            return int(val) if val.is_integer() else val

        if isinstance(self.v, BoundingBox):
            d["v"] = f"{norm(self.v.l)},{norm(self.v.t)},{norm(self.v.r)},{norm(self.v.b)},pts"
        else:
            d["v"] = f"{norm(self.v.x)},{norm(self.v.y)},{norm(self.v.scale)},pts"

        return d

    def encode(self, safe='') -> str:
        encoded_params = []

        for key, value in self.as_params().items():
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(value, datetime.datetime):
                value = value.strftime("%Y%m%d%H%M%S")


            encoded_params.append(f"{key}={urllib.parse.quote(str(value), safe=safe)}")

        return '&'.join(encoded_params)

@dataclasses.dataclass(kw_only=True)
class BreatheCamThumbnail:
    KEYS = [
        "root", "fps", "startDwell", "endDwell", 
        "width", "height", "format", "tileFormat", 
        "fromScreenshot", "minimalUI", "disableUI"
    ]
    
    root: str
    fps: int
    startDwell: float = 0
    endDwell: float = 0
    width: int = 400
    height: int = 300
    format: str = "png"
    tileFormat: str = "mp4"
    fromScreenshot: bool = True
    minimalUI: bool = True
    disableUI: bool = False

    def as_params(self):
        return dataclasses.asdict(self)

    def encode(self, safe='') -> str:
        encoded_params = []

        for key, value in self.as_params().items():
            if isinstance(value, bool):
                if value:
                    encoded_params.append(key)
            else:
                if isinstance(value, float) and value.is_integer():
                    value = int(value)

                encoded_params.append(f"{key}={urllib.parse.quote(str(value), safe=safe)}")

        return '&'.join(encoded_params)

class BreatheCamFrame:
    BASE_URL: Final[str] = "https://thumbnails-v2.createlab.org/thumbnail"
    ROOT_URL: Final[str] = "https://breathecam.org"

    def __init__(self, params: dict[str, Any], root_params: dict[str, Any]):
        if "root" not in params:
            raise f"Malformed URL: missing parameter `root`"

        self.params = {**root_params, **params}


    @staticmethod
    def from_thumbnail(url: str):
        thumbnail = BreatheCamFrame._parse_params(url)

        if "root" not in thumbnail:
            raise f"Malformed URL: missing parameter `root`"

        view = BreatheCamFrame._parse_params(thumbnail["root"].replace('#', '?'))

        BreatheCamFrame._setup_frame(thumbnail, view)

        return BreatheCamFrame(thumbnail, view)

    @staticmethod
    def from_view(url: str, **kwargs: Any):
        view = BreatheCamFrame._parse_params(url.replace('#', '?'))
        validate = {
            "width": int,
            "height": int,
            "format": str,
            "tileFormat": str,
            "fromScreenshot": bool,
            "minimalUI": bool,
            "disableUI": bool
        }

        for key, value in kwargs.items():
            if key in validate:
                if not isinstance(value, validate[key]):
                    raise f"Invalid parameter `{key}`: {value}"
            else:
                raise f"Unexpected parameter `{key}`."

        thumbnail = {
            'root': url,
            'width': kwargs.get('width', 400),
            'height': kwargs.get('height', 300),
            'format': kwargs.get('format', 'png'),
            'fps': view["fps"],
            'tileFormat': kwargs.get('tileFormat', 'mp4'),
            'startDwell': view["startDwell"],
            'endDwell': view["endDwell"],
            'fromScreenshot': kwargs.get('fromScreenshot', True),
            'minimalUI': kwargs.get('minimalUI', True),
            'disableUI': kwargs.get('disableUI', False)
        }

        BreatheCamFrame._setup_frame(thumbnail, view)

        return BreatheCamFrame(thumbnail, view)


    def copy(self):
        view = {k: self.params[k] for k in BreatheCamView.KEYS}
        thumbnail = {k: self.params[k] for k in BreatheCamThumbnail.KEYS}
        thumbnail["root"] = f"{BreatheCamFrame.ROOT_URL}#{BreatheCamView(**view).encode(safe='%,')}"

        return BreatheCamFrame(thumbnail, view)

    def get_images(self, count: int, *, subsample: int, step: int = 1, augment_frame_offsets: bool = False):
        yield self.image(subsample=subsample)

        step = max(step, 1)
        thumb = self.copy()

        for _ in range(1, count):
            thumb.shift(step, augment_frame_offsets=augment_frame_offsets)
            yield thumb.image(subsample=subsample)

    def image(self, subsample: int) -> np.ndarray:
        response = requests.get(self.to_url())

        image = Image.open(BytesIO(response.content))
        image = image.resize((image.width // subsample, image.height // subsample))

        return np.array(image) / 255.0

    def remove_labels(self):
        self.params["minimalUI"] = False
        self.params["disableUI"] = True
        
        return self

    def scale(self) -> tuple[float, float]:
        height, width, v = self.params["height"], self.params["width"], self.params["v"]

        return width / (v.right - v.left), height / (v.bottom - v.top)

    def set_scale(self, x: float, y: float):
        v = self.params["v"]
        self.params["width"] = int((v.right - v.left) * x)
        self.params["height"] = int((v.bottom - v.top) * y)

        return self

    def set_size(self, height: int, width: int):
        self.params["width"] = width
        self.params["height"] = height

        return self

    def shift(self, frames: int, *, augment_frame_offsets: bool = False):
        time_start = self.params["t"]
        time_base = int(str(time_start)[:-3])
        frame_offsets = FRAME_T_OFFSETS_2 if augment_frame_offsets else FRAME_T_OFFSETS_1
        time_offset_idx = frame_offsets.index(float(str(time_start)[-3:]))
        time_base_offset, frame_offset = divmod(time_offset_idx + frames, len(frame_offsets))

        self.params["t"] = time_base + time_base_offset + frame_offsets[frame_offset]
        self.params["bt"] += datetime.timedelta(seconds=3 * frames)
        self.params["et"] = self.params["bt"]

        return self

    def shifted(self, frames: int, *, augment_frame_offsets: bool = False):
        return self.copy().shift(frames, augment_frame_offsets=augment_frame_offsets)
    
    def show_labels(self):
        self.params["minimalUI"] = True
        self.params["disableUI"] = False

        return self

    def stream_images(self, seconds: int, *, subsample: int, augment_frame_offsets: bool = False):
        return self.get_images(np.ceil(seconds / 3), subsample=subsample, augment_frame_offsets=augment_frame_offsets)

    def thumbnail(self):
        return BreatheCamThumbnail(**{k: self.params[k] for k in BreatheCamThumbnail.KEYS})

    def to_url(self) -> str:
        view = self.view().encode(safe="%,")
        thumbnail = {k: self.params[k] for k in BreatheCamThumbnail.KEYS}
        thumbnail["root"] = f"{BreatheCamFrame.ROOT_URL}#{view}"

        return f"{BreatheCamFrame.BASE_URL}?{BreatheCamThumbnail(**thumbnail).encode()}"

    def unshift(self, frames: int, *, augment_frame_offsets: bool = False):
        time_start = self.params["t"]
        time_base = int(str(time_start)[:-3])
        frame_offsets = FRAME_T_OFFSETS_2 if augment_frame_offsets else FRAME_T_OFFSETS_1
        time_offset_idx = frame_offsets.index(float(str(time_start)[-3:]))
        frame_offset = (time_offset_idx - frames) % len(frame_offsets)
        time_base_offset = 0

        if time_offset_idx < frames:
            time_base_offset = (frames - time_offset_idx) / len(frame_offsets) + 1

        self.params["t"] = time_base - time_base_offset + frame_offsets[frame_offset]
        self.params["bt"] -= datetime.timedelta(seconds=3 * frames)
        self.params["et"] = self.params["bt"]

        return self

    def unshifted(self, frames: int, *, augment_frame_offsets: bool = False):
        return self.copy().unshift(frames, augment_frame_offsets=augment_frame_offsets)

    def view(self):
        return BreatheCamView(**{k: self.params[k] for k in BreatheCamView.KEYS})

    def with_scale(self, x: float, y: float):
        thumb = self.copy()
        thumb.set_scale(x, y)

        return thumb
    
    def with_size(self, height: int, width: int):
        thumb = self.copy()
        
        thumb.params["width"] = width
        thumb.params["height"] = height

        return thumb


    @staticmethod
    def _parse_params(url: str) -> dict[str, Any]:
        parsed_url = urllib.parse.urlparse(url)
        query_params = parsed_url.query.split('&')
        parsed_params = {}

        for param in query_params:
            if '=' in param:
                key, value = param.split('=', 1)
                parsed_params[key] = urllib.parse.unquote(value)
            else:
                parsed_params[param] = True

        return parsed_params

    @staticmethod
    def _setup_frame(parsed_params: dict[str, Any], root_params: dict[str, Any]):
        if "v" not in root_params:
            raise "Malformed URL: missing parameter `v`"
        elif isinstance(root_params['v'], str):
            if not root_params['v'].endswith(",pts"):
                raise f"Malformed URL: invalid parameter `v` :: {root_params['v']}"

            coords = root_params['v'].split(",")[:-1]

            if len(coords) == 4:
                root_params['v'] = BoundingBox(*(list(map(float, coords))))
            else:
                root_params['v'] = PixelCenter(*(list(map(float, coords))))

        root_params['t'] = float(root_params['t'])
        root_params["ps"] = float(root_params["ps"])

        if isinstance(root_params["bt"], str):
            root_params["bt"] = datetime.datetime.strptime(root_params["bt"], "%Y%m%d%H%M%S")

        root_params["et"] = root_params["bt"]

        for key, con in [("fps", int), ("startDwell", float), ("endDwell", float)]:
            if key in root_params:
                if con(root_params[key]) != (val := con(parsed_params[key])):
                    raise f"Malformed URL: invalid parameter `{key}` :: {root_params[key]} != {parsed_params[key]}"

                parsed_params[key] = val

        parsed_params["fromScreenshot"] = "fromScreenshot" in parsed_params
        parsed_params["minimalUI"] = "minimalUI" in parsed_params
        parsed_params["disableUI"] = "disableUI" in parsed_params
        parsed_params["height"] = int(parsed_params.get("height", 300))
        parsed_params["width"] = int(parsed_params.get("width", 400))
