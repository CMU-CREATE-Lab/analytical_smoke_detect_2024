import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


from common import read_image_from_url
from thumbnail_api import Thumbnail


FRAME_OFFSETS = [0.02, 0.11, 0.19, 0.27, 0.36, 0.44, 0.52, 0.61, 0.69, 0.77, 0.86, 0.94]

FRAME = {
   "direction": "left",
   "pad": {"r": 13, "t": 89},
   "showactive": False,
   "x": 0.1,
   "xanchor": "right",
   "y": 0,
   "yanchor": "top"
}


def animate_video(images: list[np.ndarray] | np.ndarray, *, title=""):
   fig, ax = plt.subplots()

   ims = []

   for i in range(len(images)):
      ims.append([ax.imshow(images[i] * 255.0, animated=True)])

   anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

   plt.show()


def get_frames(url: str, subsample: int, seconds: int) -> list[np.array]:
   return get_n_frames(url, subsample, seconds * 12)


def get_n_frames(url: str, subsample: int, count: int) -> list[np.array]:
   thumbnail = Thumbnail.from_url(url)
   time_start = thumbnail.t
   time_base = int(str(time_start)[:-3])
   time_offset_index = FRAME_OFFSETS.index(float(str(time_start)[-3:]))
   frames = [None] * count
   frames[0] = read_image_from_url(url, subsample=subsample) / 255.0
   bt = datetime.datetime.strptime(thumbnail.bt, "%Y%m%d%H%M%S")

   for i in range(1, count):
      time_offset_index = time_offset_index + 1

      if time_offset_index == 12:
         time_offset_index = 0
         time_base += 1

      bt = bt + datetime.timedelta(seconds=3)

      thumbnail.bt = bt.strftime("%Y%m%d%H%M%S")
      thumbnail.et = thumbnail.bt
      thumbnail.t = time_base + FRAME_OFFSETS[time_offset_index]
      
      frames[i] = read_image_from_url(thumbnail.to_url(), subsample=subsample) / 255.0

   return frames

def make_sliders(num_frames: int):
   return [{
      "active": 0,
      "currentvalue": {"prefix": "Frame: "},
      "steps": [{
         "method": "animate",
         "args": [
            [str(k)],
            {
               "frame": {"duration": 100, "redraw": "True"},
               "transition": {"duration": 0},
               "mode": "immediate"
            }
         ],
         "label": str(k)
      } for k in range(num_frames)]
   }]


def stack_frames(url: str, subsample: int, seconds: int) -> np.array:
   return np.array(get_n_frames(url, subsample=subsample, count=seconds * 12))


def stack_n_frames(url: str, subsample: int, count: int) -> np.array:
   return np.array(get_n_frames(url, subsample=subsample, count=count))

