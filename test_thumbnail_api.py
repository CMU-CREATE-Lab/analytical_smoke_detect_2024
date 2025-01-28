import pytest
from thumbnail_api import Thumbnail, BreathecamThumbnail
import numpy as np
import dateutil.parser
import pytz

def test_thumbnail_url():
    url = "https://thumbnails-v2.createlab.org/thumbnail?root=https%3A%2F%2Fbreathecam.org%2F%23v%3D4654%2C2127%2C4915%2C2322%2Cpts%26t%3D984.02%26ps%3D0%26bt%3D20240519135036%26et%3D20240519135036%26startDwell%3D0%26endDwell%3D0%26d%3D2024-05-19%26s%3Dclairton4%26fps%3D9&width=400&height=300&format=png&fps=9&tileFormat=mp4&startDwell=0&endDwell=0&fromScreenshot&minimalUI"
    thumbnail = Thumbnail(url)
    reconstructed_url = thumbnail.to_url()
    assert url == reconstructed_url

def test_thumbnail_dimensions():
    url = "https://thumbnails-v2.createlab.org/thumbnail?root=https%3A%2F%2Fbreathecam.org%2F%23v%3D4654%2C2127%2C4915%2C2322%2Cpts%26t%3D984.02%26ps%3D0%26bt%3D20240519135036%26et%3D20240519135036%26startDwell%3D0%26endDwell%3D0%26d%3D2024-05-19%26s%3Dclairton4%26fps%3D9&width=400&height=300&format=png&fps=9&tileFormat=mp4&startDwell=0&endDwell=0&fromScreenshot&minimalUI"
    thumbnail = Thumbnail(url)
    im = thumbnail.get_pil_image()
    assert im.width == 400
    assert im.height == 300
    assert im.width == thumbnail.width
    assert im.height == thumbnail.height

def test_breathecam_thumbnail():
    url = "https://share.createlab.org/shorturl/breathecam/f263bebc7632efa9"
    thumbnail = BreathecamThumbnail(url)
    x_scale, y_scale = thumbnail.scale()
    assert abs(x_scale - 0.518918918918919) < 1e-6
    assert abs(y_scale - 0.5187319884726225) < 1e-6
    assert thumbnail.width == 1920
    assert thumbnail.height == 1080

def test_thumbnail_set_scale():
    url = "https://share.createlab.org/shorturl/breathecam/f263bebc7632efa9"
    thumbnail = BreathecamThumbnail(url)
    thumbnail.set_scale(1, 1)
    x_scale, y_scale = thumbnail.scale()
    assert x_scale == 1.0
    assert y_scale == 1.0
    assert thumbnail.width == 3700
    assert thumbnail.height == 2082

def test_resize_view_rect_preserving_scale():
    url = "https://share.createlab.org/shorturl/breathecam/f263bebc7632efa9"
    thumbnail = BreathecamThumbnail(url)
    thumbnail.set_scale(1, 1)

    new_width = 3840
    new_height = 2160

    center_before = thumbnail.view_rect().center
    thumbnail.resize_rect_preserving_scale(new_width, new_height)
    center_after = thumbnail.view_rect().center

    # Assert the distance between the centers is less than 1
    assert np.linalg.norm(center_after - center_before) < 1

    assert thumbnail.width == new_width
    assert thumbnail.height == new_height
    assert thumbnail.view_rect().width == new_width
    assert thumbnail.view_rect().height == new_height

def test_change_begin_end_times():
    url = "https://share.createlab.org/shorturl/breathecam/f263bebc7632efa9"
    thumbnail = BreathecamThumbnail(url)
    eastern_tz = pytz.timezone("US/Eastern")

    begin_time = eastern_tz.localize(dateutil.parser.parse("2024-05-19 1:00"))
    end_time = eastern_tz.localize(dateutil.parser.parse("2024-05-19 2:00"))
    thumbnail.set_begin_end_times(begin_time, end_time)
    assert thumbnail.begin_time_in_camera_timezone() == begin_time
    assert thumbnail.end_time_in_camera_timezone() == end_time



    
