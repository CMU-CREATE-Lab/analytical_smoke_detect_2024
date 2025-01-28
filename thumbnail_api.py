import copy
import urllib.parse
import requests
import datetime
import pytz
from rectangle import Rectangle

# For breathecam, FPS is 12

def find_target_if_redirect(url):
    try:
        response = requests.head(url, allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            return response.headers['Location']
        return url  # No redirect
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
    
class Thumbnail:
    def __init__(self, url: str):
        # follow redirect if we're using an url shortener
        url = find_target_if_redirect(url)

        # Parse the URL and extract query parameters
        parsed_url = urllib.parse.urlparse(url)
        main_params = Thumbnail.parse_query_params(parsed_url)

        # Now parsed_params contains both key-value pairs and single-word flags
        
        # If the following exist, they should be numeric
        # width, height, fps, startDwell, endDwell
        non_string_params = {
            "width": int,
            "height": int,
            "fps": int,
            "startDwell": float,
            "endDwell": float
        }

        for key, type_ in non_string_params.items():
            if key in main_params:
                main_params[key] = type_(main_params[key])

        # Decode the 'root' parameter if it exists
        if 'root' in main_params:
            root_url = main_params['root']
            parsed_root = urllib.parse.urlparse(root_url.replace("#", "?"))
            root_params = Thumbnail.parse_query_params(parsed_root)
            root_non_string_params = {
                "fps": int,
                "startDwell": float,
                "endDwell": float,
                "t": float,
            }
            for key, type_ in root_non_string_params.items():
                if key in root_params:
                    root_params[key] = type_(root_params[key])
            
            if 'v' in root_params:
                coords = root_params['v'].split(',')

                if len(coords) > 4:
                    root_params['v'] = Rectangle.from_pts(root_params['v'])
                else:
                    root_params['v'] = tuple(map(float, coords[:-1]))

            # Update the 'root' value in main_params with the parsed URL without query string
            main_params['root'] = urllib.parse.urlunparse(parsed_root._replace(query=''))
        else:
            raise ValueError("Root parameter not found in the URL")
        
        # At this point, we have two dictionaries:
        # 1. main_params: contains all parameters from the main URL
        # 2. root_params: contains parameters extracted from the 'root' URL
        
        # Example of Main params
        # {'root': 'https://breathecam.org/', 'width': 400, 'height': 300, 'format': 'png', 'fps': 9, 'tileFormat': 'mp4', 'startDwell': 0.0, 'endDwell': 0.0}

        # Example of Root params
        # Root parameters: {'v': Rect(left=4654.0, top=2127.0, right=4915.0, bot=2322.0), 't': 984.02, 'ps': '0', 'bt': '20240519135036', 'et': '20240519135036', 'startDwell': 0.0, 'endDwell': 0.0, 'd': '2024-05-19', 's': 'clairton4', 'fps': 9}

        # Assuming all URLs have precisely these parameters, let's put them into instance variables
        # For duplicate keys, assert that the values are the same

        self.root = main_params['root']
        self.width = main_params['width']
        self.height = main_params['height']
        self.format = main_params['format']
        self.fps = main_params['fps']
        assert main_params['fps'] == root_params['fps'], "FPS values do not match"
        self.tile_format = main_params['tileFormat']
        self.start_dwell = main_params['startDwell']
        self.end_dwell = main_params['endDwell']
        self.from_screenshot = 'fromScreenshot' in main_params
        self.minimal_ui = 'minimalUI' in main_params
        self.disable_ui = 'disableUI' in main_params
        assert main_params['startDwell'] == root_params['startDwell'], "Start dwell values do not match"
        assert main_params['endDwell'] == root_params['endDwell'], "End dwell values do not match"
        self.v = root_params['v']
        self.t = root_params['t']
        self.ps = root_params['ps']
        self.bt = root_params['bt']
        self.et = root_params['et']
        self.d = root_params['d']
        self.s = root_params['s']
    
    def __repr__(self):
        # Output a string with all the parameters that were parsed in from_url
        return f"Thumbnail(root={self.root}, width={self.width}, height={self.height}, format={self.format}, fps={self.fps}, tile_format={self.tile_format}, start_dwell={self.start_dwell}, end_dwell={self.end_dwell}, from_screenshot={self.from_screenshot}, minimal_ui={self.minimal_ui}, disable_ui={self.disable_ui}, v={self.v}, t={self.t}, ps={self.ps}, bt={self.bt}, et={self.et}, d={self.d}, s={self.s})"

    @staticmethod
    def parse_query_params(parsed_url):
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
    def encode_query_params(params, safe=''):
        # True should become just a token without =
        # Other values should be URL encoded
        encoded_params = []
        for key, value in params.items():
            if value is True:
                encoded_params.append(key)
            elif value is False:
                # Skip False values
                pass
            else:
                # For floating point values, do not include .0 for integers
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                encoded_params.append(f"{key}={urllib.parse.quote(str(value), safe=safe)}")
        return "&".join(encoded_params)

    def remove_labels(self):
        self.minimal_ui = False
        self.disable_ui = True

    def to_url(self):
        # Construct the root URL parameters
        if isinstance(self.v, Rectangle):
            v = self.v.to_pts()
        else:
            pts = map(lambda n: int(n) if n.is_integer() else n, self.v)
            v = f"{','.join(map(str, pts))},pts"

        root_params = {
            'v': v,
            't': self.t,
            'ps': self.ps,
            'bt': self.bt,
            'et': self.et,
            'startDwell': self.start_dwell,
            'endDwell': self.end_dwell,
            'd': self.d,
            's': self.s,
            'fps': self.fps
        }

        # Encode the root URL
        root_url = self.root + '#' + Thumbnail.encode_query_params(root_params, safe='%,')
        #print("root_url:", root_url)
        # URLencode the root URL
        #root_url = urllib.parse.quote(root_url, safe='')
    
        # Construct the main URL parameters
        main_params = {
            'root': root_url,
            'width': self.width,
            'height': self.height,
            'format': self.format,
            'fps': self.fps,
            'tileFormat': self.tile_format,
            'startDwell': self.start_dwell,
            'endDwell': self.end_dwell,
            'fromScreenshot': self.from_screenshot,
            'minimalUI': self.minimal_ui,
            'disableUI': self.disable_ui
        }

        # Construct the final URL
        base_url = "https://thumbnails-v2.createlab.org/thumbnail"
        final_url = base_url + '?' + Thumbnail.encode_query_params(main_params)

        return final_url
    
    def scale(self):
        return (self.width / self.view_rect().width, self.height / self.view_rect().height)
    
    def set_scale(self, x_scale, y_scale):
        self.width = int((self.v.x2 - self.v.x1) * x_scale)
        self.height = int((self.v.y2 - self.v.y1) * y_scale)

    def view_rect(self) -> Rectangle:
        assert isinstance(self.v, Rectangle), "v is not a Rectangle"
        return self.v

    # Resize thumbnail width and height, preserving scale.
    # This can shrink or expand the visible area as expressed by the rectangle.
    # It will preserve the center of the rectangle, to the nearest pixel

    def resize_rect_preserving_scale(self, desired_width, desired_height):
        assert self.scale() == (1, 1), "For now, only works for scale 1"
        rect = self.view_rect()
    
        delta_width = desired_width - rect.width
        delta_left = -(delta_width // 2)
        delta_right = delta_width + delta_left
        
        delta_height = desired_height - rect.height
        delta_top = -(delta_height // 2)
        delta_bottom = delta_height + delta_top
        
        rect.x1 += delta_left
        rect.x2 += delta_right
        rect.y1 += delta_top
        rect.y2 += delta_bottom
        
        self.width = rect.width
        self.height = rect.height
    
    def copy(self):
        return Thumbnail.from_url(self.to_url())
    
    def get_pil_image(self):
        import requests
        from PIL import Image
        from io import BytesIO
        
        response = requests.get(self.to_url())
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Failed to fetch image. Status code: {response.status_code}")

class BreathecamThumbnail(Thumbnail):
    def __init__(self, url: str):
        super().__init__(url)
        assert self.root == "https://breathecam.org/"

    def copy(self) -> 'BreathecamThumbnail':
        return copy.deepcopy(self)

    def camera_id(self) -> str:
        return self.s
    
    def camera_timezone(self):
        # TODO: We need to unhardcode this when we have breathecams in other timezones
        return pytz.timezone('America/New_York')
    
    def begin_time_in_camera_timezone(self) -> datetime.datetime:
        return self._parse_bt_et(self.bt)

    def end_time_in_camera_timezone(self) -> datetime.datetime:
        return self._parse_bt_et(self.et)
    
    def set_begin_end_times(self, begin: datetime.datetime, end: datetime.datetime):
        begin_date = begin.astimezone(self.camera_timezone()).date()
        end_date = end.astimezone(self.camera_timezone()).date()
        assert begin_date == end_date, "Begin and end times must be on the same date"
        self.bt = self._unparse_bt_et(begin)
        self.et = self._unparse_bt_et(end)
    
    def timemachine_root_url(self) -> str:
        url_date = self.begin_time_in_camera_timezone().strftime('%Y-%m-%d')
        return f"https://tiles.cmucreatelab.org/ecam/timemachines/{self.camera_id()}/{url_date}.timemachine"

    # bt and et are in UTC, but Breathecam uses times in the camera's timezone
    def _parse_bt_et(self, yyyymmddhhss: str) -> datetime.datetime:
        datetime_utc = pytz.utc.localize(datetime.datetime.strptime(yyyymmddhhss, "%Y%m%d%H%M%S"))
        datetime_local = datetime_utc.astimezone(self.camera_timezone())
        return datetime_local
    
    def _unparse_bt_et(self, datetime_local: datetime.datetime) -> str:
        datetime_utc = datetime_local.astimezone(pytz.utc)
        return datetime_utc.strftime("%Y%m%d%H%M%S")


