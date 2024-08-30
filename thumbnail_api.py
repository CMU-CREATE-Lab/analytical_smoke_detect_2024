import urllib.parse

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @staticmethod
    def from_pts(pts: str):
        # Example: "4654,2127,4915,2322,pts"
        tokens = pts.split(",")
        assert len(tokens) == 5, "Invalid number of tokens"
        assert tokens[-1] == "pts", "Invalid token"
        x1, y1, x2, y2 = map(float, tokens[:4])
        return Rectangle(x1, y1, x2, y2)

    def to_pts(self):
        nums = [self.x1, self.y1, self.x2, self.y2]
        nums = [int(num) if num.is_integer() else num for num in nums]
        return f"{','.join(map(str, nums))},pts"

    def __repr__(self):
        return f"Rect(left={self.x1}, top={self.y1}, right={self.x2}, bot={self.y2})"

class Thumbnail:
    @staticmethod
    def from_url(url):
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
                "v": Rectangle.from_pts
            }
            for key, type_ in root_non_string_params.items():
                if key in root_params:
                    root_params[key] = type_(root_params[key])
            
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

        thumbnail = Thumbnail()
        thumbnail.root = main_params['root']
        thumbnail.width = main_params['width']
        thumbnail.height = main_params['height']
        thumbnail.format = main_params['format']
        thumbnail.fps = main_params['fps']
        assert main_params['fps'] == root_params['fps'], "FPS values do not match"
        thumbnail.tile_format = main_params['tileFormat']
        thumbnail.start_dwell = main_params['startDwell']
        thumbnail.end_dwell = main_params['endDwell']
        thumbnail.from_screenshot = 'fromScreenshot' in main_params
        thumbnail.minimal_ui = 'minimalUI' in main_params
        thumbnail.disable_ui = 'disableUI' in main_params
        assert main_params['startDwell'] == root_params['startDwell'], "Start dwell values do not match"
        assert main_params['endDwell'] == root_params['endDwell'], "End dwell values do not match"
        thumbnail.v = root_params['v']
        thumbnail.t = root_params['t']
        thumbnail.ps = root_params['ps']
        thumbnail.bt = root_params['bt']
        thumbnail.et = root_params['et']
        thumbnail.d = root_params['d']
        thumbnail.s = root_params['s']
        return thumbnail

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
        root_params = {
            'v': self.v.to_pts(),
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
        print("root_url:", root_url)
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
        return (self.width / (self.v.x2 - self.v.x1), self.height / (self.v.y2 - self.v.y1))
    
    def set_scale(self, x_scale, y_scale):
        self.width = int((self.v.x2 - self.v.x1) * x_scale)
        self.height = int((self.v.y2 - self.v.y1) * y_scale)
    
    def copy(self):
        return Thumbnail.from_url(self.to_url())
    
    def get_pil_image(self):
        # Assert scale is 1
        assert self.scale() == (1,1)
        tmp_thumbnail = self.copy()

        # vertical margin has to be added to hold title and time.  we'll remove it later
        # vertical_margin = 40
        # tmp_thumbnail.v.y1 -= vertical_margin
        # tmp_thumbnail.set_scale(1, 1)
        # tmp_url = tmp_thumbnail.to_url()
        # Fetch data at URL and turn into PIL image
        import requests
        from PIL import Image
        from io import BytesIO
        
        response = requests.get(self.to_url())
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            # Crop to remove the vertical margin
            # image = image.crop((0, vertical_margin, image.width, image.height))
            return image
        else:
            raise Exception(f"Failed to fetch image. Status code: {response.status_code}")
