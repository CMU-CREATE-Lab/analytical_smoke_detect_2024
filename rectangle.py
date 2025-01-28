import numpy as np

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def intersection(self, other) -> 'Rectangle':
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x1 < x2 and y1 < y2:
            return Rectangle(x1, y1, x2, y2)
        else:
            return None
        
    def translate(self, dx, dy) -> 'Rectangle':
        return Rectangle(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    # Convert float coordinates to integer
    # If float coordinates don't match integer coordinates, raise an error
    
    def ensure_integer(self) -> 'Rectangle':
        ret = Rectangle(int(self.x1), int(self.y1), int(self.x2), int(self.y2))
        assert self.x1 == ret.x1 and self.y1 == ret.y1 and self.x2 == ret.x2 and self.y2 == ret.y2
        return ret
    
    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    
    @property
    def center(self):
        return np.array([(self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2])
    
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
