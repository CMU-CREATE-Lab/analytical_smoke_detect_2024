from typing import Union


class View:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def intersection(self, other) -> Union["View", None]:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if left < right and top < bottom:
            return View(left, top, right, bottom)
        else:
            return None

    def translate(self, dx, dy) -> "View":
        return View(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    @staticmethod
    def full():
        return View(0, 0, 7930, 2808)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @staticmethod
    def from_pts(pts: str) -> "View":
        # Example: "4654,2127,4915,2322,pts"
        tokens = pts.split(",")
        assert len(tokens) == 5, "Invalid number of tokens"
        assert tokens[-1] == "pts", "Invalid token"
        left, top, right, bottom = map(float, tokens[:4])
        return View(left, top, right, bottom)

    def center(self):
        return (self.right - self.left) // 2, (self.bottom - self.top) // 2

    def expose(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def subsample(self, nlevels) -> "View":
        assert (nlevels & (nlevels - 1)) == 0, "Expected level to be power of 2."

        return View(
            round(self.left / nlevels),
            round(self.top / nlevels),
            round(self.right / nlevels),
            round(self.bottom / nlevels)
        )

    def to_pts(self):
        return f"{','.join(map(str, [int(num) if num.is_integer() else num for num in self.expose()]))},pts"

    def upsample(self, nlevels) -> "View":
        assert (nlevels & (nlevels - 1)) == 0, "Expected level to be power of 2."

        return View(
            round(self.left * nlevels),
            round(self.top * nlevels),
            round(self.right * nlevels),
            round(self.bottom * nlevels)
        )
    
    def __repr__(self):
        return f"View(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"
