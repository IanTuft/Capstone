import random
import time

import cv2
import numpy as np

from typing import Optional

from numpy import ndarray


class Point2D:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y


class View:
    def __init__(self, image_path: str):
        self.image: Optional[ndarray] = cv2.imread(image_path)

        if self.image is None:
            print(f"Error: Image {image_path} not found.")
            exit(1)

        self.height: int = self.image.shape[:2][0]
        self.width: int = self.image.shape[:2][1]

    def copy_image(self) -> ndarray:
        return self.image.copy()


class MainObject:
    def __init__(self, radius: int, speed: int, x: int, y: int, view_bounds: tuple[int, int]):
        self.radius: int = radius
        self.speed: int = speed
        self.position: Point2D = Point2D(x, y)
        self.view_bounds: tuple[int, int] = view_bounds
        self.direction: list[int] = [random.choice([-1, 1]), random.choice([-1, 1])]
        self.color: tuple[int, int, int] = (255, 255, 255)

    def update(self, frame_time: float) -> None:
        self.position.x += int(self.direction[0] * self.speed * frame_time)
        self.position.y += int(self.direction[1] * self.speed * frame_time)

        if self.position.x <= self.radius or self.position.x >= self.view_bounds[0] - self.radius:
            self.direction[0] = -self.direction[0]
        if self.position.y <= self.radius or self.position.y >= self.view_bounds[1] - self.radius:
            self.direction[1] = -self.direction[1]

    def draw(self, view_image: ndarray) -> None:
        cv2.circle(view_image, (self.position.x, self.position.y), self.radius,
                   self.color, -1, lineType=cv2.LINE_AA)


class MainObjectFrame:
    def __init__(self, main_object: MainObject, size_factor: int = 4):
        self.main_object: MainObject = main_object
        self.size: int = size_factor * self.main_object.radius
        self.half_size: int = self.size // 2
        self.color: tuple[int, int, int] = (255, 0, 0)
        self.center: Point2D = self.main_object.position

    def update(self) -> None:
        self.center = self.main_object.position

    def draw(self, view_image: ndarray) -> None:
        top_left: tuple[int, int] = (self.center.x - self.half_size, self.center.y - self.half_size)
        bottom_right: tuple[int, int] = (self.center.x + self.half_size, self.center.y + self.half_size)
        cv2.rectangle(view_image, top_left, bottom_right, self.color, 1)


class BroadcastFrame:
    def __init__(self, main_object_frame: MainObjectFrame, view_bounds: tuple[int, int], smoothing_factor: float = 0.2):
        self.main_object_frame: MainObjectFrame = main_object_frame
        self.width: int = self._get_tv_standard_width()
        self.height: int = self._get_tv_standard_height()
        self.color: tuple[int, int, int] = (0, 255, 0)
        self.center: Point2D = Point2D(self.main_object_frame.center.x, self.main_object_frame.center.y)
        self.top_left: Point2D = Point2D(self.center.x - self.width // 2, self.center.y - self.height // 2)
        self.bottom_right: Point2D = Point2D(self.center.x + self.width // 2, self.center.y + self.height // 2)
        self.smoothing_factor: float = smoothing_factor
        self.view_width: int = view_bounds[0]
        self.view_height: int = view_bounds[1]
        self.frame_width: int = 1

        self.target_aspect_ratio_factor: float = 1.0
        self.current_aspect_ratio_factor: float = 10.0
        self.aspect_ratio_transition_speed: float = 0.1

    def update(self, trigger: str = "edge", edge_threshold: int = 10, aspect_ratio_factor: float = 0.0) -> None:
        self._change_aspect_ratio(aspect_ratio_factor)

        self.top_left = Point2D(self.center.x - self.width // 2, self.center.y - self.height // 2)
        self.bottom_right = Point2D(self.center.x + self.width // 2, self.center.y + self.height // 2)

        main_object_frame_x: int = self.main_object_frame.center.x
        main_object_frame_y: int = self.main_object_frame.center.y
        main_object_frame_width: int = self.main_object_frame.size
        main_object_frame_height: int = self.main_object_frame.size

        if trigger == "edge":
            is_close_to_edge: bool = (
                    (main_object_frame_x - main_object_frame_width // 2) <= self.top_left.x + edge_threshold
                    or (main_object_frame_x + main_object_frame_width // 2) >= self.bottom_right.x - edge_threshold
                    or (main_object_frame_y - main_object_frame_height // 2) <= self.top_left.y + edge_threshold
                    or (main_object_frame_y + main_object_frame_height // 2) >= self.bottom_right.y - edge_threshold
            )

            if is_close_to_edge:
                self._move_broadcast_frame()

        # elif trigger == "center":
        #     center_bounds: tuple[int, int, int, int] = (
        #         main_object_frame_x - (self.main_object_frame.main_object.speed * main_object_frame_width // 2),
        #         main_object_frame_x + (self.main_object_frame.main_object.speed * main_object_frame_width // 2),
        #         main_object_frame_y - (self.main_object_frame.main_object.speed * main_object_frame_height // 2),
        #         main_object_frame_y + (self.main_object_frame.main_object.speed * main_object_frame_height // 2)
        #     )
        #
        #     if (center_bounds[0] <= self.center.x <= center_bounds[1]
        #             and center_bounds[2] <= self.center.y <= center_bounds[3]):
        #         self._move_broadcast_frame()

    def draw(self, view_image: ndarray) -> None:
        cv2.rectangle(view_image, self.top_left.to_tuple(), self.bottom_right.to_tuple(), self.color, self.frame_width)

    def _change_aspect_ratio(self, aspect_ratio_factor: float) -> None:
        if aspect_ratio_factor <= 0.0:
            return

        self.target_aspect_ratio_factor = aspect_ratio_factor

        if self.current_aspect_ratio_factor != self.target_aspect_ratio_factor:
            direction = 1 if self.target_aspect_ratio_factor > self.current_aspect_ratio_factor else -1
            self.current_aspect_ratio_factor += direction * self.aspect_ratio_transition_speed

            if direction == 1 and self.current_aspect_ratio_factor > self.target_aspect_ratio_factor:
                self.current_aspect_ratio_factor = self.target_aspect_ratio_factor
            elif direction == -1 and self.current_aspect_ratio_factor < self.target_aspect_ratio_factor:
                self.current_aspect_ratio_factor = self.target_aspect_ratio_factor

        if self.current_aspect_ratio_factor <= 0:
            return

        self.width = int(self._get_tv_standard_width() * self.current_aspect_ratio_factor)
        self.height = int(self._get_tv_standard_height() * self.current_aspect_ratio_factor)

    def _move_broadcast_frame(self) -> None:
        main_object_frame_x: int = self.main_object_frame.center.x
        main_object_frame_y: int = self.main_object_frame.center.y

        temp_center: ndarray = np.array((self.center.x, self.center.y), dtype=float)
        main_object_frame_center: ndarray = np.array((main_object_frame_x, main_object_frame_y), dtype=float)
        temp_center += (main_object_frame_center - temp_center) * self.smoothing_factor

        self.center.x = int(temp_center[0])
        self.center.y = int(temp_center[1])

        self.top_left = Point2D(self.center.x - self.width // 2, self.center.y - self.height // 2)
        self.bottom_right = Point2D(self.center.x + self.width // 2, self.center.y + self.height // 2)

        self._enforce_bounds()

    def _enforce_bounds(self) -> None:
        if self.top_left.x < 0 + self.frame_width:
            self.center.x += abs(self.top_left.x) + self.frame_width
        if self.top_left.y < 0 + self.frame_width:
            self.center.y += abs(self.top_left.y) + self.frame_width

        if self.bottom_right.x > self.view_width - self.frame_width:
            self.center.x -= (self.bottom_right.x - self.view_width) + self.frame_width
        if self.bottom_right.y > self.view_height - self.frame_width:
            self.center.y -= (self.bottom_right.y - self.view_height) + self.frame_width

        self.top_left = Point2D(self.center.x - self.width // 2, self.center.y - self.height // 2)
        self.bottom_right = Point2D(self.center.x + self.width // 2, self.center.y + self.height // 2)

    @staticmethod
    def _get_tv_standard_width() -> int:
        return 16

    @staticmethod
    def _get_tv_standard_height() -> int:
        return 9


"""
Features:
* trigger="edge" with edge_threshold=20
* is_zone_on (True/False)
"""


def simulate_bouncing_ball(image_path: str):
    fps: int = 30
    frame_time: float = 1 / fps
    is_zone_on: bool = True

    view: View = View(image_path)

    main_object: MainObject = MainObject(5, 100, view.width // 2, view.height // 2,
                                         (view.width, view.height))
    main_object_frame: MainObjectFrame = MainObjectFrame(main_object)
    broadcast_frame: BroadcastFrame = BroadcastFrame(main_object_frame, (view.width, view.height))

    zone_line_width: int = view.width // 3
    left_line_x: int = zone_line_width
    right_line_x: int = 2 * zone_line_width

    while True:
        start_time: float = time.time()
        view_image: ndarray = view.copy_image()
        factor: float = 10.0

        if is_zone_on:
            if broadcast_frame.center.x < left_line_x or broadcast_frame.center.x > right_line_x:
                factor = 5

        main_object.update(frame_time)
        main_object_frame.update()
        broadcast_frame.update(trigger="edge", edge_threshold=25, aspect_ratio_factor=factor)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            break

        if is_zone_on:
            cv2.line(view_image, (left_line_x, 0), (left_line_x, view.height), (0, 0, 255), 2)
            cv2.line(view_image, (right_line_x, 0), (right_line_x, view.height), (0, 0, 255), 2)

        main_object.draw(view_image)
        main_object_frame.draw(view_image)
        broadcast_frame.draw(view_image)

        cv2.imshow("Simulon Demo", view_image)

        elapsed_time: float = time.time() - start_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)

    cv2.destroyAllWindows()


def main() -> None:
    simulate_bouncing_ball("soccer_field.png")


if __name__ == "__main__":
    main()
