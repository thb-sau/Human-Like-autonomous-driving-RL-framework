import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import cv2

from smarts.core.agent_interface import RGB
from smarts.core.colors import Colors, SceneColors
from smarts.core.utils.observations import points_to_pixels, replace_rgb_image_color


class FilterObs:
    """Filter only the selected observation parameters."""

    def __init__(self, top_down_rgb: RGB):

        self._no_color = np.zeros((3,))
        self._wps_color = np.array(Colors.GreenTransparent.value[0:3]) * 255
        self._traffic_color = np.array(SceneColors.SocialVehicle.value[0:3]) * 255
        self._road_color = np.array(SceneColors.Road.value[0:3]) * 255
        self._lane_divider_color = np.array(SceneColors.LaneDivider.value[0:3]) * 255
        self._edge_divider_color = np.array(SceneColors.EdgeDivider.value[0:3]) * 255
        self._ego_color = np.array(SceneColors.Agent.value[0:3]) * 255
        self._goal_color = np.array(Colors.Purple.value[0:3]) * 255

        self._blur_radius = 2
        self._res = top_down_rgb.resolution
        h = top_down_rgb.height
        w = top_down_rgb.width
        shape = (
            (
                math.floor(w / 2 - 3.68 / 2 / self._res),
                math.ceil(w / 2 + 3.68 / 2 / self._res),
            ),
            (
                math.floor(h / 2 - 1.47 / 2 / self._res),
                math.ceil(h / 2 + 1.47 / 2 / self._res),
            ),
        )
        self._rgb_mask = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        self._rgb_mask[shape[0][0] : shape[0][1], shape[1][0] : shape[1][1], :] = 1


    def filter(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapts the environment's observation."""
        # fmt: off

        # Ego's heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego_heading = (obs["ego_vehicle_state"]["heading"] + np.pi) % (2 * np.pi) - np.pi
        ego_pos = obs["ego_vehicle_state"]["position"]

        # Get rgb image, remove road, and replace other egos (if any) as background vehicles
        rgb = obs["top_down_rgb"]
        h, w, _ = rgb.shape
        # rgb_noroad = replace_rgb_image_color(rgb=rgb, old_color=[self._road_color, self._lane_divider_color, self._edge_divider_color], new_color=self._no_color)
        # rgb_ego = replace_rgb_image_color(rgb=rgb_noroad, old_color=[self._ego_color], new_color=self._traffic_color, mask=self._rgb_mask)

        # rgb = replace_rgb_image_color(rgb=rgb, old_color=[self._traffic_color], new_color=self._goal_color)  # 尝试置换其他车辆颜色
        rgb_ego = rgb

        # Superimpose waypoints onto rgb image
        wps = obs["waypoint_paths"]["position"][0:11, 3:, 0:3]
        for path in wps[:]:
            wps_valid = points_to_pixels(
                points=path,
                center_position=ego_pos,
                heading=ego_heading,
                width=w,
                height=h,
                resolution=self._res,
            )
            for point in wps_valid:
                img_x, img_y = point[0], point[1]
                # if all(rgb_ego[img_y, img_x, :] == self._no_color):
                #     rgb_ego[img_y, img_x, :] = self._wps_color
                if not all(rgb_ego[img_y, img_x, :] == self._traffic_color):
                    rgb_ego[img_y, img_x, :] = self._wps_color
                    self.change_color_around_pixel(rgb_ego, img_y, img_x, self._wps_color)

        # Superimpose goal position onto rgb image       
        if not all((goal:=obs["ego_vehicle_state"]["mission"]["goal_position"]) == np.zeros((3,))):       
            goal_pixel = points_to_pixels(
                points=np.expand_dims(goal,axis=0),
                center_position=ego_pos,
                heading=ego_heading,
                width=w,
                height=h,
                resolution=self._res,
            )
            if len(goal_pixel) != 0:
                img_x, img_y = goal_pixel[0][0], goal_pixel[0][1]
                # if all(rgb_ego[img_y, img_x, :] == self._no_color) or all(rgb_ego[img_y, img_x, :] == self._wps_color):
                #     rgb_ego[
                #         max(img_y-self._blur_radius,0):min(img_y+self._blur_radius,h), 
                #         max(img_x-self._blur_radius,0):min(img_x+self._blur_radius,w), 
                #         :,
                #     ] = self._goal_color
                rgb_ego[
                    max(img_y-self._blur_radius,0):min(img_y+self._blur_radius,h), 
                    max(img_x-self._blur_radius,0):min(img_x+self._blur_radius,w), 
                    :,
                ] = self._goal_color

        
        rgb_ego = cv2.resize(rgb_ego, (80, 80), interpolation=cv2.INTER_AREA)  # 缩放

        # Channel first rgb
        rgb_ego = rgb_ego.transpose(2, 0, 1)

        filtered_obs = np.uint8(rgb_ego)

        return filtered_obs
        # fmt: on

    def change_color_around_pixel(self, image, y, x, color):
        for i in range(y-1, y+2):
            for j in range(x-1, x+2):
                if 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                    image[i, j, :] = color