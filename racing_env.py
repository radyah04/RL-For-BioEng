import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

try:
    import pygame
except Exception:
    pygame = None


class RacingEnv(gym.Env):
    """
    Simple top-down racing track environment.
    - Track is a loop of waypoints forming a rectangle.
    - Agent controls a car with discrete actions: steer left/right, accelerate, brake, no-op.
    - Observation: [cos(theta), sin(theta), v_norm, dx_next, dy_next, lateral_error]
      where (dx_next, dy_next) is the normalized vector from car to next waypoint.
    - Reward: forward progress along the active segment, minus lateral error and control cost.
    - Terminate: off-track or completed lap; Truncate: max steps.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, render_fps: int = 30):
        super().__init__()
        self.render_mode = render_mode
        # Frames per second for Pygame rendering. Use 0 for unlimited.
        self.render_fps = int(render_fps)
        # Discrete actions: 0 left, 1 right, 2 accel, 3 brake, 4 no-op
        self.action_space = spaces.Discrete(5)
        # Observation: 6-dim continuous
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Physics parameters
        self.dt = 0.1
        self.max_speed = 8.0
        self.steer_delta = math.radians(6.0)
        self.accel = 2.0
        self.brake = 2.5
        self.track_width = 40.0  # pixels half-width
        self.max_steps = 1000

        # Track definition (rectangle loop of waypoints)
        W, H = 800, 600
        margin = 120
        self.margin = margin
        self.screen_size = (W, H)
        self.waypoints = [
            (margin, margin), (W - margin, margin), (W - margin, H - margin), (margin, H - margin)
        ]
        self.num_wp = len(self.waypoints)

        # State
        self.pos = np.zeros(2, dtype=np.float32)
        self.theta = 0.0
        self.v = 0.0
        self.wp_idx = 0
        self.step_count = 0
        self.lap_count = 0
        self.trail = deque(maxlen=150)

        # Pygame
        self._screen = None
        self._clock = None
        # Colors
        self._CLR_BG = (28, 28, 28)
        self._CLR_ROAD = (50, 50, 50)
        self._CLR_LANE = (160, 160, 160)
        self._CLR_BORDER = (90, 90, 90)
        self._CLR_CAR = (220, 60, 60)
        self._CLR_TEXT = (220, 220, 220)

    def _segment_info(self, i: int) -> Tuple[np.ndarray, np.ndarray, float]:
        a = np.array(self.waypoints[i % self.num_wp], dtype=np.float32)
        b = np.array(self.waypoints[(i + 1) % self.num_wp], dtype=np.float32)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            d = np.array([1.0, 0.0], dtype=np.float32)
            L = 1.0
        t_hat = d / L
        return a, t_hat, L

    def _lateral_error(self, a: np.ndarray, t_hat: np.ndarray, p: np.ndarray) -> float:
        # Perpendicular distance from p to the line through a in direction t_hat
        ap = p - a
        # Component orthogonal to t_hat
        perp = ap - np.dot(ap, t_hat) * t_hat
        return float(np.clip(np.linalg.norm(perp) / max(self.track_width, 1e-6), 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        a, t_hat, _ = self._segment_info(self.wp_idx)
        to_next = np.array(self.waypoints[(self.wp_idx + 1) % self.num_wp], dtype=np.float32) - self.pos
        dist = np.linalg.norm(to_next)
        if dist > 1e-6:
            dn = to_next / dist
        else:
            dn = np.zeros(2, dtype=np.float32)
        lat_err = self._lateral_error(a, t_hat, self.pos)
        obs = np.array([
            math.cos(self.theta), math.sin(self.theta),
            np.clip(self.v / self.max_speed, 0.0, 1.0),
            float(np.clip(dn[0], -1.0, 1.0)), float(np.clip(dn[1], -1.0, 1.0)),
            lat_err
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.wp_idx = 0
        a, t_hat, _ = self._segment_info(self.wp_idx)
        # Start slightly behind first waypoint, aligned to segment
        self.pos = a + t_hat * 10.0
        self.theta = math.atan2(t_hat[1], t_hat[0])
        self.v = 2.0
        self.step_count = 0
        self.trail.clear()

        if self.render_mode == "human" and pygame is not None:
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(self.screen_size)
                self._clock = pygame.time.Clock()
        return self._obs(), {}

    def step(self, action: int):
        self.step_count += 1
        # Apply action
        if action == 0:
            self.theta -= self.steer_delta
        elif action == 1:
            self.theta += self.steer_delta
        elif action == 2:
            self.v = min(self.v + self.accel * self.dt, self.max_speed)
        elif action == 3:
            self.v = max(self.v - self.brake * self.dt, 0.0)
        # no-op: 4

        # Dynamics
        dx = self.v * math.cos(self.theta) * self.dt
        dy = self.v * math.sin(self.theta) * self.dt
        self.pos += np.array([dx, dy], dtype=np.float32)

        # Progress and waypoint update
        a, t_hat, L = self._segment_info(self.wp_idx)
        # Project to segment to measure progress
        ap = self.pos - a
        s = float(np.dot(ap, t_hat))
        progress = max(min(s, L), 0.0)
        # Advance waypoint when reaching near end of segment
        reached = s >= L - 10.0
        if reached:
            prev_idx = self.wp_idx
            self.wp_idx = (self.wp_idx + 1) % self.num_wp
            if prev_idx == self.num_wp - 1 and self.wp_idx == 0:
                self.lap_count += 1

        # Reward: forward progress delta along segment, minus lateral error and small control cost
        lat_err = self._lateral_error(a, t_hat, self.pos)
        reward = (self.v * self.dt) - 0.5 * lat_err - 0.01 * (action in (0, 1))

        # Off-track termination: if lateral error exceeds 1.0 (scaled by track_width)
        terminated = lat_err >= 1.0
        # Lap complete when returning to waypoint 0 after visiting all
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human" and pygame is not None:
            self.trail.append((float(self.pos[0]), float(self.pos[1])))
            self._render_frame()

        return self._obs(), reward, terminated, truncated, {}

    def _render_frame(self):
        assert pygame is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self._screen.fill(self._CLR_BG)

        # Build a road ring by drawing outer rect, then cutting inner rect
        W, H = self.screen_size
        outer_rect = pygame.Rect(int(self.margin - self.track_width), int(self.margin - self.track_width),
                                 int(W - 2 * (self.margin - self.track_width)), int(H - 2 * (self.margin - self.track_width)))
        inner_rect = pygame.Rect(int(self.margin + self.track_width), int(self.margin + self.track_width),
                                 int(W - 2 * (self.margin + self.track_width)), int(H - 2 * (self.margin + self.track_width)))

        road_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        pygame.draw.rect(road_surface, self._CLR_ROAD, outer_rect)
        pygame.draw.rect(road_surface, (0, 0, 0, 0), inner_rect)  # punch out inner area (transparent)
        self._screen.blit(road_surface, (0, 0))

        # Lane lines (inner rectangle edges)
        pygame.draw.rect(self._screen, self._CLR_LANE, inner_rect, 2)
        pygame.draw.rect(self._screen, self._CLR_BORDER, outer_rect, 2)

        # Start/finish line near first waypoint
        a = self.waypoints[0]
        b = self.waypoints[1]
        a_i = (int(a[0]), int(a[1]))
        b_i = (int(b[0]), int(b[1]))
        sf_x = int((a_i[0] + b_i[0]) // 2)
        sf_y = int((a_i[1] + b_i[1]) // 2)
        pygame.draw.line(self._screen, (240, 240, 240), (sf_x, sf_y - int(self.track_width)), (sf_x, sf_y + int(self.track_width)), 3)

        # Trail
        if len(self.trail) > 1:
            pygame.draw.lines(self._screen, (255, 140, 0), False, [(int(px), int(py)) for px, py in self.trail], 2)
        # Draw car
        x, y = float(self.pos[0]), float(self.pos[1])
        heading = self.theta
        # Car as rotated rectangle with a nose highlight
        car_len, car_w = 24, 12
        car_surf = pygame.Surface((car_len, car_w), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self._CLR_CAR, pygame.Rect(0, 0, car_len, car_w), border_radius=4)
        pygame.draw.rect(car_surf, (255, 255, 255), pygame.Rect(car_len - 4, 2, 3, car_w - 4), border_radius=2)
        car_rot = pygame.transform.rotate(car_surf, -math.degrees(heading))
        car_rect = car_rot.get_rect(center=(int(x), int(y)))
        self._screen.blit(car_rot, car_rect)
        # HUD
        font = pygame.font.SysFont(None, 24)
        hud1 = font.render(f"v={float(self.v):.2f}", True, self._CLR_TEXT)
        self._screen.blit(hud1, (10, 10))
        hud2 = font.render(f"lap={self.lap_count} wp={self.wp_idx} steps={self.step_count}", True, self._CLR_TEXT)
        self._screen.blit(hud2, (10, 34))
        # Speed bar
        v_ratio = np.clip(self.v / self.max_speed, 0.0, 1.0)
        bar_bg = pygame.Rect(10, 60, 120, 10)
        bar_fg = pygame.Rect(10, 60, int(120 * v_ratio), 10)
        pygame.draw.rect(self._screen, (80, 80, 80), bar_bg)
        pygame.draw.rect(self._screen, (80, 200, 120), bar_fg)
        pygame.display.flip()
        # If render_fps == 0, run as fast as possible
        self._clock.tick(self.render_fps)

    def close(self):
        if pygame is not None and self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None
