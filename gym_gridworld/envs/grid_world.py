import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps" : 4}
    def __init__(self, render_mode = None, size = 5):

        self.np_random, seed = seeding.np_random(42)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.size = size

        # Observation [ [] * size] * size
        self.obs_shape = [self.size, self.size]

        # nothing = 0
        # player = 1
        # target = 2
        # obstacles = 3
        self.observation_space = spaces.Box(low=0, high=3, shape=(5,5), dtype=np.uint8)

        # Action. Left, right, up, down
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
                # y, x
            0 : np.array([1, 0]), # down
            1 : np.array([0, 1]), # right
            2 : np.array([-1, 0]), # up
            3 : np.array([0, -1]) # left
        }

        # The grid
        self.grid = [ [0 for i in range(self.size)] for i in range(self.size)]

    def reset(self, seed=42):
        # The grid
        self.grid = [ [0 for i in range(self.size)] for i in range(self.size)]

        self._agent_location = self.np_random.randint(0 , self.size, size=2)

        self._target_location = self._agent_location

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.randint(0, self.size, size=2)

        # we put the agent and the goal in the 
        self.grid[self._agent_location[0]][self._agent_location[1]] = 1
        self.grid[self._target_location[0]][self._target_location[1]] = 2

        # TODO: add obstacles in our world (value 3)

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return np.array(self.grid, dtype=np.uint8)

    def step(self, action):
        if torch.is_tensor(action):
            action = action.item()

        direction = self._action_to_direction[action]

        old_agent_location = self._agent_location

        # move player if we are still inside
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # clear old spot
        self.grid[old_agent_location[0]][old_agent_location[1]] = 0
        # put in new spot
        self.grid[self._agent_location[0]][self._agent_location[1]] = 1

        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1.0 if done else 0.0
        obs = self._get_obs()
        return obs, reward, done, dict()

    def render(self, a):
        pass
        for row in self.grid:
            print(row)
        print("\n")
