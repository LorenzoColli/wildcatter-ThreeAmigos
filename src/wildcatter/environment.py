"""Driller environment module."""


from __future__ import annotations

import random
from typing import Any

import numpy as np
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
from numpy.typing import NDArray


class SimpleDriller(Env):  # type: ignore
    """Simple driller environment."""

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize environment with config dictionary."""
        self._rng = np.random.default_rng(0)
        
        self.model_type = env_config["model_type"]
                
        if self.model_type == "random":
            self.nrow = env_config["nrow"]
            self.ncol = env_config["ncol"]
        else:
            self.initial_model = np.loadtxt(
                env_config["model_path"],
                delimiter=env_config["delim"],
                dtype="int",
            )
            self.initial_model[self.initial_model>1] = 1
            self.initial_model[self.initial_model<-10] = -10
            self.nrow, self.ncol = self.initial_model.shape
        
        self.available_pipe = env_config["available_pipe"]
        self.available_wells = env_config["available_wells"]
        self.oil_price = env_config["oil_price"]
        self.relocation_cost = env_config["relocation_cost"]
        self.drilling_cost = env_config["drilling_cost"]
        self.drilling_depth_markup = env_config["drilling_depth_markup"]

        self.production = 0
        self.expenses = 0
        self.pipe_used = 0
        self.wells_started = 0
        self.trajectory = None
        self.bit_location = None
        self.last_2_actions = [None, None]

        self._size_action_space = (4             # Drilling direction
                                   + self.ncol-2 # Open new well
                                   + 1)          # Close current well / end campaign
        self.action_space = Discrete(self._size_action_space)
        self._drilling_directions = [[1, 0],  # down
                                     [0, -1],  # left
                                     [-1, 0],  # up
                                     [0, 1],  # right
                                    ]
        # Sequence of drilling directions forms complete circle.
        # This simplifies preventing 180-degree turns, which consist
        # of 3 directions either in sequence (e.g., 1-2-3 or 2-3-0)
        # or in reverse sequence (e.g., 3-2-1 or 0-3-2)

        self.observation_space = Box(
            low=-10, high=10, shape=(self.nrow, self.ncol), dtype="int"
        )
        self.reset()
        
    def action_masks(self) -> list:
        """Return list of bool specifying legal actions"""
        if self.bit_location == None: # Drilling impossible: either start a well or end campaign
            drilling = [False] * 4
            new_well = [self.wells_started < self.available_wells and ground == -10 for ground in self.state[0, 1: self.ncol-1]]
        else:
            z, x = self.bit_location
            drilling = [self.pipe_used < self.available_pipe and self.state[z + dz, x + dx]>=0 for dz, dx in self._drilling_directions]
            new_well = [False] * (self.ncol-2)
        
        end_campaign = [True]
        # Using modular arithmetic to prevent 180-degree turns
        if None in self.last_2_actions: # No risk of U-turn
            pass
        elif (   (self.last_2_actions[0]+1)%4 == self.last_2_actions[1]   # Actions in sequence
              or (self.last_2_actions[0]+3)%4 == self.last_2_actions[1]): # Actions in reverse sequence
            drilling[(self.last_2_actions[0]+2)%4] = False # Prevent action that would complete the 180-degree turn
        if len(drilling + new_well + end_campaign) != self._size_action_space:
            print("len(drilling + new_well + end_campaign):",len(drilling + new_well + end_campaign),len(drilling),len(new_well),len(end_campaign))
            print("self._size_action_space:",self._size_action_space)
            raise Action_Mask_Size_Error
        return drilling + new_well + end_campaign

    def step(  # noqa: C901
        self, action: int
    ) -> tuple[NDArray[np.int_], int, bool, dict[str, Any]]:
        """Take step based on action."""
        done = False
        info: dict[str, Any] = {}

        if action < 4: # Drill!
            dz_dx = self._drilling_directions[action]
            new_location = [prev + now for prev, now in zip(self.bit_location, dz_dx)]
            newrow, newcol = new_location
            self.bit_location = new_location
            self.trajectory[self.wells_started-1].append(new_location)
            self.pipe_used += 1
            cost = self.drilling_cost + self.drilling_depth_markup * newrow
            self.expenses += cost
            reward = - cost

        elif action < 3 + (self.ncol -2): # Open new well!
            newcol = action - 3
            try:
                previous_well = self.surface_hole_location[1]
            except TypeError:
                previous_well = newcol
            self.surface_hole_location = [0, newcol]
            self.bit_location = [0, newcol]
            self.trajectory.append([self.bit_location])
            self.wells_started += 1
            cost = self.relocation_cost * np.abs(newcol - previous_well)
            self.expenses += cost
            reward = - cost
        else:
            if self.surface_hole_location != None: # Stop drilling, extract
                oil = self.extract()
                self.production += oil
                reward = oil
                self.bit_location = None
                self.surface_hole_location = None
            else: # End campaign, sell oil
                done = True
                reward = self.oil_price * self.production

        self.update_state()

        return self.state, reward, done, info

    def update_state(self) -> None:
        """Update state method."""
        self.state[self.state == -3] = -2
        try:
            newrow, newcol = self.bit_location
            self.state[newrow, newcol] = -3
        except TypeError:
            pass

    def render(self) -> None:
        """Gym environment rendering."""
        raise NotImplementedError("No renderer implemented yet.")

    def reset(self) -> NDArray[np.int_]:
        """Reset the status of the environment."""
        if self.model_type == "random":
            self.model = self._rng.integers(low=0, high=2, size=(self.nrow, self.ncol))
        else:
            self.model = self.initial_model.copy()
        
        self.model[0] = self.model[self.nrow-1] = self.model[:, 0] = self.model[:, self.ncol-1] = -10
        self.model[1, 1:self.ncol-2] = 0

        self.surface_hole_location = None
        self.state = self.model.copy()
        self.bit_location = self.surface_hole_location
        self.trajectory = []
        self.production = 0
        self.expenses = 0
        self.pipe_used = 0
        self.wells_started = 0
        return self.state

    def extract(self) -> float:
        row, col = self.bit_location
        if self.model[row, col] <= 0:
            return 0
        else:
            neighbors = [[1, 0],  # down
                         [0, -1],  # left
                         [0, 1],  # right
                        ]
            mask = np.zeros_like(self.state)
            mask[row, col] = 2 # Starting from the end of the pipe, we find formation continuity looking laterally and downward (not up!)
            while 2 in mask:
                for z, x in np.transpose((mask==2).nonzero()):
                    mask[z, x] = 1
                    for dz, dx in neighbors:
                        if mask[z + dz, x + dx] == 0:
                            if self.state[z + dz, x + dx] > 0:
                                mask[z + dz, x + dx] = 2
                            else:
                                mask[z + dz, x + dx] = -1
            self.state[(mask==1)] = 0
            self.state[row, col] = -2 # Flag again last bit location
            return np.sum(mask==1)