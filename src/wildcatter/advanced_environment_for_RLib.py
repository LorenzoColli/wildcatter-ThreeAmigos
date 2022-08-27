"""Driller environment module."""


from __future__ import annotations

import random
from typing import Any

import numpy as np
import gym
from gym.spaces import Box, Discrete, Dict
from numpy.typing import NDArray


class AdvancedDriller(gym.Env):  # type: ignore
    """Advanced driller environment for RL.
    
    Actor choses well location and drills into a 2-D cross-section
    of the subsurface.
    
    Configuration parameters:
    - model_type [str]: determines subsurface characteristics
        - random --- each cell is a coin flip.
        - random_pockets --- a small number of valuable targets at 
          random locations. Targets are squat triangles in shape.
          Size of targets depend on size of env. Number depends on
          number of available wells.
        - from_csv --- subsurface grid is provided from an external
          csv file.
    - nrow [int]: number of rows (i.e., depths) of subsurface grid.
      Needed if model_type is random.
    - ncol [int]: number of columns (i.e., lateral extent) of
      subsurface grid. Needed if model_type is random.
    - model_path [str]: path to csv file.
    - delim [str]: value delimiter of the csv file.
    - available_pipe [int]: number of pipe lengths available for
      drilling campaign. I.e., number of cells that can be drilled.
    - available_wells [int]: number of wells that can be opened.
    - oil_price [float]: weight for rewarding oil extraction.
    - relocation_cost [float]: cost of moving the drill to a new
      well. Penalizes extra distance between wells, giving negative
      reward to action.
    - drilling_cost [float]: base cost of drilling one cell. Gives
      negative reward to action.
    - drilling_depth_markup [float]: cost of drilling increases
      linearly with depth:
          drilling_cost + depth * drilling_depth_markup.
    """

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize environment with config dictionary."""
        
        try:
            self._rng = np.random.default_rng(env_config["seed"])
        except KeyError:
            self._rng = np.random.default_rng()
        
        self.model_type = env_config["model_type"]
                
        if self.model_type == "random":
            self.nrow = env_config["nrow"]
            self.ncol = env_config["ncol"]
        elif self.model_type == "random_pockets":
            self.nrow = env_config["nrow"]
            self.ncol = env_config["ncol"]
            self.model = np.zeros((self.nrow, self.ncol), dtype=int)
        elif self.model_type == "from_csv":
            self.initial_model = np.loadtxt(
                env_config["model_path"],
                delimiter=env_config["delim"],
                dtype="int",
            )
            self.initial_model[self.initial_model>1] = 1
            self.initial_model[self.initial_model<-10] = -10
            self.nrow, self.ncol = self.initial_model.shape
        else:
            raise NotImplmentedError("Model Type Unknown")
        
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
        self.max_avail_actions = 4 + self.ncol-2 + 1
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
        # of 3 directions either in sequence (i.e.: 1-2-3, 2-3-0, 3-0-1, 0-1-2)
        # or in reverse sequence (e.g., 3-2-1 or 0-3-2)

        # Observation space augments the actual observed state with an action mask
        # that is used by the custom model to prevent the actor from taking
        # illegal moves
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.max_avail_actions, ), dtype="int"),
            "avail_actions": Box(-1, 1, shape=(self.max_avail_actions, ), dtype="int"),
            "state": Box(low=-10, high=10, shape=(self.nrow, self.ncol), dtype="int"),
        })
        
        self.reset()
        
    def action_masks(self) -> list:
        """Return list of bool specifying legal actions."""
        if self.bit_location == None:
            # Drilling impossible: either start a well (if available) or end campaign
            drilling = [False] * 4
            new_well = [self.wells_started < self.available_wells and ground == -10 for ground in self.state[0, 1: self.ncol-1]]
        else:
            # Drilling possible if there's pipe available. Just prevent drilling into faults, pipes and model boundary
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
        
        # Sanity check
        if len(drilling + new_well + end_campaign) != self._size_action_space:
            print("len(drilling + new_well + end_campaign):",len(drilling + new_well + end_campaign),len(drilling),len(new_well),len(end_campaign))
            print("self._size_action_space:",self._size_action_space)
            raise Exception
        return drilling + new_well + end_campaign

    def step(  # noqa: C901
        self, action: int
    ) -> tuple[dict, int, bool, dict[str, Any]]:
        """Take step based on action.
        
        Modify state and return tuple with:
        - new state as observed by agent,
        - reward for action,
        - done boolean flag,
        - info dict.
        """
        done = False
        info: dict[str, Any] = {}
        
        # If ray tries to interact with the environment without going through our custom model
        legal_actions = self.action_masks()
        if not legal_actions[action]:
            obs = dict({
                "action_mask": np.asarray(self.action_masks(), dtype=int),
                "avail_actions": np.ones(self.max_avail_actions),
                "state": self.state,
            })

            return obs, -100, done, info

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
        
        obs = dict({
            "action_mask": np.asarray(self.action_masks(), dtype=int),
            "avail_actions": np.ones(self.max_avail_actions),
            "state": self.state,
        })

        return obs, reward, done, info

    def update_state(self) -> None:
        """Update state method.
        
        Flag old drill bit location as pipe.
        Flag new drill bit location as such.
        """
        self.state[self.state == -3] = -2
        try:
            newrow, newcol = self.bit_location
            self.state[newrow, newcol] = -3
        except TypeError: # No new well exists yet
            pass

    def render(self) -> None:
        """Gym environment rendering."""
        raise NotImplementedError("No renderer implemented yet.")

    def reset(self) -> dict:
        """Reset the status of the environment."""
        if self.model_type == "random":
            self.model = self._rng.integers(low=0, high=2, size=(self.nrow, self.ncol))
        elif self.model_type == "random_pockets":
            self.model[:] = 0
            oil_to_bury = (self.nrow-2)*(self.ncol-1)//10
            # Divide oil between pockets. Make as many pockets as available wells (to keep max reward ~constant)
            oil_in_pocket = np.zeros(self.available_wells)
            oil_remaining = oil_to_bury
            for i in range(self.available_wells-1):
                max_oil=min(oil_remaining-(self.available_wells-i-1)+1,oil_remaining//2)
                oil_in_pocket[i] = self._rng.integers(low=1, high=oil_remaining-(self.available_wells-i-1)+1)
                oil_remaining -= oil_in_pocket[i]
            oil_in_pocket[-1] = oil_remaining
            sorted_pockets = np.sort(oil_in_pocket)
            for oil in reversed(sorted_pockets):
                pocket_width = int(np.ceil(np.sqrt(oil) * 6 / 3))
                pocket_height = int(np.sqrt(oil) * 3 // 6)
                # Choose upper-left corner at random, preventing obvious out-of-bounds
                leftmost_col = self._rng.integers(low=1, high=self.ncol - pocket_width -1)
                upper_row = self._rng.integers(low=1, high=self.nrow - (pocket_height +3) -1)
                n_iter = 0
                while (1 in self.model[upper_row:upper_row+pocket_height+3,
                                     leftmost_col:leftmost_col+pocket_width]
                       and n_iter < 100):
                    n_iter += 1
                    # keep drawing until there's no intersection with other rectangles
                    leftmost_col = self._rng.integers(low=1, high=self.ncol - pocket_width -1)
                    upper_row = self._rng.integers(low=1, high=self.nrow - (pocket_height +3) -1)
                # Draw pocket cap
                self.model[upper_row,leftmost_col+int(np.floor(pocket_width/2))] = 1
                # Draw bulk of pocket
                factor = 0.25
                rightmost_col = leftmost_col + pocket_width
                for row in range(1,pocket_height+1):
                    empty = int(np.floor(pocket_width * factor))
                    self.model[upper_row+row, leftmost_col+empty : rightmost_col-empty] = 1
                    factor = factor * 0.5
                
                # Deal with remaining oil
                oil_remaining = oil - np.sum(self.model[upper_row:upper_row+pocket_height, leftmost_col:rightmost_col])
                if 0<oil_remaining<=pocket_width:
                    self.model[upper_row+pocket_height+1, leftmost_col : int(rightmost_col -(pocket_width-oil_remaining))] = 1
                elif oil_remaining>pocket_width:
                    self.model[upper_row+pocket_height+1, leftmost_col : rightmost_col] = 1
                    self.model[upper_row+pocket_height+2, leftmost_col : int(rightmost_col -(2*pocket_width-oil_remaining))] = 1
        else:
            self.model = self.initial_model.copy()
        
        self.model[0] = self.model[self.nrow-1] = self.model[:, 0] = self.model[:, self.ncol-1] = -10
        #self.model[1, 1:self.ncol-2] = 0

        self.surface_hole_location = None
        self.state = self.model.copy()
        self.bit_location = self.surface_hole_location
        self.trajectory = []
        self.production = 0
        self.expenses = 0
        self.pipe_used = 0
        self.wells_started = 0
        
        obs = dict({
            "action_mask": np.asarray(self.action_masks(), dtype=int),
            "avail_actions": np.ones(self.max_avail_actions),
            "state": self.state,
        })
        return obs

    def extract(self) -> float:
        """Remove oil and return amount.
        
        Topologically connected oil is extracted up gravity and 
        removed from state. Total amount of oil extracted is
        returned.
        """
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