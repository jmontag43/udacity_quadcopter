import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
    def euclid_dist(self, p1, p2):
        return np.linalg.norm(p1-p2)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 2*self.sim.v[2] - self.euclid_dist(self.sim.pose[:3], self.target_pos)
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            out_of_bounds = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reached_target = self.euclid_dist(self.target_pos, self.sim.pose[:3]) <= 1
            
            reward += self.get_reward() 
            
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        
        if out_of_bounds:
            reward = reward - 50
        
        if reached_target:
            reward = reward + 50
            
        return next_state, reward, out_of_bounds or reached_target

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state