import numpy as np
from scipy.spatial import distance

class DimensionWalk:
    def __init__(self,dimensions,difficulty = 0):
        self.dimensions = dimensions
        self.observation_space = len(dimensions)*2
        self.action_space = len(dimensions)*2
        self.reset()
        while(self.start_distance==0):
            self.reset()

    def step(self,action):
        self.prev_point = np.copy(self.current_point)
        done = False
        next_point = self.current_point+action
        is_in_bounds = self.check_bounds(next_point)
        if is_in_bounds:
            self.current_point = next_point
        self.game_map[tuple(self.prev_point)] = '='
        self.game_map[tuple(self.current_point)] = 's'
        reward = -0.1
        distance1 = distance.euclidean(self.current_point,self.finish_point)
        if distance1==0:
            reward = 50
        else:
            reward+=pow(1/distance1,2)
        if distance1 == 0:
            done = True
        return self.get_observation(), reward, done, 0

    def check_bounds(self,point):
        is_in_bounds = True
        for i in range(0, len(point)):
            if (point[i] >= self.dimensions[i] or point[i] < 0):
                is_in_bounds = False
        return is_in_bounds

    def get_observation(self):
        result=np.zeros(len(self.dimensions)*2)
        for i in range(len(self.dimensions)):
            point1 = np.copy(self.current_point)
            point2 = np.copy(self.current_point)
            point1[i]+=1
            point2[i]-=1
            if (distance.euclidean(point1, self.finish_point)==0):
                result[i*2] = 1
            else:
                result[i*2] = 1/distance.euclidean(point1, self.finish_point)
            if (self.check_bounds(point1)==False):
                result[i * 2] = -1
            if (distance.euclidean(point2, self.finish_point)==0):
                result[i*2+1] = 1
            else:
                result[i*2+1] = 1/distance.euclidean(point2, self.finish_point)
            if (self.check_bounds(point2)==False):
                result[i * 2+1] = -1
        result = result-min(result)
        if (max(result!=0)):
            result = result/max(result)
        for i in range(len(result)):
            if result[i]!=1:
                result[i] = 0
        return result

    def reset(self):
        dimensions = self.dimensions
        self.game_map = np.full(dimensions, 'o', dtype='str')
        start_point = np.zeros(len(dimensions), dtype=np.dtype(np.int32))
        finish_point = np.zeros(len(dimensions), dtype=np.dtype(np.int32))
        for i in range(0, len(dimensions)):
            start_point[i] = np.random.randint(0, dimensions[i])
            finish_point[i] = np.random.randint(0, dimensions[i])
        self.finish_point = finish_point
        self.current_point = start_point
        self.start_distance = distance.euclidean(start_point, self.finish_point)
        self.game_map[tuple(start_point)] = 's'
        self.game_map[tuple(finish_point)] = 'f'
        return self.get_observation()

env = DimensionWalk([5,5])