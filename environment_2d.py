'''
Modified from exercise template from osrobotics.org
https://github.com/crigroup/osr_course_pkgs/blob/master/osr_examples/scripts/environment_2d.py

Modifications:
Added type hinting
Added collision check overload for TriangularObstacle

'''
from typing import Union
from numbers import Number

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt

class TriangularObstacle:
    def __init__(self, x0: Number, y0: Number, x1: Number, y1: Number, x2: Number, y2: Number):
        # For contains check, determine half plane from each edge (uses cross product)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.A = np.zeros((3,2))
        self.C = np.zeros(3)

        a = x1 - x0
        b = y1 - y0
        c = x2 - x0
        d = y2 - y0
        if -b*c + a*d > 0:
            self.A[0, :] = -b, a
        else:
            self.A[0, :] = b, -a
        self.C[0] = np.dot(self.A[0, :], np.array([x0,y0]))

        a = x2 - x1
        b = y2 - y1
        c = x0 - x1
        d = y0 - y1
        if -b*c + a*d > 0:
            self.A[1, :] = -b, a
        else:
            self.A[1, :] = b, -a
        self.C[1] = np.dot(self.A[1, :], np.array([x1,y1]))

        a = x0 - x2
        b = y0 - y2
        c = x1 - x2
        d = y1 - y2
        if -b*c + a*d > 0:
            self.A[2, :] = -b, a
        else:
            self.A[2, :] = b, -a
        self.C[2] = np.dot(self.A[2, :], np.array([x2,y2]))


    def contains(self, x: Number, y: Number) -> bool:
        r = np.dot(self.A, np.array([x,y])) - self.C
        return all([i>0 for i in r])
    
    def contains_np(self, x: np.array) -> bool:
        r = np.dot(self.A, x) - self.C
        return np.all(r > 0)

    def plot(self) -> None:
        plt.plot([self.x0,self.x1], [self.y0,self.y1], "r" , linewidth = 2)
        plt.plot([self.x1,self.x2], [self.y1,self.y2], "r" , linewidth = 2)
        plt.plot([self.x2,self.x0], [self.y2,self.y0], "r" , linewidth = 2)

        #plt.plot([self.x0,self.x0 + self.A[0,0]], [self.y0,self.y0 + self.A[0,1]], "b" , linewidth = 1)
        #plt.plot([self.x1,self.x1 + self.A[1,0]], [self.y1,self.y1 + self.A[1,1]], "b" , linewidth = 1)
        #plt.plot([self.x2,self.x2 + self.A[2,0]], [self.y2,self.y2 + self.A[2,1]], "b" , linewidth = 1)

        #plt.quiver(self.x0, self.y0, np.array([self.x0 + self.A[0,0]]), np.array([self.y0 + self.A[0,1]]))
        plt.quiver([self.x0, self.x1, self.x2], 
        [self.y0, self.y1, self.y2], 
        [self.x1 - self.x0, self.x2 - self.x1, self.x0 - self.x2],
        [self.y1 - self.y0, self.y2 - self.y1, self.y0 - self.y2],
        scale = 1, angles='xy', scale_units='xy')

        plt.quiver([self.x0, self.x1, self.x2], 
        [self.y0, self.y1, self.y2], 
        [self.x0 + self.A[0,0], self.x1 + self.A[1,0], self.x2 + self.A[2,0]],
        [self.y0 + self.A[0,1], self.y1 + self.A[1,1], self.y2 + self.A[2,1]],
        scale = 1, angles='xy', scale_units='xy')

class EnvironmentFast:
    """ Fast Collision Environment

        Modified from Environment for collision fast lookup
    """
    def __init__(self, size_x: Number, size_y: Number, n_obs: int):
        self.size_x = size_x
        self.size_y = size_y
        self.obs = []

        As = []
        Cs = []

        for i in range(n_obs):
            x0 = np.random.rand()*size_x
            y0 = np.random.rand()*size_y
            x1 = np.random.rand()*size_x
            y1 = np.random.rand()*size_y
            x2 = np.random.rand()*size_x
            y2 = np.random.rand()*size_y
            tri = TriangularObstacle(x0, y0, x1, y1, x2, y2)
            self.obs.append(tri)
            As.append(tri.A)
            Cs.append(tri.C)

        self.As = np.array(As)
        self.Cs = np.array(Cs)

    def check_collision(self, x: Number, y: Number) -> bool:
        r = np.matmul(self.As, np.array([x,y])) - self.Cs
        return np.any(np.all(r > 0, 1))

    def check_collision_np(self, x: np.array) -> bool:
        """ Check collision using numpy array as parameter
        
        Parameters
        ----------
        x : np.array
            x.shape == (2,)
        """
        assert x.shape == (2,)
        r = np.matmul(self.As, x) - self.Cs
        return np.any(np.all(r > 0, 1)) 

    def random_query(self) -> Union[tuple, None]:
        max_attempts = 100
        found_start = False
        found_goal = False
        for i in range(max_attempts):
            x_start = np.random.rand()*self.size_x
            y_start = np.random.rand()*self.size_y
            if not self.check_collision(x_start, y_start):
                found_start = True
                break
        for i in range(max_attempts):
            x_goal = np.random.rand()*self.size_x
            y_goal = np.random.rand()*self.size_y
            if not self.check_collision(x_goal, y_goal):
                found_goal = True
                break
        if found_start and found_goal:
            return x_start, y_start, x_goal, y_goal
        else:
            return None

    def plot(self, fig: Union[matplotlib.figure.Figure, None] = None) -> matplotlib.figure.Figure:
        if fig is None:
            fig = plt.figure()
        plt.figure(fig.number) 
        plt.plot([0, self.size_x, self.size_x, 0, 0], [0, 0, self.size_y, self.size_y, 0], "k", linewidth = 2)
        for ob in self.obs:
            ob.plot()

        ax = fig.gca()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.show()
        return fig
        

    def plot_query(self, x_start, y_start, x_goal, y_goal) -> None:
        plt.plot([x_start], [y_start], "bs", markersize = 8)
        plt.plot([x_goal], [y_goal], "y*", markersize = 12)

class Environment:
    def __init__(self, size_x: Number, size_y: Number, n_obs: int):
        self.size_x = size_x
        self.size_y = size_y
        self.obs = []
        for i in range(n_obs):
            x0 = np.random.rand()*size_x
            y0 = np.random.rand()*size_y
            x1 = np.random.rand()*size_x
            y1 = np.random.rand()*size_y
            x2 = np.random.rand()*size_x
            y2 = np.random.rand()*size_y
            self.obs.append(TriangularObstacle(x0, y0, x1, y1, x2, y2))

    def check_collision(self, x: Number, y: Number) -> bool:
        for ob in self.obs:
            if ob.contains(x, y):
                return True
        return False

    def random_query(self) -> Union[tuple, None]:
        max_attempts = 100
        found_start = False
        found_goal = False
        for i in range(max_attempts):
            x_start = np.random.rand()*self.size_x
            y_start = np.random.rand()*self.size_y
            if not self.check_collision(x_start, y_start):
                found_start = True
                break
        for i in range(max_attempts):
            x_goal = np.random.rand()*self.size_x
            y_goal = np.random.rand()*self.size_y
            if not self.check_collision(x_goal, y_goal):
                found_goal = True
                break
        if found_start and found_goal:
            return x_start, y_start, x_goal, y_goal
        else:
            return None

    def plot(self, fig: Union[matplotlib.figure.Figure, None] = None) -> matplotlib.figure.Figure:
        if fig is None:
            fig = plt.figure()
        plt.figure(fig.number) 
        plt.plot([0, self.size_x, self.size_x, 0, 0], [0, 0, self.size_y, self.size_y, 0], "k", linewidth = 2)
        for ob in self.obs:
            ob.plot()

        ax = fig.gca()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.show()
        return fig

    def plot_query(self, x_start, y_start, x_goal, y_goal) -> None:
        plt.plot([x_start], [y_start], "bs", markersize = 8)
        plt.plot([x_goal], [y_goal], "y*", markersize = 12)
