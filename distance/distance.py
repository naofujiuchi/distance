# -*- coding: utf-8 -*-
# Naomichi Fujiuchi (naofujiuchi@gmail.com), August 2022
# This is an original work by Fujiuchi (MIT license).
import copy
import numpy as np
# from sklearn.metrics import DistanceMetric

class Initmatch():

    def __init__(self, X, Y, penalty = 0):
        """
        Parameters
        ----------
        penalty: float
            The magnitude of penalty for "element lack".
            There is penalty, which magnitude is determined by the parameter "penalty", for "element lack".
            If there is a numeric data for an element of one data and there is no numeric data (nan) for an element of the other data, then the case is called "element lack".
        """
        self.X = copy.deepcopy(X) # 1D array of test data
        self.Y = copy.deepcopy(Y) # 2D array of template data
        # if isnan == False: # Replace np.nan with None
        #     self.X = [None if x != x else x for x in self.X]
        #     self.Y = [[None if x != x else x for x in row] for row in self.Y]
        self.nrow = len(self.Y) # Number of template data
        self.X = [self.X for i in range(self.nrow)] # Making 2D array of test data by replicating the given 1D array test data
        statusX = [[not np.isnan(item) for item in row] for row in self.X]
        statusY = [[not np.isnan(item) for item in row] for row in self.Y]
        statusXY = [[np.sum(x) for x in zip(*rows)] for rows in zip(statusX, statusY)] # 0: no data, 1: element lack, 2: both elements available
        # statusXY = np.sum(np.array(statusX, statusY), axis = 0)
        statusXY1 = [[item == 1 for item in row] for row in statusXY]
        statusXY2 = [[item == 2 for item in row] for row in statusXY]
        statusXY12 = [[item in [1,2] for item in row] for row in statusXY]

        # # Keeping the values where the both elements are avilable. The other values are replaced with 0.
        # self.X = [[999999 if x == None else x for x in row] for row in self.X]
        # self.X = [[np.prod(x) for x in zip(*rows)] for rows in zip(self.X, statusXY2)]
        # # self.X = np.prod(np.array(self.X, statusXY2), axis = 0)
        # self.Y = [[999999 if x == None else x for x in row] for row in self.Y]
        # self.Y = [[np.prod(x) for x in zip(*rows)] for rows in zip(self.Y, statusXY2)]
        # # self.Y = np.prod(np.array(self.Y, statusXY2), axis = 0)

        # Calculating penalty
        statusXY1sum = [np.sum(x) for x in statusXY1]
        # statusXY1sum = np.sum(statusXY1, axis = 0)
        self.penalty = [x * penalty for x in statusXY1sum] # Value of penalty for each combination of X and Y

        # Calculating variance of each axis of template data (Y)
        # self.YforV = [[np.prod(x) for x in zip(*rows)] for rows in zip(self.Y, statusXY12)]
        # # self.YforV = np.prod(np.array(self.Y, statusXY12), axis = 0)
        # npYforV = np.array(self.YforV)
        # maskedYforV = np.ma.masked_where(npYforV == 0, npYforV)
        # self.V = np.var(maskedYforV, axis=0)
        self.V = [np.ma.masked_invalid(col).var() for col in zip(*self.Y)]

    def distance(self):
        # Standardized euclidean distance will be returned. 
        # = sqrt(sum((x-y)^2 / V))
        # e.g. x = [[0,1,2,3]], y = [[3,4,5,6],[6,7,8,9],[9,10,11,12]]. V of y is [6,6,6,6]. Standardized euclidean ditances between x and each y are 2.44948974, 4.89897949, 7.34846923
        # This can be test as the following program.
        # import numpy as np
        # from sklearn.neighbors import DistanceMetric
        # X = [[0,1,2,3]]
        # Y = [[3,4,5,6],[6,7,8,9],[9,10,11,12]]
        # V = [np.var(x) for x in zip(*Y)]
        # dist = DistanceMetric.get_metric('seuclidean', V=V)
        # dist.pairwise(X=X,Y=Y)
        sqXY = [[(x-y)**2 for x,y in zip(*rows)] for rows in zip(self.X, self.Y)]
        sqXYdivV = [[np.divide(x,y) for x,y in zip(row, self.V)] for row in sqXY]
        sumsqXYdivV = [np.ma.masked_invalid(row).sum() for row in sqXYdivV]
        rtsumsqXYdivV = [x**(1/2) for x in sumsqXYdivV]
        return [np.sum(x) for x in zip(rtsumsqXYdivV, self.penalty)]

# class OrganInitmatch(Initmatch):
