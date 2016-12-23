"""
Curve-based optimization
"""
from __future__ import division
import numpy as np

def find_elbow(curve):
    nPoints = np.arange(len(curve))

    coords = np.concatenate([[nPoints, curve]]).T

    # Normalize first-last line vector
    lineVec = coords[-1,:]-coords[0,:]
    lineVecN = lineVec / np.sqrt(np.sum(lineVec**2))

    # Distance from all points to first point
    vecFromFirst = np.array(map(lambda x: x-coords[0,:], coords))

    # Calc distance to the line
    scalarProduct = np.sum(vecFromFirst*np.tile(lineVecN, (len(nPoints), 1)), axis=1)
    vecFromFirstPar = np.dot(scalarProduct.reshape(-1,1), lineVecN.reshape(1,-1))
    vecToLine = vecFromFirst-vecFromFirstPar

    # Distance to line is norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine**2, axis=1))

    return np.argmax(distToLine)
