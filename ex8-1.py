import numpy as np
import pdb

def pseudorange(rec_pos, sat_pos, rec_clock_offset):
    """Compute the estimated psuedorange measurement
    """
    meas = np.linalg.norm(rec_pos - sat_pos) + rec_clock_offset
    return meas

# satellite ECEF positions
# distances in meters
sat_pos = np.array([[7766188.44, -21960535.34, 12522838.56],
                    [-25922679.66, -6629461.28, 31864.37],
                    [-5743774.02, -25828319.92, 1692757.72],
                    [-2786005.69, -15900725.80, 21302003.49]])

# psuedo range measurements to the sats in meters
range_meas = np.array([22228206.42, 24096139.11, 21729070.63, 21259581.09])

# form guess of state (receiver location and time offset)
num_sats = sat_pos.shape[0]
rec_pos = np.zeros(3)
rec_clock_offset = 0;
dx = np.ones(num_sats)
iter = 0
max_iter = 10

x = np.hstack((rec_pos, rec_clock_offset))

# now loop and solve
while (np.linalg.norm(dx) > 1e-6) and (iter < max_iter):
    
    rec_pos = x[ 0:3 ]
    rec_clock_offset = x[ 3 ]
    H = np.zeros((num_sats, num_sats))
    meas_error  = np.zeros(num_sats)
    
    for ii, sv in enumerate(sat_pos):
        # compute H matrix,
        los = rec_pos - sv 
        H[ ii, : ] =  np.hstack((los/np.linalg.norm(los), 1))

        # expected measurement at this location
        pr = pseudorange(rec_pos, sv, rec_clock_offset)
        meas_error[ ii ] = range_meas[ ii ] - pr

    # update estimate of state (rec_pos and clock_offset)
    dx = np.linalg.inv((H.T.dot(H))).dot(H.T).dot(meas_error)
    x = x + dx
    iter = iter+1

    

