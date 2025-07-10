# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  EventDetector.py                                            ║
# ║  Package      :  magprime                                                    ║
# ║  Author       : Matthew G. Finley  <matthew.g.finley@nasa.gov>               ║
# ║                 Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>                 ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-05-22                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Implementation of RUDE described in 'Generalized Time-Series ║
# ║  Analysis for In Situ Spacecraft Observations: Anomaly Detection and Data    ║
# ║  Prioritization Using Principal Components Analysis and Unsupervised         ║
# ║  Clustering' by Finley et al., Earth and Space Science (2024). Implementation║ 
# ║  of RUDER described in forthcoming manuscript.                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
from sklearn import svm

class _RecursivePCA():
    # Methodology adapted from 'Process Monitoring Approach using Fast Moving Window PCA'
    # by Wang et al., 2005 (Industrial & Engineering Chemistry Research)
    def __init__(self):
        self.initialized = False
        self.correlation_matrix = None
        self.standard_deviation_matrix = None
        self.mean_matrix = None
        self.oldest_sample = None
        self.initialization_length = None
        self.standard_deviation_vector = None
    
    def initialize_model(self, initial_data):
        # compute mean and standard dev matrices
        b_k = np.mean(initial_data, axis=0)
        std_k = np.std(initial_data, axis=0)
        sigma_k = std_k * np.eye(len(b_k))

        # scale initial data to zero mean and unit std
        scaled_data = (initial_data - b_k) / np.std(initial_data, axis=0)

        # compute correlation matrix
        r_k = (1 / (np.shape(scaled_data)[0] - 1)) * scaled_data.transpose() @ scaled_data

        self.initialized = True
        self.standard_deviation_matrix = sigma_k
        self.correlation_matrix = r_k
        self.mean_matrix = b_k
        self.initialization_length = np.shape(initial_data)[0]
        self.standard_deviation_vector = std_k

    def iterate_model(self, new_sample, old_sample):
        if self.initialized == False:
            raise Exception('Model is not initialized. Please use _RecursivePCA.initialize_model() to initialize before iterating the model.')
        else:
            # load in some variables for notational convenience
            initialization_length = self.initialization_length
            b_k = self.mean_matrix
            r_k = self.correlation_matrix
            sigma_k = self.standard_deviation_matrix
            std_k = self.standard_deviation_vector

            # step one: remove oldest sample
            b_hat = (1 / (initialization_length - 1)) * (initialization_length * b_k - old_sample)

            # difference between means
            delta_b_hat = b_k - b_hat

            # scale discarded sample
            x_k = np.linalg.inv(sigma_k) @ (old_sample - b_k)

            # bridge over matrix 1 and 2
            r_star = r_k - np.linalg.inv(sigma_k) @ np.outer(delta_b_hat, delta_b_hat) @ np.linalg.inv(sigma_k) - (1 / (initialization_length - 1)) * x_k @ x_k.transpose()

            # step two: add new sample
            b_kp = (1 / initialization_length) * ((initialization_length - 1) * b_hat + new_sample)

            # difference between means
            delta_b_kp = b_kp - b_hat

            # standard deviation of matrix 3
            std_kp = std_k + delta_b_kp**2 - delta_b_hat**2 + ((new_sample - b_kp)**2 - (old_sample - b_k)**2) / (initialization_length - 1)
            sigma_kp = std_kp * np.eye(len(b_kp))

            # scale the new sample
            x_kp = np.linalg.inv(sigma_kp) @ (new_sample - b_kp)

            # correlation matrix of matrix 3
            r_kp = np.linalg.inv(sigma_kp) @ sigma_k @ r_star @ sigma_k @ np.linalg.inv(sigma_kp) + np.linalg.inv(sigma_kp) @ delta_b_kp @ delta_b_kp.transpose() + (1 / (initialization_length - 1)) * x_kp @ x_kp.transpose()

            self.correlation_matrix = r_kp
            self.mean_matrix = b_kp
            self.standard_deviation_matrix = sigma_kp
            self.standard_deviation_vector = std_kp

    def compute_pcs(self, n_components=2):
        # eigh used here since it's faster, but assumes that the matrix is symmetric, 
        # which our correlation matrix has to be
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
        # need to flip since they're sorted in ascending order based on the eigenvalues
        principal_components = np.fliplr(eigenvectors)

        return principal_components[:,:n_components] # only output the desired number of components

class _TrajectoryMatrix():
    def __init__(self, x=None):
        self.x = x
    
    def update_single_point(self, new_point):
        # This expects columns of time series data with length rows
        n_rows = np.shape(self.x)[0]
        n_cols = np.shape(self.x)[1]

        # make temp copy of x
        temp_x = np.copy(self.x)

        for row in range(n_rows-1):
            temp_x[row,:] = self.x[row+1,:]
        for col in range(n_cols-1):
            temp_x[n_rows-1,col] = self.x[0,col+1]
        temp_x[n_rows-1,n_cols-1] = new_point

        self.x = temp_x

    def update_row(self, new_row):
        temp_x = np.copy(self.x)
        old_row = temp_x[0,:]
        temp_x = np.vstack([temp_x, new_row])

        self.x = temp_x[1:,:]

        return old_row

    def from_timeseries(self, time_series, window_length=66):
        trajectory_matrix = [time_series[i:i+window_length] \
                             for i in range(0,int(window_length * np.floor(len(time_series)/window_length)),window_length)] 
                            # The floor math ensures that there are no 'short' intervals < interval_length
        self.x = np.array(trajectory_matrix)
    
class DataStream():
    def __init__(self, filename=None):
        self.filename = filename
    def stream(self, filename=None):
        # yield a new data sample from a file
        if filename:
            self.filename = filename
        for row in open(filename, 'r'):
            yield row

def RUDER(window_length, initialization_length, data_stream: DataStream, col_n, filename, nu=0.1):
    '''
    Inputs: 
        window_length: Defines the length of the window used to construct the reduced trajectory matrix
        initialization_length: Defines the number of windows used in the computation of the PCA model 
        data_stream: DataStream class object to read data from; assumes a text file with rows being observations, columns as features (only one row for header)
        col_n: Column number in the DataStream text file to use
        filename: Filename that will be used to output computed anomaly score
        nu: Nu value to use in OC-SVM
    ''' 
    # step one: load initial data and perform PCA intialization
    anomaly_weights = []
    data = []
    data_gen = data_stream.stream(data_stream.filename)
    next(data_gen) # iterate past the header
    for i in range(window_length*initialization_length): # get initialization data
        data.append(float(next(data_gen).split(sep=',')[col_n])) # This gets the C component of the Swarm NEC data
    traj = _TrajectoryMatrix()
    traj.from_timeseries(data, window_length=window_length)
    traj_data = traj.x

    rpca_model = _RecursivePCA()
    rpca_model.initialize_model(traj_data)

    # step two: generate first set of principal component projections and classify with ocsvm
    principal_components = rpca_model.compute_pcs(n_components=2)
    # centering the data here ensures that the values output are the same as those
    # output by the sklearn PCA algorithm; however, there's still ambiguity in the 
    # sign of the eigenvectors output (i.e., v = -v)
    fit_data = (traj_data - rpca_model.mean_matrix) @ principal_components

    ocsvm = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    ocsvm.fit(fit_data)
    y_pred = ocsvm.predict(fit_data)
    y_pred[y_pred < 0] = 0 # Floor at zero for easy summation analysis

    # step three: generate flags for each window of data
    anomaly_flags = np.zeros(len(data))
    interval_start = 0
    interval_stop = window_length
    for i in y_pred:
        if i != 1:
            anomaly_flags[interval_start:interval_stop] += 1
        interval_start = interval_stop
        interval_stop = interval_stop + window_length
    anomaly_weights.append(anomaly_flags)
    anomaly_weights = np.array(anomaly_weights[0])

    # step four: iteratively load data from data_stream
    # when enough data to add a new window is loaded, iterate model
    iteration_num = 1
    with open(filename, 'w') as f:
        while True: # This is probably bad practice, but this will continue to grab a new chunk of data until it breaks
            try:
                print(f'Iteration: {iteration_num}', end='\r')
                new_data = []
                for i in range(window_length):
                    new_data.append(float(next(data_gen).split(sep=',')[col_n]))
                old_data = traj.update_row(new_data)
                new_traj = traj.x
                rpca_model.iterate_model(new_data, old_data)
                new_principal_components = rpca_model.compute_pcs(n_components=2)
                new_fit_data = (new_traj - rpca_model.mean_matrix) @ new_principal_components

                ocsvm.fit(new_fit_data)
                y_pred = ocsvm.predict(new_fit_data)
                y_pred[y_pred < 0] = 0

                new_anomaly_flags = np.zeros(len(data))
                interval_start = 0
                interval_stop = window_length
                for i in y_pred:
                    if i != 1:
                        new_anomaly_flags[interval_start:interval_stop] += 1
                    interval_start = interval_stop
                    interval_stop = interval_stop + window_length

                anomaly_weights[window_length:] += new_anomaly_flags[:-window_length]
                anomaly_weights = np.concatenate((anomaly_weights, new_anomaly_flags[-window_length:]))
                for element in anomaly_weights[0:window_length]:
                    f.write(str(element) + '\n')
                anomaly_weights = np.delete(anomaly_weights, np.s_[0:window_length])
                iteration_num += 1
            except:
                print('Finished Processing...')
                break

    return anomaly_weights

if __name__ == '__main__':
    pass