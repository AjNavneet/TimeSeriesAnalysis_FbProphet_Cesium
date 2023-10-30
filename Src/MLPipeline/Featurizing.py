import numpy as np
from cesium import featurize

# Define a class for feature extraction using cesium
class Featurizing:

    # Feature: Mean of the signal
    def mean_signal(self, t, m, e):
        return np.mean(m)

    # Feature: Standard Deviation of the signal
    def std_signal(self, t, m, e):
        return np.std(m)

    # Feature: Mean of the squared signal values
    def mean_square_signal(self, t, m, e):
        return np.mean(m ** 2)

    # Feature: Absolute differences between consecutive signal values
    def abs_diffs_signal(self, t, m, e):
        return np.sum(np.abs(np.diff(m)))

    # Function to extract features from the time series data
    def exec(self, cs_df):
        # Define the features to extract
        guo_features = {
            "mean": self.mean_signal,
            "std": self.std_signal,
            "mean2": self.mean_square_signal,
            "abs_diffs": self.abs_diffs_signal,
        }

        # Use the cesium library to featurize the time series data
        features = featurize.featurize_time_series(times=cs_df["ts"],
                                                  values=cs_df["y"],
                                                  errors=None,
                                                  features_to_use=list(guo_features.keys()),
                                                  custom_functions=guo_features)

        # Return the extracted features
        return features
