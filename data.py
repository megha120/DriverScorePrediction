"""Data utils librairy."""
import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Columns names in CSV file.
TRIP_DATA_COLUMNS = [
    'time',
    'speed',
    'shift',
    'engine_Load',
    'car_accel',
    'rpm',
    'pitch',
    'lateral_acceleration',
    'passenger_count',
    'car_load',
    'ac_status',
    'window_opening',
    'radio_volume',
    'rain_intensity',
    'visibility',
    'driver_wellbeing',
    'driver_rush'
]

# Numerical features used for analysis.
NUMERICAL_FEATURES = [
    'speed',
    'car_accel',
    'lateral_acceleration',
    'rpm',
    'pitch',
    'shift'
]

# Categorical features used for analysis.
CATEGORICAL_FEATURES = [
    'driver_rush',
    'visibility',
    'rain_intensity',
    'driver_wellbeing'
]


def load_dataset_as_dataframe(data_dir_path: str):
    """Load dataset as a panda dataframe.

    Parameters
    ----------
    data_dir_path : str
        Path to data dir.

    Returns
    -------
    pandas.DataFrame
        Dataset in DataFrame format.

    """
    # Initialize returned output
    dataframe_list = []
    max_time = 0

    # Check file exists
    if not os.path.exists(data_dir_path):
        raise FileNotFoundError('Path "{}" not found'.format(data_dir_path))

    for filename in os.listdir(data_dir_path):
        if fnmatch.fnmatch(filename, 'fileID*_ProcessedTripData.csv'):
            # Load dataset and initialise Dataframe
            df = pd.read_csv(os.path.join(data_dir_path, filename), header=None)
            df.columns = TRIP_DATA_COLUMNS

            # Assign time to keep record continuity
            df.time = df.time + max_time
            max_time = df.time.max()
            dataframe_list.append(df)

    # Merge all together
    concatenated_df = pd.concat(dataframe_list, ignore_index=True)

    # Assign idx based on new index. Useful to brakedown events later on.
    concatenated_df['idx'] = concatenated_df.index

    return concatenated_df


def plot_feature_distributions(
                                df: pd.DataFrame,
                                figsize: tuple,
                                features=TRIP_DATA_COLUMNS):
    """Plot features distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the features.
    figsize : tuple
        Matplotlib figsize param.
    features : list
        List of features.

    """
    nrows = len(features)//2 + 1 if len(features) % 2 else len(features)//2

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    fig.delaxes(axes[nrows-1][1])  # Removes last empty subplot

    for i, col_name in enumerate(features):
        row = i // 2
        col = i % 2
        sns.distplot(df[col_name], ax=axes[row][col])


def filter_acceleration_entries(df: pd.DataFrame, threshold=2, above=True):
    """Extract acceleration events out of dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing driving measures.
    threshold : int
        Treshold for significant acceleration in m/s^2.
    above : boolean
        Should the acceleration be above or bellow threshold.

    Returns
    -------
    pd.DataFrame
        Dataframe containing only acceleration events driving measures.

    """
    return df[df.car_accel > threshold] if above else df[df.car_accel < threshold]


def extract_events(df: pd.DataFrame, interval=2):
    """Extract a list of brake event time series.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing driving measures.
    interval : int
        Time interval in seconds where 2 events are considered distincts.

    Returns
    -------
    list
        List of DataFrame containing events time series.

    """
    # prepare output
    events = []

    # Measures are taken at a 100Hz frequency.
    boundaries = (df.time - df.time.shift()) > (interval)
    new_boundaries = boundaries.reset_index()

    # boundaries_indexes = new_boundaries[new_boundaries['idx']].index
    boundaries_indexes = new_boundaries[new_boundaries['time']].index

    for i in range(len(boundaries_indexes)):
        min_bound = 0 if i == 0 else boundaries_indexes[i-1]
        max_bound = boundaries_indexes[i]

        if len(df[min_bound:max_bound]) > 10:
            events.append(df[min_bound:max_bound])

    return events


def calculate_event_metrics(event_df: pd.DataFrame):
    """Calculate metrics for given driving event.

    Parameters
    ----------
    event_df : pd.DataFrame
        Driving event dataframe containing all car measures.

    Returns
    -------
    pd.Series
        Series containing event metrics.

    """
    # Build numerical data metrics
    num_metrics = [event_df[feature].describe().add_prefix(feature + '_')
                   for feature in NUMERICAL_FEATURES]
    num_metrics_ds = pd.concat(num_metrics, axis=0)

    # Build categorical data metrics
    cat_metrics = [event_df[feature].mean()
                   for feature in CATEGORICAL_FEATURES]
    cat_metrics_ds = pd.Series(cat_metrics, index=CATEGORICAL_FEATURES)

    # Merge numerical and categorical metrics
    metrics_ds = pd.concat([num_metrics_ds, cat_metrics_ds], axis=0)

    # Clean duplicated 'count' columns and rename labels
    duplicated_cols = [col + '_count' for col in NUMERICAL_FEATURES[1:]]
    metrics_ds.drop(labels=duplicated_cols, inplace=True)
    metrics_ds.rename({'speed_count': 'observations'}, inplace=True)
    metrics_ds.rename(lambda x: x.replace('%', ''), inplace=True)

    return metrics_ds


def get_events_metrics(events: list):
    """Return events metrics dataframe.

    Parameters
    ----------
    events : list
        List of driving events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all event metrics.

    """
    metrics = [calculate_event_metrics(event) for event in events]

    # Format dataframe
    metrics_df = pd.concat(metrics, axis=1).T.reset_index()

    metrics_df.drop(columns=['index'], inplace=True)

    return metrics_df


def rescale_events_metrics(metrics_df: pd.DataFrame):
    """Rescale numerical metrics features.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataset.

    Returns
    -------
    pd.DataFrame
        Rescaled metrics dataframe.

    """
    # metrics dataset column suffixes
    col_name_suffix = [
        'mean',
        'std',
        'min',
        '25',
        '50',
        '75',
        'max'
    ]
    # Build metrics dataset columns names.
    num_feat_col_names = [feature + '_' + suffix
                          for feature in NUMERICAL_FEATURES
                          for suffix in col_name_suffix]

    # Rescale numerical features.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(metrics_df[num_feat_col_names])
    scaled_df = pd.DataFrame(scaled_data, columns=num_feat_col_names)

    return pd.concat([scaled_df, metrics_df[CATEGORICAL_FEATURES]], axis=1)


def calculate_harsh_braking_ratio(
                    brake_df: pd.DataFrame,
                    threshold=-3,
                    verbose=False):
    """Calculate ratio of harsh braking.

    Parameters
    ----------
    brake_df : pd.DataFrame
        DataFrame containing braking metrics.
    threshold : int
        Car acceleration threshold.
    verbose : boolean
        Print out result.

    Returns
    -------
    float
        Ratio of harsh brakings.

    """
    num_harsh_brakings = len(brake_df[brake_df.car_accel_25 < threshold])
    num_brakings = len(brake_df)

    harsh_braking_ratio = round(num_harsh_brakings / num_brakings, 3)

    if verbose:
        print('---------------------------')
        print('Harsh braking ratio : ', harsh_braking_ratio)
        print('Harsh braking count : ', num_harsh_brakings)
        print('Total braking count : ', num_brakings)
        print('---------------------------')

    return harsh_braking_ratio


def calculate_harsh_acceleration_ratio(
                    acceleration_df: pd.DataFrame,
                    threshold=-3,
                    verbose=False):
    """Calculate ratio of harsh braking.

    Parameters
    ----------
    acceleration_df : pd.DataFrame
        DataFrame containing acceleration metrics.
    threshold : int
        Car acceleration threshold.
    verbose : boolean
        Print out result.

    Returns
    -------
    float
        Ratio of harsh accelerations.

    """
    num_harsh_accel = len(acceleration_df[acceleration_df.car_accel_75 > threshold])
    num_accelerations = len(acceleration_df)

    harsh_accel_ratio = round(num_harsh_accel / num_accelerations, 3)

    if verbose:
        print('---------------------------')
        print('Harsh acceleration ratio : ', harsh_accel_ratio)
        print('Harsh acceleration count : ', num_harsh_accel)
        print('Total acceleration count : ', num_accelerations)
        print('---------------------------')

    return harsh_accel_ratio
