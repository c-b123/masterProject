import pandas


def get_window_data(dataframe: pandas.DataFrame, exclude_columns: list, window_length: str, center: bool,
                    squeezing_method: str):

    # Exclude columns
    rolling_columns = [column for column in dataframe.columns if column not in exclude_columns]

    if squeezing_method == 'mean':
        dataframe[rolling_columns] = dataframe[rolling_columns].rolling(window=window_length, center=center).mean()
    elif squeezing_method == 'sum':
        dataframe[rolling_columns] = dataframe[rolling_columns].rolling(window=window_length, center=center).sum()
    elif squeezing_method == 'max':
        dataframe[rolling_columns] = dataframe[rolling_columns].rolling(window=window_length, center=center).max()
    elif squeezing_method == 'min':
        dataframe[rolling_columns] = dataframe[rolling_columns].rolling(window=window_length, center=center).min()
    elif squeezing_method == 'median':
        dataframe[rolling_columns] = dataframe[rolling_columns].rolling(window=window_length, center=center).median()
    else:
        raise ValueError(f"Invalid squeezing method specified: {squeezing_method}")

    dataframe.dropna(inplace=True, ignore_index=True)

    return dataframe


def lag_dataframe(dataframe: pandas.DataFrame, non_lag_columns: list, num_lag_steps: int):
    """
    Lags all non-excluded columns of a pandas dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A pandas dataframe with at least one column.
    non_lag_columns : list
        List of all columns which should not get lagged.
    num_lag_steps : str
        Integer indicating the number of lagging steps.

    Returns
    -------
    pandas.DataFrame
        returns pandas dataframe

    Raises
    -------
    ValueError
        if non-existing column name is specified
    """

    lagged_dataframe = dataframe.copy()

    # Validate if non-existing columns are specified
    invalid_columns = [col for col in non_lag_columns if col not in dataframe.columns]
    if invalid_columns:
        raise ValueError(f"Invalid column(s) specified: {', '.join(invalid_columns)}")

    # Apply lagging
    for column in dataframe.columns:
        if column not in non_lag_columns:
            for lag_step in range(1, num_lag_steps + 1):
                lagged_dataframe[column] = dataframe[column].shift(lag_step)

    return lagged_dataframe


def balance_dataframe(dataframe: pandas.DataFrame, target_column: str):
    """
    Balances a pandas dataframe according to the specified target column. Requires a target column. Balancing is done
    by dropping rows of the more frequent categories. Remaining rows are selected randomly.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A pandas dataframe with at least one column.
    target_column : str
        The name of the target column.

    Returns
    -------
    pandas.DataFrame
        returns a balanced dataframe
    """

    # Count the number of occurrences
    class_counts = dataframe[target_column].value_counts()
    min_class_count = class_counts.min()

    # Balance dataframe, samples get selected randomly
    balanced_df = dataframe.groupby(target_column).apply(lambda x: x.sample(min_class_count, random_state=42))
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def add_relative_return(dataframe: pandas.DataFrame, lower_percentile: str, upper_percentile: str):
    """
    Adds three columns to a pandas dataframe. Each added column indicates whether the return is underperforming,
    neutral, or outperforming compared to all stocks in the S&P500. Requires three columns: "return" and the
    lower_percentile and upper_percentile column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A pandas dataframe with two columns. One column containing the lower percentile of return. The other containing
        the upper percentile of return.
    lower_percentile : str
        The name of the column containing the lower percentile.
    upper_percentile : str
        The name of the column containing the upper percentile.

    Returns
    -------
    pandas.DataFrame
        returns the original dataframe and 3 added columns: "under", "neutral", "out"
    """

    # Create new columns and initialize with default value
    dataframe['under'] = 0
    dataframe['neutral'] = 0
    dataframe['out'] = 0

    # Compare return with the percentiles and assign labels
    dataframe.loc[dataframe['return'] < dataframe[lower_percentile], 'under'] = 1
    dataframe.loc[(dataframe[lower_percentile] < dataframe['return']) & (
            dataframe['return'] < dataframe[upper_percentile]), 'neutral'] = 1
    dataframe.loc[dataframe['return'] > dataframe[upper_percentile], 'out'] = 1

    return dataframe


def add_relative_return_ordinal(dataframe, lower_percentile, upper_percentile):
    """
    Adds one column to a pandas dataframe. The added column indicates whether the return is underperforming,
    neutral, or outperforming compared to all stocks in the S&P500. Requires three columns: "return" and the
    lower_percentile and upper_percentile column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A pandas dataframe with two columns. One column containing the lower percentile. The other containing the upper
        percentile.
    lower_percentile : str
        The name of the column containing the lower percentile.
    upper_percentile : str
        The name of the column containing the upper percentile.

    Returns
    -------
    pandas.DataFrame
        returns the original dataframe and one added columns: "relative_return"
    """

    # Create a new column and initialize with default value
    dataframe['relative_return'] = 2

    # Compare 'return' with the percentiles and assign labels
    dataframe.loc[dataframe['return'] < dataframe[lower_percentile], 'relative_return'] = 1
    dataframe.loc[dataframe['return'] > dataframe[upper_percentile], 'relative_return'] = 3

    return dataframe
