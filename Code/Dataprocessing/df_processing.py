import pandas


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
