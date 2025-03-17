import os

import pandas as pd


def add_constants_to_columns(
    csv_path, column_names: list[str], constant_values: list[int], output_path=None
):
    """
    Load a CSV file, add a constant value to all rows of a specified column, and save the result.

    Args:
        csv_path (str): Path to the input CSV file
        column_names (list[str]): Names of the columns to modify
        constant_values (list[float]): Values to add to each row in the specified columns
        output_path (str, optional): Path to save the modified CSV. If None, overwrites the input.

    Returns:
        pd.DataFrame: The modified DataFrame
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Check if column exists
    for column_name in column_names:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file")

    # Add constant to the specified column
    for column_name, constant_value in zip(column_names, constant_values):
        df[column_name] = df[column_name].apply(
            lambda x: x + constant_value if pd.notnull(x) else x
        )

    # Save the modified DataFrame
    if output_path is None:
        output_path = csv_path
    df.to_csv(output_path, index=False)

    print(f"Added {constant_values} to columns {column_names} in {csv_path}")
    print(f"Saved result to {output_path}")

    return df


if __name__ == "__main__":
    path = "logs/model-tiny-new/metrics/metrics copy 2.csv"
    add_constants_to_columns(path, ["step", "epoch"], [29798, 40])
