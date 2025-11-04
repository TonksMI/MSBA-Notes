from pathlib import Path

import pandas as pd


def clean_coffee_data(
    input_path: Path,
    output_path: Path,
) -> None:
    coffee_df = pd.read_csv(input_path)

    # Normalize temporal columns for consistent downstream usage.
    coffee_df['date'] = pd.to_datetime(coffee_df['date'], errors='coerce')
    coffee_df['time'] = pd.to_datetime(
        coffee_df['time'],
        format='%H:%M:%S.%f',
        errors='coerce',
    ).dt.time

    outlier_threshold = coffee_df['money'].quantile(0.99)
    coffee_df['is_outlier'] = (coffee_df['money'] < 0) | (coffee_df['money'] > outlier_threshold)
    coffee_df['missing_hour'] = coffee_df['hour_of_day'].isna().astype(int)
    coffee_df['missing_type'] = coffee_df['cash_type'].isna().astype(int)

    coffee_df['skip_steps'] = coffee_df[['is_outlier', 'missing_hour', 'missing_type']].sum(axis=1)
    coffee_df['skip_step_score'] = 100 - (coffee_df['skip_steps'] / 3 * 100)

    coffee_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_csv = base_dir / "Coffee Sales.csv"
    output_csv = base_dir / "Coffee_Sales_Cleaned.csv"
    clean_coffee_data(input_csv, output_csv)
