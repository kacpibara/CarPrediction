import pandas as pd

init_data = pd.read_csv("./data/car_price_dataset.csv")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['Brand', 'Model', 'Year', 'Engine_Size', 'Fuel_Type',
                             'Transmission', 'Mileage', 'Doors', 'Owner_Count', 'Price'])


def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df['Year'] = df['Year'].astype(int)
    df['Engine_Size'] = df['Engine_Size'].astype(float)
    df['Mileage'] = df['Mileage'].astype(int)
    df['Doors'] = df['Doors'].astype(int)
    df['Owner_Count'] = df['Owner_Count'].astype(int)
    df['Price'] = df['Price'].astype(int)
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['Brand'] = df['Brand'].str.strip().str.title()
    df['Model'] = df['Model'].str.strip().str.upper()
    df['Fuel_Type'] = df['Fuel_Type'].str.strip().str.capitalize()
    df['Transmission'] = df['Transmission'].str.strip().str.capitalize()
    return df


def remove_invalid_entries(df: pd.DataFrame) -> pd.DataFrame:
    current_year = pd.Timestamp.now().year
    df = df[(df['Year'] >= 1980) & (df['Year'] <= current_year)]
    df = df[(df['Mileage'] >= 0) & (df['Price'] > 0) & (df['Doors'] > 0)]
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def pipeline(data, *funcs) -> pd.DataFrame:
    for f in funcs:
        data = f(data)
    return data


cleaned = pipeline(init_data, remove_duplicates, drop_missing_values,
                   correct_data_types, clean_text_columns, remove_invalid_entries, reset_index)
cleaned.to_csv("./data/cleaned.csv", index=False)
