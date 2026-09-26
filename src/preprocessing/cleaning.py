def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)