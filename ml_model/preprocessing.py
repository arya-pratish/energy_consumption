import pandas as pd




def con_str_num(a):
    if a.isdigit():
        return int(a)
    elif len(a)==0:
        return pd.NA
    return a

def pre_process(df):
# Strip whitespace from column names
    df.columns = df.columns.str.strip()

# Remove leading/trailing whitespace from string values
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)



# Convert numerical string into integer
    df=df.map(lambda x: con_str_num(x) if isinstance(x,str) else x)

# Drop rows with any missing values
    df.dropna(inplace=True)

# Drop duplicate rows
    df.drop_duplicates(inplace=True)

# Map string variable to integer
    df['Building Type'] = df['Building Type'].map({'Residential':0,'Commercial':1,'Industrial':2})
    df['Day of Week'] = df['Day of Week'].map({'Weekday':0,'Weekend':1})

    return df


