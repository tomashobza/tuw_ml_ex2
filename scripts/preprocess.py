# %%
import pandas as pd
import os

hp_path = os.path.join("..", "data", "Housing.csv")
df_hp = pd.read_csv(hp_path)

# %%
# one hot encoding for categorical features
# furnishingstatus    object

df_hp = pd.get_dummies(df_hp, columns=["furnishingstatus"], drop_first=True)

# convert yes/no to 1/0
yes_no_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
for col in yes_no_columns:
    df_hp[col] = df_hp[col].map({"yes": 1, "no": 0})

# rename price to y
df_hp = df_hp.rename(columns={"price": "y"})

# save processed data with postfix '_processed'
processed_hp_path = os.path.join("..", "data", "Housing_processed.csv")
df_hp.to_csv(processed_hp_path, index=False)

# %%
spotify_path = os.path.join("..", "data", "dataset.csv")
df_spotify = pd.read_csv(spotify_path)

# %%
df_spotify.dtypes

# %%
# drop
# track_id             object
# artists              object
# album_name           object
# track_name           object
df_spotify = df_spotify.drop(
    columns=["track_id", "artists", "album_name", "track_name"]
)

# convert explicit to 1/0
df_spotify["explicit"] = df_spotify["explicit"].map({True: 1, False: 0})

# encode track_genre as integer
df_spotify["track_genre"] = df_spotify["track_genre"].astype("category").cat.codes

# rename popularity to y
df_spotify = df_spotify.rename(columns={"popularity": "y"})

# downsample to 100000 rows for faster processing
df_spotify = df_spotify.sample(n=10000, random_state=42)

# save processed data with postfix '_processed'
processed_spotify_path = os.path.join("..", "data", "dataset_processed.csv")
df_spotify.to_csv(processed_spotify_path, index=False)

# %%
sp_path = os.path.join("..", "data", "Student_Performance.csv")
df_sp = pd.read_csv(sp_path)

# %%
# convert Extracurricular Activities to 1/0 from Yes/No
df_sp = df_sp.assign(
    **{
        "Extracurricular Activities": df_sp["Extracurricular Activities"].map(
            {"Yes": 1, "No": 0}
        )
    }
)

# rename Performance Index to y
df_sp = df_sp.rename(columns={"Performance Index": "y"})

# save processed data with postfix '_processed'
processed_sp_path = os.path.join("..", "data", "Student_Performance_processed.csv")
df_sp.to_csv(processed_sp_path, index=False)

# %%
