import pathlib
from tqdm import tqdm
import pandas as pd

path = pathlib.Path(r"")


df = pd.read_csv(path / "metadata.csv")

df_nan = df[df['transcription'].isna()]

df2 = df.dropna(subset=['transcription'])
delete_file = df_nan['file_name'].tolist()

input("Delete " + str(len(delete_file)) + " files!, press to continue")

for i in tqdm(delete_file):
  print(f"Delete {i}")
  (path / i).unlink()

df2.to_csv(path / "metadata.csv")