import re
import pathlib
from tqdm import tqdm


format_transcript_text = lambda s : re.sub(r'\[[^\[]*\]', ' ', s)

def format_folder(path):
  path = pathlib.Path(path)
  input("This will rewrite the text, make sure you have backed up")

  for i in tqdm(list(path.rglob("*.txt"))):
    with open(i, 'r', encoding='utf8') as r:
      raw_text = r.read()
      formatted_text = format_transcript_text(raw_text)     
    
    with open(i, 'w', encoding='utf8') as w:
      w.write(formatted_text)


if __name__ =='__main__':
  path = r""
  format_folder(path)