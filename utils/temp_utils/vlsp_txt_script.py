import pathlib
import pandas as pd


def format(path, des_path):
  path = pathlib.Path(path)
  des_path = pathlib.Path(des_path)

  files = []
  trans = []

  with open(path, 'r', encoding='utf8') as r:
    text = r.read().split('\n')
    
    for i in text:
      i = i.split(' ')
      files.append(i[0].split('_')[0] + '/' + i[0] + '.wav')
      print(files)
      scr = ' '.join(i[1:len(i)]).lower()
      print(scr)
      
      trans.append(scr)

  df = pd.DataFrame.from_dict({'file_name' : files, 'transcription' : trans})
  df.to_csv(des_path, index=False)


if __name__ == '__main__':
  txt = r"C:\Users\quang\Desktop\vivos\vivos\test\prompts.txt"
  des = r'C:\Users\quang\Desktop\metadata_vivos.csv'
  format(txt, des)