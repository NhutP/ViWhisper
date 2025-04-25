import soundata
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('dana', help = 'dataset name')
  parser.add_argument('stro', help='path to processed data strogage folder', type=str)

  args = parser.parse_args()

  dataset_name = str(args.dana)
  store = str(args.stro)

  dataset = soundata.initialize(dataset_name, data_home=store)
  dataset.download()  # download the dataset
  dataset.validate()  # validate that all the expected files are there

  example_clip = dataset.choice_clip()  # choose a random example clip
  print(example_clip)  # see the available data