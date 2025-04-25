import pandas as pd
import evaluate
import argparse

def compute(csv_path, pred_col, label_col):

  df = pd.read_csv(csv_path)

  pred = df[pred_col].tolist()
  label = df[label_col].tolist()
  metric = evaluate.load("wer")
  

  return 100 * metric.compute(predictions=pred, references=label)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Process a string input.")

  # Add the arguments
  parser.add_argument('csv_file', type=str, help="The csv file")
  parser.add_argument('pred_col', type=str, help="The pred column name")
  parser.add_argument('label_col', type=str, help="The label column name")
  # Parse the arguments
  args = parser.parse_args()

 
  file = args.csv_file
  pred_col = args.pred_col
  label_col = args.label_col

  print(compute(file, pred_col, label_col))