./utils/preprocess.py
- Drop Columns
- Feature Scaling
- Categorical Feature to One Hot Encoding

combine_and_split.py
- combine all dataset in ./ALL and split into ./Eval ./Train with preprocessing

get_dataframe.py
Usage: python get_dataframe.py <input_csv_path>
- get pandas dataframe from csv file
- load from pickle file if exist