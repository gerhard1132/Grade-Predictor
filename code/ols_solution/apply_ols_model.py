import os
import sys
import pickle
import pandas as pd
import statsmodels.api as sm


def main(output_folder, model_path, activation_data_path):
    with open(model_path, "rb") as file:
        ols_model = pickle.load(file)
    activation_data = pd.read_csv(activation_data_path)
    prediction = ols_model.predict(activation_data)
    
    with open(os.path.join(output_folder, "ols_result.txt"),"w") as file:
        file.write(str(prediction))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        script_name = sys.argv[0]
        volume_path = sys.argv[1]
        model_path = sys.argv[2]
        activation_data_path = sys.argv[3]
        main(volume_path, model_path, activation_data_path)
    else:
        print("No arguments given!")
    