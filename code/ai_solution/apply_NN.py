
import pandas as pd
import sys
import os
from tensorflow.keras.models import load_model # type: ignore

def main(volume_path, model_path, activation_data_path):
    model = load_model(model_path)
    
    test_data = pd.read_csv(activation_data_path)
    input = test_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]] 

    # Predict using the trained model
    output_pred = model.predict(input)
    
    with open(os.path.join(volume_path, 'prediction.txt'), "w") as file:
        file.write(str(output_pred))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        script_name = sys.argv[0]
        volume_path = sys.argv[1]
        model_path = sys.argv[2]
        activation_data_path = sys.argv[3]
        main(volume_path, model_path, activation_data_path)
    else:
        print("No arguments given!")
    