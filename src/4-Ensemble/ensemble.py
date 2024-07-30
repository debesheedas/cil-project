import pandas as pd
import numpy as np
import os
import json
# Takes two or more prediction files (probability of positive class) and produces an ensembled final prediction

# Load configuration from config.json
config_path = '../config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

output_path = config['output_dir']
prediction_paths = config['prediction_paths']

data = []

# Weights are by default initialized to give equal weightage to each prediction set. 
weights = [(1/len(prediction_paths))]*len(prediction_paths) 
#Can be modified as below
#weights = [0.2, 0.8]

print("Weights", weights)

#Make sure weights sum up to 1
assert np.sum(weights) == 1

for pred_file in prediction_paths:
    df = pd.read_csv(pred_file)
    data.append(df['Prediction']-0.5)

final_probs = np.average(data, weights=weights, axis=0)

final_preds = [-1 if val <0.0 else 1 for val in final_probs]

final_df = pd.DataFrame(final_preds, columns=["Prediction"])
final_df.index.name = "Id"
final_df.index += 1
final_df.to_csv(os.path.join(output_path, "test_ensemble.csv"))
print("Final output predictions written to: ", os.path.join(output_path, "test_ensemble.csv"))



