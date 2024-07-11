import pandas as pd
import numpy as np
import os
# Takes two or more prediction files (probability of positive class) and produces an ensembled final prediction

# Path to the directory where you want the final prediction file to be saved
output_path = '/home/debdas/cil/cil-project/output'

# List of paths to the files containing the prediction probabilities for the test set
prediction_paths = ['/home/debdas/cil/cil-project/output/probabilities_0.csv', '/home/debdas/cil/cil-project/output/probabilities_1.csv']


data = []
# Weights are by default initialized to give equal weightage to each prediction set. Can be modified below
weights = [(1/len(prediction_paths))]*len(prediction_paths)
# weights = [0.2,0.8]
print("Weights", weights)
#Make sure weights sum up to 1
assert np.sum(weights) == 1

for pred_file in prediction_paths:
    df = pd.read_csv(pred_file)
    # print(len(df['Prediction']))
    data.append(df['Prediction'])

# print(np.array(data).shape)

final_probs = np.average(data, weights=weights, axis=0)
# print(final_probs.shape)

# final_preds = final_probs
final_preds = [-1 if val <0.5 else 1 for val in final_probs]
# print(np.sum(final_preds))

final_df = pd.DataFrame(final_preds, columns=["Prediction"])
final_df.index.name = "Id"
final_df.index += 1
final_df.to_csv(os.path.join(output_path, "test_ensemble.csv"))
print("Final output predictions written to: ", os.path.join(output_path, "test_ensemble.csv"))



