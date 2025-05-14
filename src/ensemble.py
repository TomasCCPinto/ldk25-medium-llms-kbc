import argparse
import string
from typing import List, Dict, Union

import pandas as pd
import numpy as np

from file_io import *
from evaluate import *
import random
from collections import Counter



# Relation-Based Model Selection
def relation_based(model1, model2, model3, eval1, eval2, eval3):
    results=[]

    for row in range(len(model1)):
        rel = model1[row]["Relation"]
        
        # Retrieve F1 scores for the current relation from each evaluation DataFrame
        f1_1 = eval1.loc[rel, "f1"]
        f1_2 = eval2.loc[rel, "f1"]
        f1_3 = eval3.loc[rel, "f1"]
        
        # Find the maximum F1 score model
        max_f1 = max(f1_1, f1_2, f1_3)
        
        # Determine which models have the max F1 score
        best_models = []
        if f1_1 == max_f1:
            best_models.append(model1[row])
        if f1_2 == max_f1:
            best_models.append(model2[row])
        if f1_3 == max_f1:
            best_models.append(model3[row])
        
        # Choose one of the best models at random if there's a tie
        selected_row = random.choice(best_models)
        results.append(selected_row)
    return results


def majority_voting(model1, model2, model3, eval1, eval2, eval3):
    results = []
    
    for row in range(len(model1)):
        rel = model1[row]["Relation"]
        
        # Collect responses for each model, converting lists to tuples if necessary
        responses = [
            tuple(model1[row]["ObjectEntitiesID"]) if isinstance(model1[row]["ObjectEntitiesID"], list) else model1[row]["ObjectEntitiesID"],
            tuple(model2[row]["ObjectEntitiesID"]) if isinstance(model2[row]["ObjectEntitiesID"], list) else model2[row]["ObjectEntitiesID"],
            tuple(model3[row]["ObjectEntitiesID"]) if isinstance(model3[row]["ObjectEntitiesID"], list) else model3[row]["ObjectEntitiesID"]
        ]
        
        # Count occurrences of each unique response
        response_counts = Counter(responses)
        most_common_response, count = response_counts.most_common(1)[0]
        
        # Determine selected response
        if count > 1:  # If there is a majority
            selected_response = most_common_response
        else:  # If there's no majority, choose based on F1 or randomly
            f1_scores = [eval1.loc[rel, "f1"], eval2.loc[rel, "f1"], eval3.loc[rel, "f1"]]
            max_f1_index = f1_scores.index(max(f1_scores))
            selected_response = responses[max_f1_index]
        
        # Identify the model that provided the selected response and append the full row
        if selected_response == (tuple(model1[row]["ObjectEntitiesID"]) if isinstance(model1[row]["ObjectEntitiesID"], list) else model1[row]["ObjectEntitiesID"]):
            selected_row = model1[row]
        elif selected_response == (tuple(model2[row]["ObjectEntitiesID"]) if isinstance(model2[row]["ObjectEntitiesID"], list) else model2[row]["ObjectEntitiesID"]):
            selected_row = model2[row]
        else:
            selected_row = model3[row]
        
        # Append the full row to results
        results.append(selected_row)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Model ensemble with 3 models")

    parser.add_argument(
        "-m1",
        "--model1",
        type=str,
        required=True,
        help="Path to the 1st model predictions file (required)"
    )
    parser.add_argument(
        "-m2",
        "--model2",
        type=str,
        required=True,
        help="Path to the 2nd model predictions file (required)"
    )

    parser.add_argument(
        "-m3",
        "--model3",
        type=str,
        required=True,
        help="Path to the 3rd model predictions file (required)"
    )

    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground_truth file (required)"
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        help="type of ensemble approach used (r - relation based / m - majority voting)"
    )

    args = parser.parse_args()

    #eval1 = eval(args.model1,args.ground_truth)
    #eval2 = eval(args.model2,args.ground_truth)
    #eval3 = eval(args.model3,args.ground_truth)
    eval1 = eval("","")
    eval2 = eval("","")
    eval3 = eval("","")
    model1 = read_lm_kbc_jsonl(args.model1)
    model2 = read_lm_kbc_jsonl(args.model2)
    model3 = read_lm_kbc_jsonl(args.model3)

    if args.type == "r":
        results = relation_based(model1,model2,model3,eval1,eval2,eval3)
        with open('new_ensemble_relation_based.jsonl', "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(eval('new_ensemble_relation_based.jsonl',args.ground_truth))

    elif args.type == "m":
        results = majority_voting(model1,model2,model3,eval1,eval2,eval3)
        with open('new_ensemble_majority_voting.jsonl', "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(eval('new_ensemble_majority_voting.jsonl',args.ground_truth))
    else:
        print("type error")

    
    

    #print(model1)
    



if __name__ == "__main__":
    main()

