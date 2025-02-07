import os
import json


model_type = "OpenAI"
model_var = "o1-preview"

org_data_path = f"logs/{model_type}/{model_var}_0_shot_predictions.json"
org_outputs = json.load(open(org_data_path, "r"))

mor_data_path = f"logs/{model_type}/{model_var}_0_shot_predictions_morphology_t1.json"
mor_outputs = json.load(open(mor_data_path, "r"))

org_acc = {'total': [0, 0]}
for data in org_outputs:
    if data["level_1"] != "Morphology":
    # if data["level_2"] != "Morpheme":
        continue

    if data["level_2"] not in org_acc:
        org_acc[data["level_2"]] = [0, 0]

    org_acc[data["level_2"]][0] += 1
    org_acc["total"][0] += 1
    if int(data["label"]) == int(data["prediction"]):
        org_acc[data["level_2"]][1] += 1
        org_acc["total"][1] += 1

mor_acc = {'total': [0, 0]}
for data in mor_outputs:
    if data["level_2"] not in mor_acc:
        mor_acc[data["level_2"]] = [0, 0]

    mor_acc[data["level_2"]][0] += 1
    mor_acc["total"][0] += 1
    if int(data["label"]) == int(data["prediction"]):
        mor_acc[data["level_2"]][1] += 1
        mor_acc["total"][1] += 1

print("[ Original Acc ]")
for key in ["Part-of-Speech", "Morpheme", "Word Formation", "total"]:
    print(f"{key}: {org_acc[key][1]/org_acc[key][0] * 100:.2f} [%]")

print("\n[ mor Acc ]")
for key in ["Part-of-Speech", "Morpheme", "Word Formation", "total"]:
    print(f"{key}: {mor_acc[key][1]/mor_acc[key][0] * 100:.2f} [%]")
