import os
import json


model_type = "Claude"               # OpenAI  ||  Claude
model_var = "claude-3-5-sonnet-20240620"            # gpt-4o  ||  o1-preview  ||  claude-3-5-sonnet-20240620

org_data_path = f"logs/{model_type}/{model_var}_0_shot_predictions.json"
org_outputs = json.load(open(org_data_path, "r"))

pho_data_path = f"logs/{model_type}/{model_var}_0_shot_predictions_phonology.json"
pho_outputs = json.load(open(pho_data_path, "r"))

org_acc = {'total': [0, 0]}
for data in org_outputs:
    if data["level_1"] != "Phonology":
        continue

    if data["level_2"] not in org_acc:
        org_acc[data["level_2"]] = [0, 0]

    org_acc[data["level_2"]][0] += 1
    org_acc["total"][0] += 1
    if int(data["label"]) == int(data["prediction"]):
        org_acc[data["level_2"]][1] += 1
        org_acc["total"][1] += 1

pho_acc = {'total': [0, 0]}
for data in pho_outputs:
    if data["level_2"] not in pho_acc:
        pho_acc[data["level_2"]] = [0, 0]

    pho_acc[data["level_2"]][0] += 1
    pho_acc["total"][0] += 1
    if int(data["label"]) == int(data["prediction"]):
        pho_acc[data["level_2"]][1] += 1
        pho_acc["total"][1] += 1

print("[ Original Acc ]")
for key in org_acc:
    print(f"{key}: {org_acc[key][1]/org_acc[key][0] * 100:.2f} [%]")

print("\n[ Pho Acc ]")
for key in pho_acc:
    print(f"{key}: {pho_acc[key][1]/pho_acc[key][0] * 100:.2f} [%]")
