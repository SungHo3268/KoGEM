import os
import sys
import json
import random
from g2pk import G2p
from tqdm import tqdm
from openai import OpenAI
sys.path.append(os.getcwd())
from srcs.functions import init_random

init_random(42)
g2p = G2p()


################################
#      Call OpenAI Client      #
################################
"""
model_variants: 'gpt-3.5-turbo-0125' || 'gpt-4o' || 'gpt-4-mini'
"""
access_token = open("api_tokens/openai_token.txt", "r").read().strip()
client = OpenAI(api_key=access_token)

# model_var = "gpt-3.5-turbo-0125"
# model_var = "gpt-4o"
# model_var = "gpt-4o-mini"
model_var = "o1-preview"
# model_var = "o1-mini"


################################
#       Load the Dataset       #
################################
"""
( #class = 5 )
" NUAT(HS1)  ||  NUAT(HS2)  ||  NUAT(HS3)  ||  CSAT "

( #class = 4 )
" HSQE  ||  LCSE(G9)  ||  LCSE(G7)  ||  NCSE(G9)  ||  NCSE(G7) "  
"""
total_dataset = json.load(open(f"datasets/KoGEM_benchmark.json", "r"))

batch_size = 100
batch_num = len(total_dataset) // batch_size + 1


################################
#     Experimental settings    #
################################
max_tokens = 110

output_dir = f"logs/OpenAI/"
os.makedirs(output_dir, exist_ok=True)


################################
#         Do Evaluation        #
################################
acc = 0
cannot_generate = 0
for i in range(batch_num):
    dataset = total_dataset[i*batch_size: (i+1)*batch_size]

    ##################################
    #        Prompt Evaluation       #
    ##################################
    for n, data in tqdm(enumerate(dataset), total=len(dataset),
                        desc=f"({i+1}/{batch_num}) th Generating answers using '{model_var}' model with 0-shot eval...",
                        bar_format="{l_bar}{bar:15}{r_bar}"):
        if data["level_1"] != "Phonology":
            continue

        if data['data_src'] in ['NUAT(HS1)', 'NUAT(HS2)', 'NUAT(HS3)', 'CSAT']:
            cand_num = 5
        elif data['data_src'] in ['HSQE', 'LCSE(G9)', 'LCSE(G7)', 'NCSE(G9)', 'NCSE(G7)']:
            cand_num = 4
        else:
            raise NotImplementedError

        question = data['question'].strip()
        passage = data['passage'].strip()
        paragraph = data['paragraph'].strip()
        candidates = [cand.strip() for cand in data['candidates']]
        candidates = [cand + f" (발음: {g2p(cand)})" for cand in candidates]
        candidates = "\n ".join(candidates)

        # prompt
        if passage == '':
            if paragraph == '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            if paragraph == '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"지문: {passage} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문과 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"지문: {passage} 설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "


        label = str(data['label'])

        ##################################
        #       Answer Generation        #
        ##################################
        prediction = ''
        num_repeat = 0
        temperature = 0.
        generated_text = ''
        random_select = False
        while prediction == '' and num_repeat < 5:
            messages = [
                {
                    "role": "system",
                    "content": pre_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            if "o1" in model_var:
                for m, message in enumerate(messages):
                    if message["role"] == "system":
                        messages[m]["role"] = "user"
                response = client.chat.completions.create(
                    messages=messages,
                    model=model_var,
                    seed=123,
                )
            else:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model_var,
                    seed=123,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            generated_text = response.choices[0].message.content
            for char in generated_text:
                if char.isnumeric() and char in [str(n) for n in range(cand_num, 0, -1)]:
                    prediction = char
                    break
                else:
                    prediction = ''

            if prediction == '':
                num_repeat += 1
                if num_repeat == 2:  # after 3rd trial, increase the temperature
                    temperature = 0.3

        if prediction == '':
            prediction = str(random.randint(1, 5))
            cannot_generate += 1
            random_select = True

        data['prediction'] = prediction
        data['generated_ans'] = generated_text
        data['random_sel'] = int(random_select)

        if prediction == label:
            acc += 1


    ###################################
    #       Save the Predictions      #
    ###################################
    json.dump(dataset,
              open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_phonology.json"), "w", encoding="utf-8"),
              ensure_ascii=False,
              indent=2
              )

    # print("Predictions are saved.")
    # print("Done.")


# Aggregate all saved files
all_dataset = []
for i in range(batch_num):
    dataset = json.load(open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_phonology.json"), "r"))
    all_dataset.extend(dataset)

phonology_dataset = []
for data in all_dataset:
    if data["level_1"] == "Phonology":
        phonology_dataset.append(data)


print(f"Zero-shot Accuracy: {acc / len(phonology_dataset) * 100:.2f} [%]")
json.dump(phonology_dataset,
            open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_phonology.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2
            )

# remove previous files
for i in range(batch_num):
    os.remove(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_phonology.json"))

print("All predictions are saved.")
print("Done.")
