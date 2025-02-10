import os
import sys
import json
import random
import anthropic
from tqdm import tqdm
from kiwipiepy import Kiwi
sys.path.append(os.getcwd())
from srcs.functions import init_random

init_random(42)

kiwi = Kiwi()


################################
#      Call Claude Client      #
################################
"""
                          Fastest                       Fast                   Moderately fast                      Fastest                            Fast
model_variants: 'claude-3-haiku-20240307' || 'claude-3-sonnet-20240229' || 'claude-3-opus-20240229'   ||  'claude-3-5-haiku-20241022'  ||  'claude-3-5-sonnet-20240620'
"""
access_token = open("api_tokens/claude_token.txt", "r").read().strip()

client = anthropic.Anthropic(api_key=access_token)

model_var = "claude-3-5-haiku-20241022"


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

output_dir = f"logs/Claude/"
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
        if data["level_1"] != "Morphology":
            continue

        if data['data_src'] in ['NUAT(HS1)', 'NUAT(HS2)', 'NUAT(HS3)', 'CSAT']:
            cand_num = 5
        elif data['data_src'] in ['HSQE', 'LCSE(G9)', 'LCSE(G7)', 'NCSE(G9)', 'NCSE(G7)']:
            cand_num = 4
        else:
            raise NotImplementedError

        question = data['question'].strip()
        context = data['context'].strip()
        paragraph = data['paragraph'].strip()
        candidates = [cand.strip() for cand in data['candidates']]
        candidates = [cand + f" (형태소: {[tok_set.form + '(' + tok_set.tag + ')'  for tok_set in kiwi.tokenize(cand)]})" for cand in candidates]            # "안녕하세요, 반갑습니다. (형태소: ['안녕하세요(NNP)', ',(SP)', '반갑(VA-I)', '습니다(EF)', '.(SF)'])"
        candidates = "\n ".join(candidates)

        # prompt
        if context == '':
            if paragraph == '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            if paragraph == '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"지문: {context} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문과 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
                prompt = f"지문: {context} 설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "


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
                    "role": "user",
                    "content": prompt,
                }
            ]

            response = client.messages.create(
                system=pre_prompt,
                messages=messages,
                model=model_var,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = [response.content[j].text for j in range(len(response.content))][0]
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
              open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_morphology.json"), "w", encoding="utf-8"),
              ensure_ascii=False,
              indent=2
              )


# Aggregate all saved files
all_dataset = []
for i in range(batch_num):
    dataset = json.load(open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_morphology.json"), "r"))
    all_dataset.extend(dataset)

morphology_dataset = []
for data in all_dataset:
    if data["level_1"] == "Morphology":
        morphology_dataset.append(data)


print(f"Zero-shot Accuracy: {acc / len(morphology_dataset) * 100:.2f} [%]")
json.dump(morphology_dataset,
            open(os.path.join(output_dir, f"{model_var}_0_shot_predictions_morphology.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2
            )

# remove previous files
for i in range(batch_num):
    os.remove(os.path.join(output_dir, f"{model_var}_0_shot_predictions_{i}th_morphology.json"))

print("All predictions are saved.")
print("Done.")
