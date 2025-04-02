import os
import sys
import json
import random
import argparse
import anthropic
from tqdm import tqdm
sys.path.append(os.getcwd())
from srcs.functions import init_random

init_random(42)


#####################################
#           Default Settings         #
######################################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="debug")
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--model_var', type=str, required=True,
                    choices=["claude-3-haiku-20240307",
                             "claude-3-sonnet-20240229",
                             "claude-3-opus-20240229",
                             "claude-3-5-sonnet-20240620",
                             ])
parser.add_argument('--access_token_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_tokens', type=int, default=200)
parser.add_argument('--shot_num', type=str, default=0, help="0  ||  1  ||  5  ||  each_major_one  ||  each_sub_one")
args = parser.parse_args()


################################
#     Load the Basic Info      #
################################
kogem_info = json.load(open("utils/KoGEM_info.json", "r"))


################################
#      Call Claude Client      #
################################
access_token = open(args.access_token_path, "r").read().strip()

client = anthropic.Anthropic(api_key=access_token)

model_var = args.model_var


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

batch_size = args.batch_size
batch_num = len(total_dataset) // batch_size + 1


################################
#     Experimental settings    #
################################
max_tokens = args.max_tokens

output_dir = f"logs/Claude/"
os.makedirs(output_dir, exist_ok=True)


################################
#         Do Evaluation        #
################################
acc = 0
cannot_generate = 0
for i in range(batch_num):
    dataset = total_dataset[i*batch_size: (i+1)*batch_size]

    if args.shot_num != str(0):
        if args.shot_num == 'each_major_one':
            major = kogem_info["major_categories"]
            major_data_ids = {cat: [] for cat in major}
            for d_id, data in enumerate(dataset):
                major_data_ids[data['level_1']].append(d_id)
            random_ids = []
            for cat in major:
                random_ids.append(random.choice(major_data_ids[cat]))
        elif args.shot_num == 'each_sub_one':
            sub = kogem_info["sub_categories"]
            sub_data_ids = {cat: [] for cat in sub}
            for d_id, data in enumerate(dataset):
                sub_data_ids[data['level_2']].append(d_id)
            random_ids = []
            for cat in sub:
                random_ids.append(random.choice(sub_data_ids[cat]))
        elif args.shot_num in [str(sn) for sn in range(10)]:
            random_ids = [random.randint(0, len(dataset) - 1) for _ in range(int(args.shot_num))]
        else:
            raise NotImplementedError

        examples = [dataset[i] for i in random_ids]
        new_dataset = []
        for idx, data in enumerate(dataset):
            if idx not in random_ids:
                new_dataset.append(data)

        dataset = new_dataset


    ##################################
    #        Prompt Evaluation       #
    ##################################
    for n, data in tqdm(enumerate(dataset), total=len(dataset),
                        desc=f"({i+1}/{batch_num}) th Generating answers using '{model_var}' model with {args.shot_num}-shot eval...",
                        bar_format="{l_bar}{bar:15}{r_bar}"):
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
        candidates = "\n ".join(candidates)

        # pre_prompt
        if passage.strip != '':
            if paragraph.strip != '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문과 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
        else:
            if paragraph.strip != '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."

        if args.shot_num != str(0):
            # pre_prompt
            if passage.strip != '':
                if paragraph.strip != '':
                    pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문과 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르시오."
                else:
                    pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르시오."
            else:
                if paragraph.strip != '':
                    pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르시오."
                else:
                    pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 질문에 대한 정답으로 올바른 번호를 선택지에서 고르시오."

            example_prompts = []
            example_answers = []
            for ex in examples:
                q = ex['question'].strip()
                c = ex['passage'].strip()
                p = ex['paragraph'].strip()
                cands = [cand.strip() for cand in ex['candidates']]
                cands = "\n ".join(cands)
                ans = ex['label']

                if ex['data_src'] in ['NUAT(HS1)', 'NUAT(HS2)', 'NUAT(HS3)', 'CSAT']:
                    ex_cand_num = 5
                elif ex['data_src'] in ['HSQE', 'LCSE(G9)', 'LCSE(G7)', 'NCSE(G9)', 'NCSE(G7)']:
                    ex_cand_num = 4
                else:
                    raise NotImplementedError

                if passage == '':
                    if paragraph == '':
                        ex_prompt = f"질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                    else:
                        ex_prompt = f"설명: {p} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                else:
                    if paragraph == '':
                        ex_prompt = f"지문: {c} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                    else:
                        ex_prompt = f"지문: {c} 설명: {p} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "

                example_prompts.append(ex_prompt)
                example_answers.append(str(ans))

        # prompt
        if passage == '':
            if paragraph == '':
                prompt = f"질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                prompt = f"설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            if paragraph == '':
                prompt = f"지문: {passage} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
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
            if args.shot_num != str(0):
                messages = []
                for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                    messages.extend([
                        {
                            "role": "user",
                            "content": ex_prompt,
                        },
                        {
                            "role": "assistant",
                            "content": ex_ans,
                        }
                    ])
                messages.extend([
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ])
            else:       # zero-shot
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

            try:
                response = client.messages.create(
                    system=pre_prompt,
                    messages=messages,
                    model=model_var,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except anthropic.InternalServerError:
                print("\nInternal Server Error occurred.\n")
                prediction = ''
                break


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
              open(os.path.join(output_dir, f"{model_var}_{args.shot_num}_shot_predictions_{i}th.json"), "w", encoding="utf-8"),
              ensure_ascii=False,
              indent=2
              )

    # print("Predictions are saved.")
    # print("Done.")


# Aggregate all saved files
all_dataset = []
for i in range(batch_num):
    dataset = json.load(open(os.path.join(output_dir, f"{model_var}_{args.shot_num}_shot_predictions_{i}th.json"), "r"))
    all_dataset.extend(dataset)


print(f"Zero-shot Accuracy: {acc / len(all_dataset) * 100:.2f} [%]")
print(f"Cannot generate: {cannot_generate}/ {len(all_dataset)} ({cannot_generate / len(all_dataset) * 100:.2f} [%])")
json.dump(all_dataset,
            open(os.path.join(output_dir, f"{model_var}_{args.shot_num}_shot_predictions.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2
            )


# remove previous files
for i in range(batch_num):
    os.remove(os.path.join(output_dir, f"{model_var}_{args.shot_num}_shot_predictions_{i}th.json"))

print("All predictions are saved.")
print("Done.")
