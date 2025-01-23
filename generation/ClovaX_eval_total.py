import os
import sys
import json
import random
import requests
from tqdm import tqdm
sys.path.append(os.getcwd())
from srcs.functions import init_random

init_random(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


################################
#     Call Clova X Client      #
################################
# https://clovastudio.apigw.ntruss.com/testapp/v1/completions/LK-B
# https://clovastudio.apigw.ntruss.com/testapp/v1/completions/LK-D2
# -*- coding: utf-8 -*-

class ChatCompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id, model_var="HCX-DASH-001"):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
        self._model_var = model_var

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + f'/testapp/v1/chat-completions/{self._model_var}',
                           headers=headers, json=completion_request, stream=True) as r:
            res = ''
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line[:4] == 'data':
                        line_dict = json.loads(line[5:])
                        if 'outputLength' in line_dict:
                            res = line_dict['message']['content']
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            return res


model_var = "HCX-DASH-001"          # 'HCX-DASH-001' || 'HCX-003'
chat_completion_executor = ChatCompletionExecutor(
            host='https://clovastudio.stream.ntruss.com',
            api_key='NTA0MjU2MWZlZTcxNDJiY6RwvAzX+D21GTpaNG276PGB64aDXtyA0k35Ezr25Dpj',
            api_key_primary_val='JmH9YTPiNjnz9aQxyLTMPwPeIeQ46tWimn1cFssk',
            request_id='7ec0cf16-dba2-4da4-acb5-121544436a46',
            model_var=model_var
        )


################################
#       Load the Dataset       #
################################
"""
( #class = 5 )
" mock_high1  ||  mock_high2  ||  mock_high3  ||  suneung "

( #class = 4 )
" school_qual_exam  ||  local_office7  ||  local_office9  ||  national_office7  ||  national_office9 "  
"""
total_dataset = json.load(open(f"datasets/KoGram_benchmark_list.json", "r"))

batch_size = 100
batch_num = len(total_dataset) // batch_size + 1


shot_num = '0'        # 0, 1, 5, 'each_major_one', 'each_sub_one'
cot = True
cot_prompt = "단계별로 생각해 보자."
cot_file_index = "_cot" if cot else ""


max_tokens = 200 if cot else 50


output_dir = f"logs/ClovaX/"
os.makedirs(output_dir, exist_ok=True)


acc = 0
cannot_generate = 0
for i in range(batch_num):
    dataset = total_dataset[i*batch_size: (i+1)*batch_size]

    if shot_num != str(0):
        if shot_num == 'each_major_one':
            major = ["통사론", "형태론", "의미론", "음운론", "규범", "복합"]
            major_data_ids = {cat: [] for cat in major}
            for d_id, data in enumerate(dataset):
                major_data_ids[data['level_1']].append(d_id)
            random_ids = []
            for cat in major:
                random_ids.append(random.choice(major_data_ids[cat]))
        elif shot_num == 'each_sub_one':
            sub = ["문장의 짜임", "문법요소", "형태소", "품사", "단어의 짜임", "단순어휘", "어휘의미론", "화용론",
                   "음운체계", "음운변동", "표준어", "맞춤법", "로마자표기법", "표준발음법", "외래어표기법", "규범 복합", "복합"]
            sub_data_ids = {cat: [] for cat in sub}
            for d_id, data in enumerate(dataset):
                sub_data_ids[data['level_2']].append(d_id)
            random_ids = []
            for cat in sub:
                random_ids.append(random.choice(sub_data_ids[cat]))
        elif type(shot_num) == int and shot_num != str(0):
            random_ids = [random.randint(0, len(dataset)) for _ in range(shot_num)]
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
                        desc=f"({i+1}/{batch_num}) th Generating answers using '{model_var}' model with {shot_num}-shot{cot_file_index} eval...",
                        bar_format="{l_bar}{bar:15}{r_bar}"):
        if data['data_src'] in ['mock_high1', 'mock_high2', 'mock_high3', 'suneung']:
            cand_num = 5
        elif data['data_src'] in ['school_qual_exam', 'local_office7', 'local_office9', 'national_office7',
                                  'national_office9']:
            cand_num = 4
        else:
            raise NotImplementedError

        question = data['question'].strip()
        context = data['context'].strip()
        paragraph = data['paragraph'].strip()
        candidates = [cand.strip() for cand in data['candidates']]
        candidates = "\n ".join(candidates)

        # pre_prompt
        if context.strip != '':
            if paragraph.strip != '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문과 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 지문을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
        else:
            if paragraph.strip != '':
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 설명을 보고, 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."
            else:
                pre_prompt = "다음은 한국어 언어 이해에 대한 객관식 문제입니다. 주어진 질문에 대한 정답으로 올바른 번호를 선택지에서 고르고, 그에 맞는 해설을 100자 내로 설명하시오."

        if shot_num != str(0):
            # pre_prompt
            if context.strip != '':
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
                c = ex['context'].strip()
                p = ex['paragraph'].strip()
                cands = [cand.strip() for cand in ex['candidates']]
                cands = "\n ".join(cands)
                ans = ex['label']

                if ex['data_src'] in ['mock_high1', 'mock_high2', 'mock_high3', 'suneung']:
                    ex_cand_num = 5
                elif ex['data_src'] in ['school_qual_exam', 'local_office7', 'local_office9', 'national_office7',
                                        'national_office9']:
                    ex_cand_num = 4
                else:
                    raise NotImplementedError

                if context == '':
                    if paragraph == '':
                        ex_prompt = f"질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                    else:
                        ex_prompt = f"설명: {p} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                else:
                    if paragraph == '':
                        ex_prompt = f"지문: {context} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                    else:
                        ex_prompt = f"지문: {context} 설명: {p} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "

                example_prompts.append(ex_prompt)
                example_answers.append(str(ans))

        # prompt
        if context == '':
            if paragraph == '':
                prompt = f"질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                prompt = f"설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            if paragraph == '':
                prompt = f"지문: {context} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
            else:
                prompt = f"지문: {context} 설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "

        if cot:
            prompt_extended = prompt[: -len(" 정답: ")] + cot_prompt + "\n"


        label = str(data['label'])

        ##################################
        #       Answer Generation        #
        ##################################
        prediction = ''
        num_repeat = 0
        temperature = 0.
        generated_text = ''
        random_select = False
        cot_answer = ''
        while prediction == '' and num_repeat < 5:
            if shot_num != str(0):
                messages = [
                    {
                        "role": "system",
                        "content": pre_prompt,
                    }
                ]
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
                if cot:
                    messages = [
                        {
                            "role": "system",
                            "content": pre_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt_extended,
                        }
                    ]
                else:
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

            request_data = {
                'messages': messages,
                'topP': 1e-08,
                'topK': 0,
                'maxTokens': max_tokens,
                'temperature': 1e-08,
                'repeatPenalty': 5.0,
                'stopBefore': [],
                'includeAiFilters': True,
                'seed': 123
            }

            if cot:
                cot_answer = chat_completion_executor.execute(request_data)
                prompt_extended += cot_answer + "\n" + " 정답: "
                messages = [
                    {
                        "role": "system",
                        "content": pre_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt_extended,
                    }
                ]
                request_data = {
                    'messages': messages,
                    'topP': 1e-08,
                    'topK': 0,
                    'maxTokens': 50,
                    'temperature': 1e-08,
                    'repeatPenalty': 5.0,
                    'stopBefore': [],
                    'includeAiFilters': True,
                    'seed': 123
                }

            generated_text = chat_completion_executor.execute(request_data)
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
        if cot:
            data['cot_answer'] = cot_answer

        if prediction == label:
            acc += 1


    ###################################
    #       Save the Predictions      #
    ###################################
    json.dump(dataset,
              open(os.path.join(output_dir, f"{model_var}_{shot_num}_shot{cot_file_index}_predictions_{i}th.json"), "w", encoding="utf-8"),
              ensure_ascii=False,
              indent=2
              )

    # print("Predictions are saved.")
    # print("Done.")


# Aggregate all saved files
all_dataset = []
for i in range(batch_num):
    dataset = json.load(open(os.path.join(output_dir, f"{model_var}_{shot_num}_shot{cot_file_index}_predictions_{i}th.json"), "r"))
    all_dataset.extend(dataset)


print(f"Zero-shot Accuracy: {acc / len(all_dataset) * 100:.2f} [%]")
print(f"Cannot generate: {cannot_generate}/ {len(all_dataset)} ({cannot_generate / len(all_dataset) * 100:.2f} [%])")
json.dump(all_dataset,
            open(os.path.join(output_dir, f"{model_var}_{shot_num}_shot{cot_file_index}_predictions.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2
            )


# remove previous files
for i in range(batch_num):
    os.remove(os.path.join(output_dir, f"{model_var}_{shot_num}_shot{cot_file_index}_predictions_{i}th.json"))

print("All predictions are saved.")
print("Done.")
