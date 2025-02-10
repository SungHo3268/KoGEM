import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import torch
import random
import argparse
from distutils.util import strtobool as _bool
from time import time
from tqdm import tqdm
from huggingface_hub import login
from typing import Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

sys.path.append(os.getcwd())
from srcs.functions import init_random


################################
#     Load the Basic Info      #
################################
kogem_info = json.load(open("utils/KoGEM_info.json", "r"))


def get_tokenizer_and_model(torch_model_name):
    if 'KoGPT' in torch_model_name:
        access_token = open("api_tokens/hf_token.txt", "r").read().strip()
        tokenizer = AutoTokenizer.from_pretrained(
            'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
            bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]',
            token=access_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
            pad_token_id=tokenizer.eos_token_id,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            token=access_token
        ).to(device='cuda', non_blocking=True)
        _ = model.eval()
    else:
        #################################
        #       Set the Tokenizer
        #################################
        if 'KORani-v3' in torch_model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                torch_model_name,  # or float32 version: revision=KoGPT6B-ryan1.5b
                use_fast=False,
                padding_side="right",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                torch_model_name,  # or float32 version: revision=KoGPT6B-ryan1.5b
            )

        #################################
        #       Set the Model
        #################################
        if 'flan' in torch_model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                torch_model_name,  # or float32 version: revision=KoGPT6B-ryan1.5b
                pad_token_id=tokenizer.eos_token_id,
                device_map='auto'
            )
        elif 'polyglot' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,  # or float32 version: revision=KoGPT6B-ryan1.5b
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
        elif 'SOLAR' in torch_model_name or 'EEVE' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,  # or float32 version: revision=KoGPT6B-ryan1.5b
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map='auto',
            )
        elif (('llama-3' in torch_model_name.lower()) or
              ('KORani-v1' in torch_model_name) or ('KORani-v3' in torch_model_name) or
              ('KULLM3' in torch_model_name)):
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif 'KoAlpaca' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif 'EXAONE' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif 'deepseek' in args.torch_model_name:       # It must be located in advance of Qwen.
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype="auto",
                device_map="auto",
                pad_token_id=tokenizer.eos_token_id,
            )
        elif 's1' in args.torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif 'Qwen' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        elif 'gemma' in torch_model_name or 'mistral' in torch_model_name:
            model = AutoModelForCausalLM.from_pretrained(
                torch_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            raise NotImplementedError(f"Model name {torch_model_name} is not implemented. Please enter a valid model name.")

    _ = model.eval()
    return tokenizer, model


def load_dataset():
    """
    ( #class = 5 )
    " NUAT(HS1)  ||  NUAT(HS2)  ||  NUAT(HS3)  ||  CSAT "

    ( #class = 4 )
    " HSQE  ||  LCSE(G9)  ||  LCSE(G7)  ||  NCSE(G9)  ||  NCSE(G7) "
    """
    total_dataset = json.load(open(f"datasets/KoGEM_benchmark.json", "r"))
    return total_dataset


def make_prompts(args, data, examples):
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

    if args.shot_num != 0:
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

            if ex['data_src'] in ['NUAT(HS1)', 'NUAT(HS2)', 'NUAT(HS3)', 'CSAT']:
                ex_cand_num = 5
            elif ex['data_src'] in ['HSQE', 'LCSE(G9)', 'LCSE(G7)', 'NCSE(G9)', 'NCSE(G7)']:
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
                    ex_prompt = f"지문: {c} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "
                else:
                    ex_prompt = f"지문: {c} 설명: {p} 질문: 다음 선택지 1 부터 {ex_cand_num} 중 {q}\n 선택지: {cands}\n 정답: "

            example_prompts.append(ex_prompt)
            example_answers.append(str(ans))
    else:
        example_prompts = []
        example_answers = []

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

    if args.cot:
        prompt_extended = prompt[: -len(" 정답: ")] + args.cot_prompt + "\n"
    else:
        prompt_extended = None

    label = str(data['label'])
    return cand_num, pre_prompt, prompt, prompt_extended, label, example_prompts, example_answers


def get_messages(args, pre_prompt, prompt, prompt_extended, example_prompts, example_answers):
    if (('llama-3' in args.torch_model_name.lower()) or
            ('KORani-v3' in args.torch_model_name) or
            ('KULLM3' in args.torch_model_name)):
        if args.shot_num != 0:
            messages = [
                {"role": "system",
                 "content": f"{pre_prompt}"}
            ]
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages.extend([
                    {"role": "user",
                     "content": ex_prompt},
                    {"role": "assistant",
                     "content": ex_ans}
                ])
            messages.extend([
                {"role": "user",
                 "content": prompt},
            ])
        else:
            if args.cot:
                messages = [
                    {"role": "system",
                     "content": f"{pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt_extended}"},
                ]
            else:
                messages = [
                    {"role": "system",
                     "content": f"{pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt}"},
                ]
    elif 'gemma' in args.torch_model_name:
        if args.shot_num != 0:
            messages = [
                {"role": "user",
                 "content": f"{pre_prompt}"}
            ]
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages.extend([
                    {"role": "assistant",
                     "content": ex_prompt},
                    {"role": "user",
                     "content": ex_ans}
                ])
            messages.extend([
                {"role": "assistant",
                 "content": prompt},
            ])
        else:
            if args.cot:
                messages = [
                    {"role": "user",
                     "content": f"{pre_prompt + prompt_extended}"
                     },
                ]
            else:
                messages = [
                    {"role": "user",
                     "content": f"{pre_prompt + prompt}"},
                ]
    elif 'mistral' in args.torch_model_name:
        if args.shot_num != 0:
            messages = [
                {"role": "system",
                 "content": f"{pre_prompt}"}
            ]
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages.extend([
                    {"role": "user",
                     "content": ex_prompt},
                    {"role": "assistant",
                     "content": ex_ans}
                ])
            messages.extend([
                {"role": "user",
                 "content": prompt},
            ])
        else:
            if args.cot:
                messages = [
                    {"role": "system",
                     "content": f"{pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt_extended}"},
                ]
            else:
                messages = [
                    {"role": "system",
                     "content": f"{pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt}"},
                ]
    elif 'deepseek' in args.torch_model_name:       # It must be located in advance of Qwen.
        if args.shot_num != 0:
            messages = [
                {"role": "user",
                 "content": f"{pre_prompt}"}
            ]
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages.extend([
                    {"role": "user",
                     "content": ex_prompt},
                    {"role": "user",
                     "content": ex_ans}
                ])
            messages.extend([
                {"role": "user",
                 "content": prompt},
            ])
        else:
            if args.cot:
                messages = [
                    {"role": "user",
                     "content": f"{pre_prompt + prompt_extended}"
                     },
                ]
            else:
                messages = [
                    {"role": "user",
                     "content": f"{pre_prompt + prompt}"},
                ]
        messages[-1]["content"] += f" 다른 과정 필요 없이, 최종 답변: [integer], 해설: [string] 형태의 한국어로만 답해줘. <think>\n"

    elif 'EXAONE' in args.torch_model_name or 'Qwen' in args.torch_model_name or 's1' in args.torch_model_name:
        if 'EXAONE' in args.torch_model_name:
            set_role = "You are EXAONE model from LG AI Research, a helpful assistant."
        elif 'Qwen' in args.torch_model_name:
            set_role = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        elif 's1' in args.torch_model_name:
            set_role = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
        else:
            raise NotImplementedError

        if args.shot_num != 0:
            messages = [
                {"role": "system",
                 "content": f"{set_role + ' ' + pre_prompt}"}
            ]
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages.extend([
                    {"role": "user",
                     "content": ex_prompt},
                    {"role": "assistant",
                     "content": ex_ans}
                ])
            messages.extend([
                {"role": "user",
                 "content": prompt},
            ])
        else:
            if args.cot:
                messages = [
                    {"role": "system",
                     "content": f"{set_role + ' ' + pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt_extended}"},
                ]
            else:
                messages = [
                    {"role": "system",
                     "content": f"{set_role + ' ' + pre_prompt}"},
                    {"role": "user",
                     "content": f"{prompt}"},
                ]
    else:
        if args.shot_num != 0:
            messages = pre_prompt + '\n'
            for ex_prompt, ex_ans in zip(example_prompts, example_answers):
                messages += ex_prompt + '\n' + ex_ans + '\n'
            messages += prompt
        else:
            if args.cot:
                messages = pre_prompt + '\n' + prompt_extended
            else:
                messages = pre_prompt + '\n' + prompt
    return messages


def get_response(args, messages, model, tokenizer):
    if ('llama-3' in args.torch_model_name.lower()) or ('KORani-v3' in args.torch_model_name) or ('KULLM3' in args.torch_model_name):
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            tokens,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            # do_sample=False,
            top_p=None,
            temperature=1e-10,
            repetition_penalty=args.repeat_penalty
        )
        outputs = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokens, outputs)
        ]
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # For processing of the output of 'KORani-v3'
        if '\n### Assistant:' in generated_text:
            generated_text = generated_text[generated_text.find('\n### Assistant:') + len('\n### Assistant:'):]
        elif '\n### Human:' in generated_text:
            generated_text = generated_text[generated_text.find('\n### Human:') + len('\n### Human:'):]
        else: pass
    elif 'EXAONE' in args.torch_model_name:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        outputs = model.generate(
            tokens.to(model.device),
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1e-10,
            # do_sample=False,
        )
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        ender = "[|assistant|]"
        input_len = []
        for message in messages:
            cur_message_len = len(f"[|{message['role']}|]\n") + len(message['content'])
            input_len.append(cur_message_len)
        generated_text = generated[sum(input_len) + len(ender):].strip()
    elif 'deepseek' in args.torch_model_name:       # It must be located in advance of Qwen.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id
            # do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif 's1' in args.torch_model_name:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            temperature=1e-10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif ('Qwen' in args.torch_model_name) or ('gemma' in args.torch_model_name):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=1e-10 if 'Qwen' in args.torch_model_name else 0,
            # do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        tokens = tokenizer.encode(messages, return_tensors='pt').to(device='cuda', non_blocking=True)
        # tokens = tokenizer(messages, return_tensors='pt').to(device='cuda', non_blocking=True)
        gen_tokens = model.generate(tokens,
                                    max_new_tokens=args.max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id,
                                    repetition_penalty=args.repeat_penalty,
                                    temperature=1e-10,
                                    # do_sample=False
                                    )
        generated = tokenizer.batch_decode(gen_tokens)[0]
        # generated_text = generated[-max_new_tokens:]
        if 'EEVE' in args.torch_model_name:
            generated_text = generated[len(f"{tokenizer.bos_token} ") + len(messages):]
        else:
            generated_text = generated[len(messages):]
    # print(f"Generated text: {generated_text}")
    return generated_text


def zeroshot_eval(args, total_dataset, tokenizer, model):
    init_random(args.seed)

    st_time = time()

    batch_num = len(total_dataset) // args.batch_size + 1
    cot_file_index = "_cot" if args.cot else ""

    cannot_generate = 0
    with torch.no_grad():
        for i in range(batch_num):
            if args.continue_batch_num and i < args.continue_batch_num:
                continue

            dataset = total_dataset[i * args.batch_size: (i+1) * args.batch_size]
            if args.shot_num != 0:
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
                elif type(args.shot_num) == int and args.shot_num != 0:
                    random_ids = [random.randint(0, len(dataset)) for _ in range(args.shot_num)]
                else:
                    raise NotImplementedError

                examples = [dataset[i] for i in random_ids]
                new_dataset = []
                for idx, data in enumerate(dataset):
                    if idx not in random_ids:
                        new_dataset.append(data)

                dataset = new_dataset
            else:
                examples = None

            ##################################
            #        Prompt Evaluation       #
            ##################################
            for n, data in tqdm(enumerate(dataset), total=len(dataset),
                                desc=f"({i + 1}/{batch_num}) th Generating answers using '{args.torch_model_name}' model with {args.shot_num}-shot{cot_file_index} eval...",
                                bar_format="{l_bar}{bar:15}{r_bar}"):
                cand_num, pre_prompt, prompt, prompt_extended, label, example_prompts, example_answers = make_prompts(args, data, examples)

                messages = get_messages(args, pre_prompt, prompt, prompt_extended, example_prompts, example_answers)
                generated_text = get_response(args, messages, model, tokenizer)
                if args.cot:
                    cot_answer = generated_text
                    prompt_extended += cot_answer + "\n" + " 정답: "
                    messages = get_messages(args, pre_prompt, prompt, prompt_extended, example_prompts, example_answers)
                    generated_text = get_response(args, messages, model, tokenizer)

                if generated_text.strip() == '':
                    ans = '-1'
                else:
                    if 'deepseek' in args.torch_model_name:
                        ans_start_idx = generated_text.find("</think>")
                        if ans_start_idx == -1:
                            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                            generated_ids = model.generate(
                                **model_inputs,
                                max_new_tokens=2000,
                                temperature=0.6,
                                pad_token_id=tokenizer.eos_token_id
                                # do_sample=False
                            )
                            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in
                                zip(model_inputs.input_ids, generated_ids)
                            ]
                            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            ans_start_idx = generated_text.find("</think>")

                        ans_part = generated_text[ans_start_idx+len("</think>"):].strip()
                        print(ans_part)
                        for char in ans_part:
                            if char.isnumeric() and char in [str(n) for n in range(cand_num, 0, -1)]:
                                ans = char
                                break
                            else:
                                ans = '-1'
                                cannot_generate += 1
                    elif 's1' in args.torch_model_name:
                        ans_start_idx = generated_text.find("answer")
                        if ans_start_idx == -1:
                            text = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                            generated_ids = model.generate(
                                **model_inputs,
                                max_new_tokens=2000,
                                temperature=1e-10,
                            )
                            generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                            ]
                            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            ans_start_idx = generated_text.find("answer")

                        ans_part = generated_text[ans_start_idx + len("answer"):].strip()
                        print(f"ans_part: {ans_part}")
                        for char in ans_part:
                            if char.isnumeric() and char in [str(n) for n in range(cand_num, 0, -1)]:
                                ans = char
                                break
                            else:
                                ans = '-1'
                                cannot_generate += 1
                    else:
                        print(generated_text)
                        for char in generated_text:
                            if char.isnumeric() and char in [str(n) for n in range(cand_num, 0, -1)]:
                                ans = char
                                break
                            else:
                                ans = '-1'
                                cannot_generate += 1

                data['prediction'] = ans
                data['generated_ans'] = generated_text
                if "deepseek" in args.torch_model_name or 's1' in args.torch_model_name:
                    data["answer_part"] = ans_part
                if args.cot:
                    data['cot_answer'] = cot_answer

            ###################################
            #       Save the Predictions      #
            ###################################
            json.dump(dataset,
                      open(os.path.join(args.output_dir, f"{args.shot_num}_shot{cot_file_index}_predictions_{i}th.json"), "w", encoding="utf-8"),
                      ensure_ascii=False,
                      indent=2
                      )

    elapsed_time = time() - st_time

    # Aggregate all saved files
    all_dataset = []
    for i in range(batch_num):
        dataset = json.load(open(os.path.join(args.output_dir, f"{args.shot_num}_shot{cot_file_index}_predictions_{i}th.json"), "r"))
        all_dataset.extend(dataset)

    cor_cnt = 0
    for data in all_dataset:
        if int(data['prediction']) == int(data['label']):
            cor_cnt += 1

    print(f"Zero-shot Accuracy: {cor_cnt / len(all_dataset) * 100:.2f} [%]")
    print(f"Cannot generate: {cannot_generate}/ {len(all_dataset)}")
    print(f"Elapsed time: {elapsed_time:.2f} [s]")
    json.dump(all_dataset,
              open(os.path.join(args.output_dir, f"{args.shot_num}_shot{cot_file_index}_predictions.json"), "w", encoding="utf-8"),
              ensure_ascii=False,
              indent=2
              )
    # remove previous files
    for i in range(batch_num):
        os.remove(os.path.join(args.output_dir, f"{args.shot_num}_shot{cot_file_index}_predictions_{i}th.json"))

    print("All predictions are saved.")
    print("Done.")


if __name__ == "__main__":
    access_token = open("api_tokens/hf_token.txt", "r").read().strip()
    login(token=access_token)

    ######################################
    #           Default Settings         #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="debug")
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=56789)
    parser.add_argument('--torch_model_name', type=str, default="simplescaling/s1-32B",
                        choices=["upstage/SOLAR-10.7B-Instruct-v1.0",
                                 "yanolja/EEVE-Korean-10.8B-v1.0",
                                 "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
                                 "MLP-KTLim/llama-3-Korean-Bllossom-8B",
                                 "nlpai-lab/KULLM3",
                                 "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
                                 "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                                 "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
                                 "Qwen/Qwen2.5-7B-Instruct",
                                 "Qwen/Qwen2.5-14B-Instruct",
                                 "Qwen/Qwen2.5-32B-Instruct",
                                 "google/gemma-2-9b-it",
                                 "google/gemma-2-27b-it",
                                 "meta-llama/Llama-3.1-8B-Instruct",
                                 # "meta-llama/Llama-3.1-70B-Instruct",
                                 "meta-llama/Llama-3.2-3B-Instruct",
                                 # "meta-llama/Llama-3.3-70B-Instruct",
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                                 "mistralai/Mistral-Small-24B-Instruct-2501",
                                 "simplescaling/s1-32B"
                                 ])
    parser.add_argument('--repeat_penalty', type=float, default=1.05)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--shot_num', type=Union[str, int], default=0, help="0  ||  1  ||  5  ||  each_major_one  ||  each_sub_one")
    parser.add_argument('--cot', type=_bool, default=False)
    parser.add_argument('--continue_batch_num', type=int, default=0)
    args = parser.parse_args()

    model_name = args.torch_model_name.split("/")[-1]
    args.output_dir = f"logs/{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    ################################
    #          Load Dataset        #
    ################################
    dataset = load_dataset()

    ######################################
    #       Load Model & Tokenizer       #
    ######################################
    tokenizer, model = get_tokenizer_and_model(args.torch_model_name)

    ##################################
    #       Zero-shot learning       #
    ##################################
    zeroshot_eval(args, dataset, tokenizer, model)
