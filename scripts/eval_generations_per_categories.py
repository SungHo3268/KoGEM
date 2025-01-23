import json

"""
[ API CALL ]
OpenAI
: gpt-3.5-turbo-0125  ||  gpt-4o  ||  gpt-4o-mini  ||  o1-preview

Claude
: claude-3-5-sonnet-20240620  ||  claude-3-haiku-20240307

Gemini
: gemini-2.0-flash-exp || gemini-1.5-flash

Llama
: llama3-70b  ||  llama3.1-405b

ClovaX
: HCX-DASH-001  ||  HCX-003

[ HF CALL ]
EXAONE-3.0-7.8B-Instruct
EXAONE-3.5-7.8B-Instruct
EXAONE-3.5-32B-Instruct
llama-3-Korean-Bllossom-8B
SOLAR-10.7B-Instruct-v1.0
EEVE-Korean-10.8B-v1.0
EEVE-Korean-Instruct-10.8B-v1.0
KULLM3
Qwen2.5-32B-Instruct
"""

model_type = "ClovaX"
model_var = "HCX-003"
eval_method = "0"          # 0, 1, 5, 'each_major_one', 'each_sub_one'
cot = False
cot_file_index = "_cot" if cot else ""


if model_type in ["OpenAI", "Claude", "Gemini", "Llama", "ClovaX"]:
    dataset = json.load(open(f"logs/{model_type}/{model_var}_{eval_method}_shot{cot_file_index}_predictions.json", "r"))
else:
    dataset = json.load(open(f"logs/{model_type}/{eval_method}_shot{cot_file_index}_predictions.json", "r"))



major = ["음운론", "형태론", "통사론", "의미론", "규범", "복합"]
sub = ["음운체계", "음운변동", "형태소", "품사", "단어의 짜임", "문장의 짜임", "문법요소", "단순어휘", "어휘의미론", "화용론",
       "표준어", "맞춤법", "로마자표기법", "표준발음법", "외래어표기법", "규범 복합", "복합"]

major_sub_map = {"음운론": ["음운체계", "음운변동"],
                 "형태론": ["형태소", "품사", "단어의 짜임"],
                 "통사론": ["문장의 짜임", "문법요소"],
                 "의미론": ["단순어휘", "어휘의미론", "화용론"],
                 "규범": ["표준어", "맞춤법", "로마자표기법", "표준발음법", "외래어표기법", "규범 복합"],
                 "복합": ["복합"]
                 }

major_results = {cat: [0, 0] for cat in major}
sub_results = {cat: [0, 0] for cat in sub}
for data in dataset:
    label = int(data['label'])
    try:
        prediction = int(data['prediction'])
    except ValueError:
        prediction = -1
    level_1 = data['level_1']
    level_2 = data['level_2']

    cor = (label == prediction) * 1
    major_results[level_1][0] += cor
    major_results[level_1][1] += 1

    sub_results[level_2][0] += cor
    sub_results[level_2][1] += 1



print("############################################")
print(f"[ Data Source: All ]")
if model_var:
    print(f" - Model: {model_type} ({model_var})")
else:
    print(f" - Model: {model_type}")
print("############################################")
firs_line = ""
second_line = ""
only_major = ""
for maj in major_sub_map:
    print(f"{maj}: ", end="")
    for sub in major_sub_map[maj]:
        cor, total = sub_results[sub]
        print(f"{cor/total*100:.2f}", end="\t")
        if maj in ["음운론", "형태론", "통사론"]:
            firs_line += f"{cor/total*100:.2f}\t"
        else:
            second_line += f"{cor/total*100:.2f}\t"

    major = major_results[maj]
    only_major += f"{major[0] / major[1] * 100:.2f}\t"
    if maj == '복합':
        print("")
    else:
        cor, total = major_results[maj]
        print(f"{cor/total*100:.2f}")
        if maj in ["음운론", "형태론", "통사론"]:
            firs_line += f"{cor/total*100:.2f}\t"
        else:
            second_line += f"{cor/total*100:.2f}\t"

print(f"\n총계: {sum([major_results[cat][0] for cat in major_results])/sum([major_results[cat][1] for cat in major_results])*100:.2f}")
print(f"총계 (w/o 복합): {sum([major_results[cat][0] for cat in major_results if cat != '복합'])/sum([major_results[cat][1] for cat in major_results if cat != '복합'])*100:.2f}\n\n")
second_line += f"{sum([major_results[cat][0] for cat in major_results])/sum([major_results[cat][1] for cat in major_results])*100:.2f}"
only_major += f"{sum([major_results[cat][0] for cat in major_results])/sum([major_results[cat][1] for cat in major_results])*100:.2f}"

print(f"For Excel Copy-Paste")
print(firs_line[:-1])
print(second_line)
print(only_major)



# print("Total Results")
# cor = sum([major_results[cat][0] for cat in major_results])
# total = sum([major_results[cat][1] for cat in major_results])
# print(f"{cor}/{total} ({cor/total*100:.2f} [%])")
# print("\n")
#
# print("Major Categories")
# for cat in major_results:
#     cor, total = major_results[cat]
#     print(f"{cat}: {cor}/{total} ({cor/total*100:.2f} [%])")
# print("\n")
#
# print("Sub Categories")
# for cat in sub_results:
#     cor, total = sub_results[cat]
#     print(f"{cat}: {cor}/{total} ({cor/total*100:.2f} [%])")
# print("\n")
