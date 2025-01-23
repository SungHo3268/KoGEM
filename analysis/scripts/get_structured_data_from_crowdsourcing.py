import json
import pandas as pd
import numpy as np

task_name = "school_qual"    # "office" or "school_qual"
if task_name == 'office':
    dataset = pd.read_excel('datasets/human_eval/office/embrain_office_data_241111_v2.xlsx', sheet_name='raw', header=0)
    data_size = 415
elif task_name == 'school_qual':
    dataset = pd.read_excel('datasets/human_eval/school_qual/embrain_school_qual_exam_data_241210.xlsx', sheet_name='Raw', header=0)
    data_size = 178
else:
    raise NotImplementedError(f"Invalid task_name: {task_name}")

data_mapping = {
    'sex': {
        1: 'male',
        2: 'female'
    },
}

total_responses = 0
male = 0
female = 0
age_stats = {10: [], 20: [], 30: [], 40: [], 50: [], 60: [], 70: [], 80: [], 90: []}
data_per_person = []
data_per_nid = {}
for i in range(len(dataset)):
    data = dataset.iloc[i]
    pid = int(data['NO'])
    sex = data_mapping['sex'][data['SQ1']]
    age = int(data['SQ2'])

    if sex == 'male':
        male += 1
    else:
        female += 1

    age_stats[age // 10 * 10].append(age)

    responses = []
    for j in range(1, data_size + 1):
        nid = j
        if str(data[f'Q{j}']) == 'nan':
            continue
        else:
            prd = data[f'Q{j}']
            exp = data[f'Q{j}_1']
            response = {
                'nid': int(nid),
                'prd': int(prd),
                'exp': exp
            }
            responses.append(response)

            if nid not in data_per_nid:
                data_per_nid[nid] = {'pid': [pid], 'sex': [sex], 'age': [age], 'prd': [int(prd)], 'exp': [exp]}
            else:
                data_per_nid[nid]['pid'].append(pid)
                data_per_nid[nid]['sex'].append(sex)
                data_per_nid[nid]['age'].append(age)
                data_per_nid[nid]['prd'].append(int(prd))
                data_per_nid[nid]['exp'].append(exp)

    total_responses += len(responses)

    row_data = {
        'pid': pid,
        'sex': sex,
        'age': age,
        'responses': responses
    }
    data_per_person.append(row_data)

print("")
print(f"Total Responses: {total_responses}")
print("")

##########################
#       Get Answers      #
##########################
raw_details = []
cur_dump = {}
for line in open(f"datasets/total_{task_name}_que_cands_detail.txt", "r").readlines():
    line = line.strip()
    if line == "":
        if cur_dump:
            raw_details.append(cur_dump)
            cur_dump = {}
        else:
            continue
    elif ("label" in line) or ("level_" in line):
        continue
    else:
        line = [tok.strip() for tok in line.split(":")]
        if line[0] == '문항 번호':
            line[0] = 'nid'
            line[1] = int(line[1])
        elif line[0] == 'idx':
            line[1] = int(line[1])
        else:
            pass
        cur_dump[line[0]] = line[1]

total_benchmark = json.load(open("datasets/KoGram_benchmark_list.json", "r"))
for i, raw_detail in enumerate(raw_details):
    cur = total_benchmark[raw_detail['idx']]
    assert cur['num_id'] == raw_detail['num_id']
    raw_details[i]['label'] = cur['label']

    nid = raw_detail['nid']
    data_per_nid[nid]['label'] = cur['label']
    score = round(np.mean(np.array(data_per_nid[nid]['prd']) == cur['label']), 4) * 100
    data_per_nid[nid]['score'] = score
    if score == 0:
        print(f"Zero Score!!!: id = {nid}")

##########################
#       Evaluation       #
##########################
total_results = []
total_acc = []
male_acc_list = []
female_acc_list = []
age_acc_list = {10: [], 20: [], 30: [], 40: [], 50: [], 60: [], 70: [], 80: [], 90: []}
if task_name == 'office':
    src_acc_list = {'local_office9': [], 'local_office7': [], 'national_office9': [], 'national_office7': []}
elif task_name == 'school_qual':
    src_acc_list = {'school_qual_exam': []}
for i in range(len(raw_details)):
    result = {}
    
    detail = raw_details[i]
    nid = detail['nid']
    data = data_per_nid[nid]

    acc_list = np.array(data['prd']) == detail['label']

    # Accuracy per Sex
    for j, sex in enumerate(data['sex']):
        acc = data['prd'][j] == detail['label']
        if sex == 'male':
            male_acc_list.append(acc)
        else:
            female_acc_list.append(acc)

    # Accuracy per Age
    for j, age in enumerate(data['age']):
        temp_age = int(age // 10 * 10)
        acc = data['prd'][j] == detail['label']
        age_acc_list[temp_age].append(acc)

    # Accuracy per Source
    src_acc_list[detail['src']].extend(acc_list)

    result['nid'] = nid
    result['idx'] = detail['idx']
    result['src'] = detail['src']
    result['num_id'] = detail['num_id']
    result['acc'] = round(np.mean(acc_list) * 100 , 2)
    result['label'] = detail['label']
    result['pid'] = data['pid']
    result['sex'] = data['sex']
    result['age'] = data['age']
    result['prd'] = data['prd']
    result['exp'] = data['exp']
    total_results.append(result)

    total_acc.extend(acc_list)


print("")
print(f"Total Accuracy: {np.mean(total_acc) * 100:.2f}[%]")
print("")

print("[ Average Sex ]")
print(f"# male: {male}")
print(f"# female: {female}")
print("")

print("[ Accuracy per Sex ]")
male_acc = np.mean(male_acc_list)
female_acc = np.mean(female_acc_list)
age_acc = {age: -1 for age in age_acc_list}
for age in age_acc_list:
    if len(age_acc_list[age]) != 0:
        age_acc[age] = np.mean(age_acc_list[age])
print(f"male: {male_acc * 100:.2f}[%]")
print(f"female: {female_acc * 100:.2f}[%]")
print("")

print("[ Average Age ]")
for age in age_stats:
    if len(age_stats[age]) != 0:
        print(f"{age}s: {np.mean(age_stats[age]):.2f}")
print(f"total: {np.mean([np.mean(age_stats[age]) for age in age_stats if len(age_stats[age]) != 0]):.2f}")
print("")

print("[ Accuracy per Age ]")
for age in age_acc:
    if age_acc[age] >= 0:
        print(f"{age}: {age_acc[age] * 100:.2f}[%]")
print("")


for src in src_acc_list:
    acc = np.mean(src_acc_list[src])
    print(f"{src}: {acc * 100:.2f}[%]")



###########################
#       Save Results      #
###########################
json.dump(total_results, open(f"datasets/human_eval/{task_name}/total_results.json", "w"), ensure_ascii=False, indent=2)
json.dump(data_per_person, open(f"datasets/human_eval/{task_name}/data_per_person.json", "w"), ensure_ascii=False, indent=2)
json.dump(data_per_nid, open(f"datasets/human_eval/{task_name}/data_per_num_id.json", "w"), ensure_ascii=False, indent=2)
