import os
import uuid
import json
import time
import requests
from tqdm import tqdm


data_name = "school_qual_exam"      # e.g., mock_high1, mock_high2, mock_high3, suneung, school_qual_exam, local_office7, local_office9, national_office7, national_office9
# affix = ""
affix = "_dropped"


data_dir = f"datasets/{data_name}/scrapped{affix}"
file_list = os.listdir(data_dir)
file_list.sort()

if data_name in ["mock_high1", "mock_high2", "mock_high3", "suneung"]:
    num_cands = 5
elif data_name in ['school_qual_exam', 'local_office7', 'local_office9', 'national_office7', 'national_office9']:
    num_cands = 4
else:
    raise NotImplementedError


for file_name in tqdm(file_list, desc="Checking file_name format...", bar_format="{l_bar}{bar:15}{r_bar}"):
    f_name, file_ext = file_name.split(".")
    year = f_name.split("_")[0]
    if data_name in ["mock_high1", "mock_high2", "mock_high3"]:
        month = f_name.split("_")[1]
        if f_name.split("_")[2] in ['A', 'B', '언매']:
            detail = f_name.split("_")[2]
        else:
            detail = ''
    elif data_name in ["suneung", "local_office7", "local_office9", "national_office7", "national_office9"]:       # 1년에 한 번
        if f_name.split("_")[1] in ['A', 'B', '언매', '가', '나', '다', '책', '책2', '인', '사', 'S',
                                    '고', '공', '봉', '인', '우', '2책', '3형']:
            detail = f_name.split("_")[1]
        else:
            detail = ''
    elif data_name in ['school_qual_exam', ]:           # 1년에 두 번
        rnd = f_name.split("_")[1]
        if f_name.split("_")[2] in ['A', 'O']:
            detail = f_name.split("_")[2]
        else:
            detail = ''
    else:
        raise NotImplementedError

    last_index = f_name.split("_")[-1]
    if last_index in ['문제', '지문']: pass
    elif last_index == 'P':
        last_index = f_name.split("_")[-2]
        if last_index in ['지문', '문제']: pass
        else: raise NotImplementedError
    elif last_index == 'I':
        last_index = f_name.split("_")[-2]
        if last_index in ['지문', '문제']: pass
        else: raise NotImplementedError
    elif last_index in ['맞춤법', '기타', '규범', 'ETC']:
        continue
    else:
        raise NotImplementedError

total_data = []
preprocess_context = False
preprocess_question = False
image_context = False
image_question = False
cur_data_set = {'num_id': '', 'context': '', 'question': '', 'paragraph': '', 'candidates': []}
for file_name in tqdm(file_list, desc="Extracting data from scrapped images", bar_format="{l_bar}{bar:15}{r_bar}"):
    f_name, file_ext = file_name.split(".")
    year = f_name.split("_")[0]
    etc = False
    if data_name in ["mock_high1", "mock_high2", "mock_high3"]:
        month = f_name.split("_")[1]
        if f_name.split("_")[2] in ['A', 'B', '언매']:
            detail = f_name.split("_")[2]
        else:
            detail = ''
    elif data_name in ["suneung", "local_office7", "local_office9", 'national_office7', 'national_office9']:       # e.g., suneung
        if f_name.split("_")[1] in ['A', 'B', '언매', '가', '나', '다', '책', '책2', '인', '사', 'S',
                                    '고', '공', '봉', '인', '우', '2책', '3형']:
            detail = f_name.split("_")[1]
        else:
            detail = ''
    elif data_name in ['school_qual_exam']:
        rnd = f_name.split("_")[1]
        if f_name.split("_")[2] in ['A', 'O']:
            detail = f_name.split("_")[2]
        else:
            detail = ''
    else:
        raise NotImplementedError

    last_index = f_name.split("_")[-1]
    if last_index in ['문제', '지문']:
        pass
    elif last_index == 'P':
        last_index = f_name.split("_")[-2]
        if last_index == '지문':
            preprocess_context = True
        elif last_index == '문제':
            preprocess_question = True
        else:
            raise NotImplementedError
    elif last_index == 'I':
        last_index = f_name.split("_")[-2]
        if last_index == '지문':
            image_context = True
        elif last_index == '문제':
            image_question = True
        else:
            raise NotImplementedError
    elif last_index in ['기타', '규범', 'ETC']:
        etc = True
        last_index = f_name.split("_")[-2]
    elif last_index == '맞춤법':
        continue
    else:
        raise NotImplementedError

    if last_index in ["문제", "지문"]:
        ####################################
        # HyperCLOVA X OCR REQUEST & PARSE #
        ####################################
        api_url = open("utils/clovax_ocr_api_url.txt", 'r').readline().strip()
        secret_key = open('utils/clovax_ocr_api_key.txt', 'r').readline().strip()
        image_file = os.path.join(data_dir, f_name + '.' + file_ext)

        request_json = {
            'images': [
                {
                    'format': 'png',
                    'name': 'demo'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = [
            ('file', open(image_file, 'rb'))
        ]
        headers = {
            'X-OCR-SECRET': secret_key
        }
        response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
        response_text = json.loads(response.text)['images'][0]['fields']
        text = [t['inferText'] for t in response_text]
        text = ' '.join(text)

        ####################################
        #      Preprocess OCR result       #
        ####################################
        if last_index == "문제":
            if cur_data_set['num_id'] != '':
                total_data.append(cur_data_set)
                cur_data_set = {'num_id': '', 'context': '', 'question': '', 'paragraph': '', 'candidates': []}

            q_end = text.find("?")+1
            question = text[: q_end].strip()
            num_start = True
            for j, q in enumerate(question):
                if num_start:
                    if q.isnumeric():
                        num_start = False
                        continue
                    else:
                        continue
                else:
                    if q.isnumeric():
                        continue
                    else:
                        if q == '.':
                            question = question[j+1:].strip()
                        else:
                            question = question[j:].strip()
                        break
            text = text[q_end:]

            paragraph = text
            for cand_id in [str(k) for k in range(num_cands+1, 0, -1)]:
                cand_start = paragraph.rfind(cand_id)
                paragraph = paragraph[:cand_start]

            text = text[len(paragraph):].strip()        # text for candidates
            paragraph = paragraph.strip()

            candidates = []
            for i in range(1, num_cands+1):
                if i == num_cands:
                    cand = text.strip()
                    try:
                        if cand[0].isnumeric() and cand[1] != '.':
                            cand = cand[0] + '. ' + cand[1:].strip()
                    except IndexError:
                        pass
                    candidates.append(cand)
                else:
                    cand_end = text.find(str(i+1))
                    cand = text[: cand_end].strip()
                    try:
                        if cand[0].isnumeric() and cand[1] != '.':
                            cand = cand[0] + '. ' + cand[1:].strip()
                    except IndexError:
                        pass
                    candidates.append(cand)
                    text = text[cand_end:]

            if image_context or image_question or preprocess_context or preprocess_question or etc:
                num = f_name.split("_")[-3]
            else:
                num = f_name.split("_")[-2]

            if data_name in ["mock_high1", "mock_high2", "mock_high3"]:
                cur_data_set['num_id'] = '-'.join([year, month, num])
                if detail:
                    cur_data_set['num_id'] = '-'.join([year, month, detail, num])
            elif data_name in ["suneung", "local_office7", "local_office9", "national_office7", "national_office9"]:
                cur_data_set['num_id'] = '-'.join([year, num])
                if detail:
                    cur_data_set['num_id'] = '-'.join([year, detail, num])
            elif data_name in ['school_qual_exam']:
                cur_data_set['num_id'] = '-'.join([year, rnd, num])
                if detail:
                    cur_data_set['num_id'] = '-'.join([year, rnd, detail, num])
            else:
                raise NotImplementedError

            if image_question:
                cur_data_set['num_id'] += '-IQ'
                image_question = False
            if preprocess_question:
                cur_data_set['num_id'] += '-PQ'
                preprocess_question = False
            if image_context:
                cur_data_set['num_id'] += '-IC'
                image_context = False
            if preprocess_context:
                cur_data_set['num_id'] += '-PC'
                preprocess_context = False

            if etc:
                cur_data_set['num_id'] += '-ETC'

            cur_data_set['question'] = question
            cur_data_set['paragraph'] = paragraph
            cur_data_set['candidates'] = candidates

        elif last_index == "지문":
            cur_data_set['context'] = text
            if cur_data_set['num_id'] != '':        # If '지문'이 '문제보다 먼저 나와서 'num_id'가 없는 경우
                if image_context:
                    cur_data_set['num_id'] += '-IC'
                    image_context = False
                if preprocess_context:
                    cur_data_set['num_id'] += '-PC'
                    preprocess_context = False
                if etc:
                    cur_data_set['num_id'] += '-ETC'
            else: pass
    else:
        raise NotImplementedError

# add last data to total_data
if cur_data_set['num_id'] != '':
    total_data.append(cur_data_set)


###################################
#       Save extracted data       #
###################################
save_dir = os.path.dirname(data_dir)
save_path = os.path.join(save_dir, f"(before)extracted{affix}.json")
json.dump(total_data, open(save_path, "w"), indent=2, ensure_ascii=False)

answers = list()
for line in total_data:
    cur_ans_data = {
        'num_id': line['num_id'],
        'label': -1
    }
    answers.append(cur_ans_data)
ans_save_path = os.path.join(save_dir, f"(before)answers{affix}.json")
json.dump(answers, open(ans_save_path, "w"), indent=2, ensure_ascii=False)
