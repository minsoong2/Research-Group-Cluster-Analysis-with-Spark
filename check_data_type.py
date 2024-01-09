import json


def check_data_format(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            # 각 키에 대한 값의 데이터 타입을 문자열로 저장
            types_str = ", ".join([f"{key}: {type(value).__name__}" for key, value in data.items()])
            output_file.write(types_str + '\n')


with open('check_data_types.txt', 'w') as f:
    check_data_format('data.txt', f)