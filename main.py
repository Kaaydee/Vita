import json

# Step 1: Specify the file path
file_path = 'camera_source.json'

# Step 2: Open the file and read the JSON data
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

def find_id_by_display_name(display_name):
    for feature in data['features']:
        if feature['properties']['displayName'] == display_name:
            return feature['properties']['id']
    return None  # Trả về None nếu không tìm thấy

# Ví dụ: tìm id cho displayName "Tô Ngọc Vân - TX25"
display_name_input = "Võ Văn Kiệt - An Dương Vương 2"
id_result = find_id_by_display_name(display_name_input)

if id_result:
    print(f"ID tương ứng với '{display_name_input}' là: {id_result}")
else:
    print(f"Không tìm thấy ID cho displayName '{display_name_input}'")
