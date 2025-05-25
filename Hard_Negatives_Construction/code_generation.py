import json
import os
import shutil

from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import random, time
from google import genai


# )
client = OpenAI(
    api_key="",
    base_url="",
)

model = "gemini-2.5-pro-preview-03-25"

def process_data(prompt, model):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion =client.chat.completions.create(
            model=model,
            messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            }
        ]
        }
            ],
            max_tokens=20000
            )
            answer = completion.choices[0].message.content
            return answer
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''


prompt = '''Based on the given question and answer, generate a geometric diagram using Python code. Please ensure the generated diagram is accurate and meets the following requirements:

1. Use only black color; do not include any colored elements.
2. Do not include a title in the figure.
3. If there are extended lines used for reasoning, ignore them in the drawing.
4. Only mark geometric elements (coordinates, lengths, angles, etc.) explicitly mentioned in the **question**. Do not include any information inferred or only mentioned in the **answer**.
5. For analytic geometry problems, include known equations if they are stated in the question.
6. Do not place any geometric elements outside the visible diagram area.
7. Save the final image as "question.png".
question:
```
{}
```
answer:
```
{}```'''

MM_Math_path = "/path/to/MM_Math.json"
image_path = "/path/to/image"
#原始数据的文件夹
with open(MM_Math_path, 'rt', encoding='utf-8') as f:
    datas = json.load(f)

all_questions = []

for key, value in datas.items():
    all_questions.append(key)


chosen_questions = all_questions

def process_question(question):
    if os.path.exists(f"/save_path/{question}"):
        return question, False
    prompt_data = prompt.format(datas[question]['question'], datas[question]['solution'])
    answer = process_data(prompt_data, model)
    if answer == '':
        return question, False

    os.makedirs(f"/save_path/{question}", exist_ok=True)
    # Get source image path
    try:
        image_question_path = os.path.join(image_path, datas[question]['image_file_name'], datas[question]['image'][0])
        # Copy image to question folder
        dest_path = os.path.join(f"/save_path/{question}")
        shutil.copy2(image_question_path, dest_path)
    except:
        print(f"Error: {question}")
    with open(f"/save_path/{question}/prompt.json", 'wt', encoding='utf-8') as f:
        json.dump(answer, f, indent='', ensure_ascii=False)
    return question, True

num = 0
results = []
with ThreadPoolExecutor(max_workers=80) as executor:
    futures = {executor.submit(process_question, question): question for question in chosen_questions}
    for future in tqdm(as_completed(futures), total=len(futures)):
        question, success = future.result()
        if success:
            num += 1
        if num >= 20:
            break

