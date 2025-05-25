import json
import pdb
import os
import glob
import shutil
import subprocess
from tqdm import tqdm
import sys
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

client = ZhipuAI(api_key="")
wrong_code_list = []
wrong_code_list_lock = threading.Lock()
path = "/path/to/image_code"
def process_file(file):
    prompt_path = os.path.join(path, file, "prompt.json")
    png_path = os.path.join(path, file, "question.png")
    py_path = os.path.join(path, file, "question_code.py")
    #如果没有prompt说明是空文件夹
    if not os.path.exists(prompt_path):
        return
    if os.path.exists(png_path):
        os.remove(png_path)

    # 2. 运行一次生成的code（切换到py文件所在目录，保证图片生成在正确位置）
    code_dir = os.path.dirname(py_path)
    cwd = os.getcwd()
    result = subprocess.run([sys.executable, py_path], capture_output=True, text=True, cwd=code_dir)

    with open(py_path, "rt", encoding="utf-8") as f:
        code = f.read()
    # 5. 如果运行出错，则调用API修正代码，再保存并再运行一次
    if result.returncode != 0:
        # 记录第一次运行的错误日志
        log_path = os.path.join(code_dir, "run_error.log")
        with open(log_path, "wt", encoding="utf-8") as log_f:
            log_f.write("STDOUT:\n")
            log_f.write(result.stdout or "")
            log_f.write("\nSTDERR:\n")
            log_f.write(result.stderr or "")

        # 调用API修正代码
        input_prompt = """Perform the following operations on the given Python code:
        1. Check for any coding errors and fix them if found.
        2. Ensure the final diagram is saved as a PNG file in the same directory as the code. Add the saving command if it's missing.
        3. If `plt.show()` is used, remove it.
        4. Return the **modified code only** without any additional explanations or comments.
        \n{code}""".format(code=code)
        chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_prompt,
                    }
                ],
                model="glm-4-plus",
                temperature=0.01
            )
        code_fixed = chat_completion.choices[0].message.content

        if code_fixed.startswith("```python"):
            code_fixed = code_fixed[len("```python"):].strip()
            if code_fixed.endswith("```"):
                code_fixed = code_fixed[:-3].strip()
        else:
            code_fixed = code_fixed

        # 保存修正后的代码
        with open(py_path, "wt", encoding="utf-8") as f:
            f.write(code_fixed)

        # 再运行一次（同样切换到py文件目录）
        result2 = subprocess.run([sys.executable, py_path], capture_output=True, text=True, cwd=code_dir)

        # 如果第二次还出错，记录log
        if result2.returncode != 0:
            log_path2 = os.path.join(code_dir, "run_error_second.log")
            with open(log_path2, "wt", encoding="utf-8") as log_f:
                log_f.write("STDOUT:\n")
                log_f.write(result2.stdout or "")
                log_f.write("\nSTDERR:\n")
                log_f.write(result2.stderr or "")
            # 记录错误文件
            with wrong_code_list_lock:
                wrong_code_list.append(file)
        # 如果第二次运行正常则不做记录
    return

def main():
    files = os.listdir(path)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for file in files:
            futures.append(executor.submit(process_file, file))
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    with open("wrong_code_list.txt", "wt", encoding="utf-8") as f:
        for code in wrong_code_list:
            f.write(code + "\n")

if __name__ == "__main__":
    main()
