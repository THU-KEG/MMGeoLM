
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
import pdb


import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pdb
from multiprocessing import Process, Queue
import multiprocessing

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_chunk(chunk, args, queue):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, cache_dir=args.cache_dir)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    answers = []
    for line in tqdm(chunk):

        image_file = line['image_path']

        qs = line["prompt"]
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]


        with open(answers_file, "a", encoding='utf-8') as ans_file:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            line['text'] = outputs
            print(outputs)
            line['prompt'] = cur_prompt
            ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")


def eval_model(args):
    disable_torch_init()

    with open(args.question_file, 'rt', encoding='utf-8') as f:
        questions = json.load(f)

    chunks = split_list(questions, args.num_chunks)
    processes = []
    queue = Queue()

    for i in range(args.num_chunks):
        p = Process(target=process_chunk, args=(chunks[i], args, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cache-dir", type=str, default="answer.jsonl")
    
    args = parser.parse_args()

    eval_model(args)


