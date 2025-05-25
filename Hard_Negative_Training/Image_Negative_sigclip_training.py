#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train a CLIP-like dual encoder model using text and vision encoders.
Supports various vision and text models from HuggingFace Transformers.
"""

import os
import sys
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoProcessor,
    AltCLIPModel,
    AltCLIPConfig,
    SiglipModel,
    SiglipConfig,
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndProjection,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"
try:
    import wandb
    wandb.init = lambda *args, **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
except ImportError:
    pass

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Wandb has been disabled for this run.")

@dataclass
class AltCLIPOutput(ModelOutput):
    """
    Output for AltCLIP-like models.
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: Optional[BaseModelOutputWithPooling] = None
    vision_model_output: Optional[BaseModelOutputWithPooling] = None

    def to_tuple(self):
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

def contrastive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)

def clip_loss(similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return contrastive_loss(similarity, labels)

def process_text(conversations):
    for conv in conversations:
        if conv.get('from') == 'gpt':
            return [conv['value']]
    return ''

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch, tag, or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer (backed by the tokenizers library)."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code from the Hub."},
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters."}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    extract_train_data: bool = field(
        default=False, metadata={"help": "Whether to extract data from the training file."}
    )
    image_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder containing the images."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file (a jsonlines file)."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input testing data file (a jsonlines file)."}
    )
    topk_negative: Optional[int] = field(
        default=100, metadata={"help": "The number of negative samples for topk."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples to this value if set."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, data_args, processor: AutoImageProcessor):
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, "r") as f:
                list_data_dict = json.load(f)
        elif isinstance(data_path, list):
            list_data_dict = data_path
        else:
            raise ValueError("data_path must be a string path to a JSON file or a list of data")
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.processor = processor
        self.image_folder = self.data_args.image_folder

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        image_file = self.list_data_dict[i]['positive_image_path']
        image_true = Image.open(os.path.join(image_file)).convert('RGB')
        negative_files = self.list_data_dict[i]['negative_image_path'][:self.data_args.topk_negative]
        image_false = [Image.open(os.path.join(neg_file)).convert('RGB') for neg_file in negative_files]
        input_text_true = [self.list_data_dict[i]['conversations'][1]['value']]
        input_text = input_text_true
        images = [image_true] + image_false
        label = torch.tensor(images.index(image_true), dtype=torch.long)
        inputs = self.processor(text=input_text, images=images, return_tensors="pt", padding=True)
        return dict(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], labels=label)

def collate_fn(examples, tokenizer):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    input_ids = [torch.tensor(example['input_ids'][0], dtype=torch.long) for example in examples]
    input_ids_padded = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input_ids_padded = input_ids_padded[:, :tokenizer.model_max_length]
    labels = torch.stack([example["labels"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids_padded,
        "return_loss": True,
        "labels": labels
    }

class ImageNegativeModel(SiglipModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        self.logit_bias = nn.Parameter(torch.randn(1))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.shape
        batch_size, n, channel, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * n, channel, height, width)
        input_ids = input_ids.view(-1, seq_len)
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, seq_len)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds.view(batch_size, n, -1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.zeros(batch_size * n, device=image_embeds.device)
        for i in range(batch_size):
            sim = torch.matmul(image_embeds[i], text_embeds[i].t()) * logit_scale + self.logit_bias
            logits_per_image[i * n:(i + 1) * n] = sim.squeeze(-1)
        logits_per_image = logits_per_image.view(batch_size, n)
        logits_per_text = logits_per_image.t()
        loss = None
        # Siglip loss
        if return_loss:
            eye = -torch.ones_like(logits_per_image, device=image_embeds.device)
            eye[range(batch_size), labels] = 1
            loglik = torch.nn.functional.logsigmoid(eye * logits_per_image)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
        # ALICLIP loss
        # if return_loss:
        #     loss = clip_loss(logits_per_text, labels)

        if not return_dict:
            output = (logits_per_image, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return AltCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_clip", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = ImageNegativeModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config = model.config

    train_dataset = LazySupervisedDataset(
        data_path=data_args.train_file,
        data_args=data_args,
        processor=processor
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)
    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    set_seed(training_args.seed)

    image_transformations = Transform(
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    image_transformations = torch.jit.script(image_transformations)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    finetuned_from = model_args.model_name_or_path
    if os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()