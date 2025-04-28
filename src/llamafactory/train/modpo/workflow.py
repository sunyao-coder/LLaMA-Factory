# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional
from typing import List, Dict, Text, Tuple

from transformers import PreTrainedModel

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import MODPOTrainer
from .utils import ImplicitRewardWrapper, RewardWrapperList



if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, MarginRewardArguments, MarginRewardPair, MarginRewardPairList

def create_margin_reward_model(
    model_args: "ModelArguments",
    margin_reward_args: "MarginRewardArguments",
) -> "PreTrainedModel":
    margin_reward_model_args = ModelArguments.copyfrom(
        model_args,
        model_name_or_path=margin_reward_args.margin_reward_model,
        adapter_name_or_path=margin_reward_args.ref_model_adapters,
        quantization_bit=margin_reward_args.ref_model_quantization_bit,
    )
    margin_reward_model_finetuning_args = FinetuningArguments()
    tokenizer = load_tokenizer(margin_reward_model_args)['tokenizer']
    margin_reward_model = load_model(
        tokenizer, margin_reward_model_args, margin_reward_model_finetuning_args, is_trainable=False, add_valuehead=False
    )

    return margin_reward_model


def create_margin_reward_model_list(
    model_args: "ModelArguments",
    margin_reward_pair_list: "MarginRewardPairList",
)-> RewardWrapperList:
    reward_wrapper_list = []
    for margin_reward_pair in margin_reward_pair_list:
        if isinstance(margin_reward_pair, MarginRewardPair):
            margin_reward_model = create_margin_reward_model(model_args, margin_reward_pair.margin_reward_model)
            margin_reward_ref_model = create_margin_reward_model(model_args, margin_reward_pair.margin_reward_ref_model)

            margin_reward_model_pair = ImplicitRewardWrapper(
                model=margin_reward_model,
                ref_model=margin_reward_ref_model,
            )
            reward_wrapper_list.append(margin_reward_model_pair)
        else:
            raise ValueError(f"Unsupported margin reward pair type: {type(margin_reward_pair)}")
        
    return RewardWrapperList(reward_wrapper_list)
    

def modpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Create the reward model list
    wrapped_margin_reward_model_list = create_margin_reward_model_list(model_args, finetuning_args.margin_reward_model_list)

    # Initialize our Trainer
    trainer = MODPOTrainer(
        model=model,
        ref_model=ref_model,
        margin_reward_model_list=wrapped_margin_reward_model_list,
        w=finetuning_args.w_list,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "rewards/accuracies"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
