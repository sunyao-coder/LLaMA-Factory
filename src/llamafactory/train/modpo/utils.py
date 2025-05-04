from abc import ABC, abstractclassmethod
from dataclasses import dataclass, asdict
from typing import Any, Text, List, Dict, Optional

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase, PreTrainedModel


@dataclass
class RewardWrapperInput:
    raw_prompt: List[str]
    response: List[str]


@dataclass
class RewardWrapperBase(ABC):
    @abstractclassmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError


@dataclass
class RewardWrapperList:
    reward_wrapper_list: List[RewardWrapperBase]

    def map(self, func):
        for i in range(len(self.reward_wrapper_list)):
            self.reward_wrapper_list[i] = func(self.reward_wrapper_list[i])
        return self

    def __len__(self):
        return len(self.reward_wrapper_list)


@dataclass
class ImplicitRewardWrapper(RewardWrapperBase):
    """
    An implicit reward model parameterized as r(x,y) = logp(y|x)-logp_{ref}(y|x)
    """
    model: PreTrainedModel
    ref_model: PreTrainedModel

    def __call__(self, inputs: list) -> None:
        pass


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_flash_attention_2=True, # flash attn
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    implicit_reward = ImplicitRewardWrapper(
        model=model,
        ref_model=model,
        tokenizer=tokenizer,
    )

    implicit_reward({"raw_prompt": ["who are you", "hi"], "response": ["i am your dad", "goodbye"]})
    breakpoint()
