from typing import Optional, List

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from rtpt import RTPT

from reasonBias.api.languageModelInterface import LanguageModelInterface

default_opt_model = "facebook/opt-30b" #"facebook/opt-66b" #

opt_device_map = "auto"  # distribute parameter across all available gpus (and cpu if needed)
#opt_device_map = 0 #None  # single gpu


class OptLM(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = 50, dry_run: bool = False, model_name=None, max_tokens=50, chunk_batches=10):
        super().__init__(key, limit=limit, dry_run=dry_run)
        rtpt = RTPT(name_initials='MoWi', experiment_name='', max_iterations=100)
        rtpt.start()

        if model_name is None:
            model_name = default_opt_model
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.chunk_batches = chunk_batches

        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map=opt_device_map).cuda()
        # self.model.eval()

        self.config = AutoConfig.from_pretrained(self.model_name, torch_dtype=torch.float16)
        with init_empty_weights():
            emodel = AutoModelForCausalLM.from_config(self.config)

        #TODO the "OPTDecoderLayer" specific to OPT, but we might also use this class for T5 (probably has no effect there)
        self.device_map = infer_auto_device_map(emodel, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map=self.device_map).cuda()

        # the fast tokenizer currently does not work correctly
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def _make_deterministic(self, seed=123):
        set_seed(seed)

    def do_query(self, query_text: str) -> str:
        #input_ids = self.tokenizer(query_text, return_tensors="pt").input_ids.cuda(self.model.main_device) ERROR no main_device
        input_ids = self.tokenizer(query_text, return_tensors="pt").input_ids.cuda()

        # greedy generation
        generated_ids = self.model.generate(
            input_ids,
            # additional parameters for GenerationConfig
            num_return_sequences=1,
            max_new_tokens=self.max_tokens
        )

        results = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return results[0][len(query_text):]

    def do_query_variance(self, query_text: str, temperature: float, num_samples: int, **kwargs) -> List[str]:
        if self.chunk_batches is not None and num_samples > self.chunk_batches:
            res = []
            while num_samples > 0:
                res.extend(self.do_query_variance(query_text, temperature, min(self.chunk_batches, num_samples)))
                num_samples -= self.chunk_batches
            return res

        input_ids = self.tokenizer(query_text, return_tensors="pt").input_ids.cuda(0)
        #input_ids = self.tokenizer(query_text, return_tensors="pt").input_ids.cuda(self.model.main_device) ERROR no main_device

        #self._make_deterministic()  # be careful with this - if we split our queries into block, each block would hold the same result
        generated_ids = self.model.generate(
            input_ids,
            # additional parameters for GenerationConfig
            num_return_sequences=num_samples,
            max_new_tokens=self.max_tokens,
            temperature=temperature,
            do_sample=True,
            #top_p=1.0,
            #top_k=50,  # enable top-k sampling
            repetition_penalty=1.0,  # 1.0 = no penalty
            encoder_repetition_penalty=1.0,  # 1.0 = no penalty
            **kwargs
        )

        results = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        response_texts = []
        for result in results:
            response_texts.append(result[len(query_text):])
        return response_texts
