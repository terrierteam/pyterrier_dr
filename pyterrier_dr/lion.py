import torch 
from transformers import T5ForConditionalGeneration, LlamaForCausalLM, BertForMaskedLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import ujson
from huggingface_hub import hf_hub_download
import os 
class LLM2Retriever(torch.nn.Module):
    _tied_weights_keys = None
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def encode(self, **inputs):
        raise NotImplementedError
    
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        rank_loss = self.loss_fn(logits, labels)
        
        query_reg_loss = self.reg_loss(query_reps)
        doc_reg_loss = self.reg_loss(context_reps) 
        
        return {
            "rank": rank_loss,
            "query_reg": query_reg_loss,
            "doc_reg": doc_reg_loss
        }    
    
    def doc_encode(self, **inputs):
        return self.encode(**inputs)
    
    def query_encode(self, **inputs):
        return self.encode(**inputs)
    
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    @classmethod 
    def build(cls, model_name_or_path, args, config=None):
        if config is not None:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            model_config.update(config)
            print("to modify model config: ", config)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, config=model_config)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if args.lora:
            lora_config = LoraConfig(
                    base_model_name_or_path=args.model_name_or_path,
                    task_type=None,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=cls.TARGET_MODULES,
                    inference_mode=False,
                    modules_to_save=args.lora_modules_to_save
                )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(lora_model) 
            lora_model.print_trainable_parameters()
        else:
            model = cls(base_model)
            
        return model
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False,
             access_token=None):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path,
                                                         access_token=access_token)
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, 
                                                   lora_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=is_trainable)
            if merge_peft:
                lora_model = lora_model.merge_and_unload()
            else:
                lora_model.print_trainable_parameters()
            model = cls(lora_model)
        else:
            model = cls(base_model)
        
        return model

    @classmethod
    def load_from_lora(cls,
                       lora_name_or_path,
                       merge_peft=True, 
                       is_trainable=False,
                       access_token=None):
        if os.path.isdir(lora_name_or_path):
            adapter_config_path = os.path.join(lora_name_or_path, "adapter_config.json")
        else:
            adapter_config_path = hf_hub_download(lora_name_or_path, "adapter_config.json")
            
        with open(adapter_config_path, "r") as f:
            adapter_config = ujson.load(f)
            
        base_model_name_or_path = adapter_config["base_model_name_or_path"]
        return cls.load(base_model_name_or_path, 
                        lora_name_or_path=lora_name_or_path, 
                        merge_peft=merge_peft, 
                        is_trainable=is_trainable,
                        access_token=access_token)
            
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

class DecoderOnlyBiDense(LLM2Retriever):
    def __init__(self, base_model, T=0.01):
        super().__init__(base_model)
        self.hidden_size = self.base_model.config.hidden_size
        self.T = T
        
    def forward(self, **inputs):
        query_reps = self.encode(**inputs["tokenized_queries"]) #[n_query, D]
        context_reps = self.encode(**inputs["tokenized_contexts"]) #[n_context, D]
        labels = inputs["target_labels"] #[n_query]
        
        n_query = query_reps.size(0)
        n_context = context_reps.size(0) 
        assert n_context % n_query == 0, (n_context, n_query)
        if self.world_size > 1:
            query_reps = self.gather(query_reps)
            context_reps = self.gather(context_reps)
            labels = self.gather(labels)
            base = torch.repeat_interleave(torch.arange(self.world_size), n_query) * n_context
            labels = labels + base.to(labels.device)
            
        logits = torch.matmul(query_reps, context_reps.transpose(1,0))
        
        rank_loss = self.loss_fn(logits / self.T, labels)
        
        return rank_loss
    
    def rerank_forward(self, **inputs):
        # it is just for evaluation
        query_reps = self.encode(**inputs["tokenized_queries"])
        doc_reps = self.encode(**inputs["tokenized_docs"])
        logits = (query_reps * doc_reps).sum(dim=-1)
        return logits
    
    def encode(self, **inputs):
        # since we do left padding, and add the cls_token_id to the last position.
        # but we make sure that it is correctly implemented 
        #seq_reps = self.base_model(**inputs, return_dict=True).last_hidden_state 
        #reps = seq_reps[:, -1] #[bz, dim]
        
        # we do average embedding
        # padding_size is from left 
        seq_lengths = inputs["attention_mask"].sum(dim=-1)
        seq_reps = self.base_model(**inputs, return_dict=True).last_hidden_state 
        seq_reps = torch.nn.functional.normalize(seq_reps, p=2, dim=-1)
        reps = torch.stack(
                [
                    seq_reps[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
        )
        
        return reps 
    
    @classmethod 
    def build(cls, model_name_or_path, args, config=None):
        if config is not None:
            model_config = AutoConfig.from_pretrained(model_name_or_path)
            model_config.update(config)
            print("to modify model config: ", config)
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, config=model_config)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
        
        if args.lora:
            lora_config = LoraConfig(
                    base_model_name_or_path=args.model_name_or_path,
                    task_type=None,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=cls.TARGET_MODULES,
                    inference_mode=False,
                    modules_to_save=args.lora_modules_to_save
                )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(lora_model, T=args.T) 
            lora_model.print_trainable_parameters()
        else:
            model = cls(base_model, T=args.T)
            
        return model
    
    @classmethod 
    def load(cls, 
             model_name_or_path, 
             lora_name_or_path=None, 
             merge_peft=True,
             is_trainable=False,
             T=0.01,
             access_token=None):
        if lora_name_or_path is not None:
            # It is hacky here, but we need to check wether the lora_name_or_path is with the expected format
            from safetensors.torch import load_file
            import os
            if os.path.isdir(lora_name_or_path):
                if os.path.exists(os.path.join(lora_name_or_path, "adapter_model.safetensors")):
                    tmp_state_dict = load_file(os.path.join(lora_name_or_path, "adapter_model.safetensors"))
                elif os.path.exists(os.path.join(lora_name_or_path, "adapter_model.bin")):
                    tmp_state_dict = torch.load(os.path.join(lora_name_or_path, "adapter_model.bin"))
            else: 
                tmp_model_bin = hf_hub_download(lora_name_or_path, "adapter_model.bin")
                tmp_state_dict = torch.load(tmp_model_bin)
            assert "base_model.model.model.layers" not in list(tmp_state_dict.keys())[0]
            assert "base_model.model.layers" in list(tmp_state_dict.keys())[0]
            tmp_state_dict = None
                
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path)
                                                         #access_token=access_token)
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, 
                                                   lora_name_or_path, 
                                                   config=lora_config,
                                                   is_trainable=is_trainable)
            if merge_peft:
                lora_model = lora_model.merge_and_unload()
            model = cls(lora_model, T=T)
            
            # we also check lorr_config here 
            assert lora_config.auto_mapping["base_model_class"] == cls.TRANSFORMER_CLS.__name__, (
                lora_config.auto_mapping["base_model_class"], cls.TRANSFORMER_CLS.__name__
            )
            if not merge_peft:
                lora_model.print_trainable_parameters()
        else:
            model = cls(base_model, T=T)
        
        return model


import torch

from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from torch import nn
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from .utils import is_transformers_attn_greater_or_equal_4_43_1

from peft import PeftModel

logger = logging.get_logger(__name__)


class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,
    "sdpa": ModifiedLlamaSdpaAttention,
}


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class LlamaBiModel(LlamaModel):
    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.43.1"
            )
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class LlamaBiDense(DecoderOnlyBiDense):
    TRANSFORMER_CLS = LlamaBiModel
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]
    

# class Qwen2BiDense(DecoderOnlyBiDense):
#     TRANSFORMER_CLS = Qwen2BiModel
#     TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "k_proj", "down_proj", "up_proj", "gate_proj"]

from .biencoder import BiEncoder
import numpy as np
from more_itertools import chunked
class LionLLamaDense(BiEncoder):

    def __init__(self, model_name="hzeng/Lion-DS-1B-llama3-marco-mntp", batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, verbose=verbose, text_field=text_field)
        from transformers import AutoTokenizer 
        self.model = LlamaBiDense.load_from_lora(model_name) 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk),  max_length=192, return_tensors='pt', padding="longest", truncation=True)
                #inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model.query_encode(**inps)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
    
    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), max_length=192, return_tensors='pt', padding='longest', truncation=True)
                #inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model.doc_encode(**inps)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
    
    
