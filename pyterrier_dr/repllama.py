from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .biencoder import BiEncoder
from .util import Variants

def _get_model(peft_model_name):
    #replace_with_xformers_attention()
    from peft import PeftModel, PeftConfig
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

def replace_with_xformers_attention():
    import torch
    import xformers.ops as xops

    from typing import Optional, Tuple
    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = None
        attn_output = xops.memory_efficient_attention(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2),
            attn_bias=xops.LowerTriangularMask()
        ).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    LlamaAttention.forward = custom_forward

class _RepLLamaBiEncoderBase(BiEncoder):
    def __init__(self, model: str, tokenizer, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = _get_model(model).to(self.device).eval()
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

    # padding handling as per https://github.com/texttron/tevatron/blob/main/examples/repllama/repllama.py

    def encode_queries_torch(self, texts, batch_size=None):
        results = []
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer([f'query: {query}</s>' for query in chunk], return_tensors='pt', padding=True, truncation=True, max_length=32)
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model(**inps, output_hidden_states=True)
            q_hidden = res.hidden_states[-1]
            attention_mask = inps['attention_mask']
            # we want the last token representation that is not padding
            sequence_lengths = attention_mask.sum(dim=1)
            last_token_indices = sequence_lengths - 1
            q_reps = q_hidden[torch.arange(q_hidden.size(0)), last_token_indices]
            q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=-1)
            results.append(q_reps)
        if not results:
            return torch.empty((0, 0))
        return torch.cat(results, dim=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                # NB: titles should be prepended in the indexing pipeline, if available
                inps = self.tokenizer([f'passage: {passage}</s>' for passage in chunk], return_tensors='pt', padding=True, truncation=True, max_length=256)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                psg_out = self.model(**inps, output_hidden_states=True)
                p_hidden = psg_out.hidden_states[-1]
                attention_mask = inps['attention_mask']
                # we want the last token representation that is not padding
                sequence_lengths = attention_mask.sum(dim=1)
                last_token_indices = sequence_lengths - 1
                p_reps = p_hidden[torch.arange(p_hidden.size(0)), last_token_indices]
                p_reps = torch.nn.functional.normalize(p_reps, p=2, dim=-1)
                results.append(p_reps.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    @classmethod
    def from_pretrained(cls, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        config = AutoConfig.from_pretrained(model_name)
        res = cls(model, tokenizer, config, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)
        res.model_name = model_name
        return res

    def __repr__(self):
        if hasattr(self, 'model_name'):
            return f'HgfBiEncoder({repr(self.model_name)})'
        return 'HgfBiEncoder()'


class _RepLLamaBiEncoder(_RepLLamaBiEncoderBase, metaclass=Variants):
    VARIANTS: dict = None
    def __init__(self, model_name=None, batch_size=32, text_field='text', verbose=False, device=None):
        self.model_name = model_name or next(iter(self.VARIANTS.values()))
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        super().__init__(model_name, tokenizer, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)

    def __repr__(self):
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_name in inv_variants:
            return f'{self.__class__.__name__}.{inv_variants[self.model_name]}()'
        return super().__repr__()


class RepLLama(_RepLLamaBiEncoder):
    """
    pip requirements:
     - tiktoken
     - peft
    
    .. automethod:: v1_7b()
    """
    VARIANTS = {
        'v1_7b': 'castorini/repllama-v1-7b-lora-passage',
    }