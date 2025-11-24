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
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map="auto", dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval().compile()
    return model

def replace_with_xformers_attention():
    import torch
    import xformers.ops as xops
    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

    def llama_xformers_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """
        A drop-in replacement for LlamaAttention.forward that uses xformers memory-efficient attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_embeddings: tuple(cos, sin) for RoPE
            attention_mask: causal / padding mask if needed
            past_key_values: Cache object (new HF API)
            cache_position: positions for caching (new API)
            **kwargs: other unused HF args
        Returns:
            (attn_output, new_past_key_values)
        """
        bsz, seq_len, hidden_size = hidden_states.size()

        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape into multi-head
        # Reshape into heads

        # Embedding / output dimension
        hidden_size = getattr(self, "hidden_size", getattr(self, "embed_dim", None))

        # Head dimension
        head_dim = self.head_dim

        # Total query heads (derived)
        num_heads = self.q_proj.weight.shape[0] // head_dim

        # KV heads (derived using grouping)
        num_kv_heads = num_heads // self.num_key_value_groups

        q = q.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)



        # q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # kv_seq_len = seq_len
        # # For k, v, use number of key/value heads
        # k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle cache (past_key_values)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Use xformers memory-efficient attention
        # Note: xformers expects shape [B, M, H, K] for q, k, v
        # Our q, k, v are [B, heads, seq_len, head_dim] => transpose heads & seq
        # Actually, xops.memory_efficient_attention expects [B, M, H, K], so we need to rearrange.
        # Here’s a typical approach:

        # transpose back: (batch, seq_len, heads, head_dim)
        q2 = q.transpose(1, 2).contiguous()
        k2 = k.transpose(1, 2).contiguous()
        v2 = v.transpose(1, 2).contiguous()

        # Compute attention via xformers
        # Use a causal mask if required
        attn_bias = None
        if attention_mask is not None:
            # If your mask is lower triangular, you can use:
            attn_bias = xops.LowerTriangularMask()

        out = xops.memory_efficient_attention(q2, k2, v2, attn_bias=attn_bias)
        # out: [batch, seq_len, hidden_size] (since heads * head_dim flattened)

        # Project back
        attn_output = self.o_proj(out)
        attn_output = attn_output.reshape(bsz, seq_len, hidden_size)

        # Prepare new cache
        new_past = past_key_values if past_key_values is not None else None

        # Return (output, key_value_cache)
        return attn_output, new_past

    # Monkey-patch it in
    LlamaAttention.forward = llama_xformers_forward

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
            q_hidden = self.model(**inps).last_hidden_state
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
                p_hidden = self.model(**inps).last_hidden_state
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
