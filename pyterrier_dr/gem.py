from typing import List, Optional, Any, Union
from collections.abc import Sequence, Iterable
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from .biencoder import BiEncoder


class GEM(BiEncoder):

    def __init__(
        self,
        model_name: str = "gem-search/GEM-4B",
        *,
        batch_size: int = 32,
        text_field: str = "text",
        verbose: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
        generation_config: Optional[dict[str, Any]] = None,
        prompt: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        """Generative Embedding Model.

        Args:
            max_length (int, optional): Max truncation length. No truncation if set to None. Defaults to None.
            generation_config (dict[str, Any], optional): Generation config.
                Expect args for transformers `generate`. If not provided, we will use
                `do_sample=False` and `max_new_tokens=1024`. Defaults to None.
            prompt (str, optional): Prompt for reasoning/generation.
                Expect a prompt with a `{text}` placeholder, e.g. "some meta-instruction ... {text}",
                where `{text}` will be replaced with a query. If not provided, a default reasoning prompt will be used. Defaults to None.

        See Also:
            https://huggingface.co/gem-search/GEM-4B

        """
        super().__init__(
            batch_size=batch_size,
            text_field=text_field,
            verbose=verbose,
        )

        try:
            from transformers.utils.import_utils import is_flash_attn_2_available, is_accelerate_available

            if is_flash_attn_2_available():
                model_kwargs.setdefault("attn_implementation", "flash_attention_2")

            if is_accelerate_available() and device is None:
                model_kwargs.setdefault("device_map", "auto")

        except ImportError:
            pass

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **model_kwargs).eval()

        if device and not model_kwargs.get("device_map"):
            self.model.to(device)

        self.max_length = max_length
        self.generation_config = generation_config or {}
        self.generation_config.setdefault("do_sample", False)
        self.generation_config.setdefault("max_new_tokens", 1024)
        self.prompt = prompt

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return self.model.encode_queries(
            self._convert_inputs(texts),
            batch_size=batch_size or self.batch_size,
            prompt=self.prompt,
            convert_to_tensor=False,
            show_progress_bar=self.verbose,
            max_length=self.max_length,
            **self.generation_config,
        )

    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        return self.model.encode_corpus(
            self._convert_inputs(texts),
            batch_size=batch_size or self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=self.verbose,
            max_length=self.max_length,
        )

    def _convert_inputs(self, texts: Any) -> Sequence[str]:
        if isinstance(texts, str):
            return [texts]
        elif isinstance(texts, Sequence):
            return texts
        elif isinstance(texts, tqdm):
            texts.disable = True
            self.verbose = True
            return list(texts)
        elif isinstance(texts, Iterable):
            return list(texts)
        else:
            raise TypeError(f"unsupported type: {type(texts)=}")
