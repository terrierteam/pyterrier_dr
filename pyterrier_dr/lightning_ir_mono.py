import numpy as np
import more_itertools
import torch
import pyterrier as pt
import pyterrier_alpha as pta

LIGHTNING_IR_AVAILIBLE = False
try:
    import lightning_ir as L
    LIGHTNING_IR_AVAILIBLE = True
except ImportError:
    pass


class LightningIRMonoScorer(pt.Transformer):
    def __init__(self,
                 model_name='webis/monoelectra-base',
                 batch_size=16,
                 text_field='text',
                 verbose=True,
                 device=None,
                 query_length=32,
                 doc_length=512):
        if not LIGHTNING_IR_AVAILIBLE:
            raise ImportError("lightning_ir is required for LightningIRMonoScorer. Please install it via 'pip install lightning-ir'")
        self.model_name = model_name
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.tokenizer = L.cross_encoder.cross_encoder_tokenizer.CrossEncoderTokenizer.from_pretrained(model_name,
                                                                                                       query_length=query_length,
                                                                                                       doc_length=doc_length)
        self.model = L.models.cross_encoders.mono.MonoModel.from_pretrained(model_name).eval().to(self.device)

    def transform(self, inp):
        pta.validate.columns(inp, includes=['query', self.text_field])
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose and len(inp):
            it = pt.tqdm(it, total=len(inp), unit='record', desc=f'{self.model_name} scoring')
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer.tokenize(queries=queries,
                                               docs=texts,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt'
                                               )["encoding"].to(self.device)
                with torch.inference_mode():
                    output = self.model.forward(inps).scores
                scores.append(output.cpu().detach().numpy())
        if not scores:
            scores = np.empty(shape=(0, 0))
        else:
            scores = np.concatenate(scores, axis=0)
        res = inp.assign(score=scores)
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res
