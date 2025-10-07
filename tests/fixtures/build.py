import pyterrier_dr, torch, pyterrier as pt
tct = pyterrier_dr.TctColBert(device=torch.device("mps"))
index = pyterrier_dr.FlexIndex("./vaswani_tct.flex")
(tct >> index).index(pt.get_dataset("vaswani").get_corpus_iter())
print(len(index))