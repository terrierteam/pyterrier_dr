import pyterrier_dr, torch, pyterrier as pt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

tct = pyterrier_dr.TctColBert(device=device)

#index = pyterrier_dr.FlexIndex("./vaswani_tct.flex")
#(tct >> index).index(pt.get_dataset("vaswani").get_corpus_iter())

index = pyterrier_dr.FlexIndex("./msmarco-passage.tct-hnp.flex")
(tct >> index).index(pt.get_dataset("msmarco_passage").get_corpus_iter())

print(len(index))