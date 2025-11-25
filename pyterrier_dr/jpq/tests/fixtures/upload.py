import pyterrier as pt

artifact = pt.Artifact.load("./msmarco-passage.tct-hnp.flex")
print(f"Loaded artifact {artifact}")
artifact.to_hf('ntonellotto/msmarco-passage.tct-hnp.flex')
