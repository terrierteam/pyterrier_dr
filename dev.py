from ir_measures import RR, Recall, nDCG

import os
import pyterrier_dr
import pyterrier as pt
import torch 
torch.set_float32_matmul_precision('high')


DEFAULT_INIT_BY_NAME: dict[str, str] = {
    "tct_colbert": "pyterrier_dr.TctColBert.hnp()",
    "tas_b": "pyterrier_dr.TasB()",
    "star": "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/star/')",
    "adore_star" : "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/adore_star/')",
    "repllama": "pyterrier_dr.RepLLama.v1_7b()",
    "e5": "pyterrier_dr.E5()",
    "dragon" : 'pyterrier_dr.Dragon()',
    "lion" : 'pyterrier_dr.LionLlamaDense()',
}


def instantiate_model(model_name: str):
    """
    Instantiate the model by evaluating the resolved expression in a restricted namespace.
    Only whitelisted globals are visible to eval().
    """
    allowed_globals = {
        "pyterrier_dr": pyterrier_dr,
    }
    expr = DEFAULT_INIT_BY_NAME[model_name]
    try:
        obj = eval(expr, allowed_globals, {})   # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate model with expression: {expr}\n"
            f"Error: {e}"
        ) from e
    return obj


if __name__ == "__main__":

    import sys
    model_name, old_indexdir, new_indexdir = sys.argv[1:4]

    old_index = pyterrier_dr.FlexIndex(old_indexdir)
    new_index = pyterrier_dr.jpq.JPQIndex(new_indexdir)
    assert len(old_index) == len(new_index)

    old_model = instantiate_model(model_name)
    if model_name == 'lion':
        new_model = old_model
    else:
        new_model = instantiate_model(model_name)
        new_model.model.from_pretrained(new_indexdir)

    p = [
        old_model >> old_index.retriever(), # type: ignore
        new_model >> new_index.retriever_pq()
    ]

    new_model.
    
    # reproducibilty settings
    import faiss
    thread_count = faiss.omp_get_max_threads()
    import os
    os.environ["MKL_CBWR"] = "COMPATIBLE"
    #faiss.omp_set_num_threads(1)

    import torch
    torch.manual_seed(0)

    save_dir = new_indexdir + "/runs"
    def _do_run(dataset, split, name):
        print(split, name)
        split_savedir = save_dir + "/" + name.replace(":", "_").replace("/", "_")
        os.makedirs(split_savedir, exist_ok=True)
        df = pt.Experiment(
            p,
            dataset.get_topics(split),
            dataset.get_qrels(split),
            eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10, "mrt"],
            names=["baseline", "JPQ pq"],
            save_dir = split_savedir,
            baseline=0
        )
        df.to_csv(split_savedir + "/metrics.csv")
        print(df)

    #for split in data.test_split.split(","):
    #    _do_run(pt.get_dataset(data.test_ds), split, split)
    #if data.eval_ds is not None:
    _do_run(pt.get_dataset("irds:msmarco-passage/dev/small"), None, "dev")

