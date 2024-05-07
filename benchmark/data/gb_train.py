from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoCodingVsIntergenomicSeqs, DemoHumanOrWorm, DrosophilaEnhancersStark, DemoMouseEnhancers, HumanEnhancersCohn, HumanEnhancersEnsembl, HumanNontataPromoters, HumanOcrEnsembl

train_datasets = {
    "demo_coding_vs_intergenomic_seqs": DemoCodingVsIntergenomicSeqs(split="train", version=0),
    "demo_human_or_worm": DemoHumanOrWorm(split="train", version=0),
    "drosophila_enhancers_stark": DrosophilaEnhancersStark(split="train", version=0),
    "dummy_mouse_enhancers_ensembl": DemoMouseEnhancers(split="train", version=0),
    "human_enhancers_cohn": HumanEnhancersCohn(split="train", version=0),
    "human_enhancers_ensembl": HumanEnhancersEnsembl(split="train", version=0),
    "human_ocr_ensembl": HumanOcrEnsembl(split="train", version=0),
    "human_nontata_promoters": HumanNontataPromoters(split="train", version=0),
}
