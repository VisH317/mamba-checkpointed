from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoCodingVsIntergenomicSeqs, DemoHumanOrWorm, DrosophilaEnhancersStark, DemoMouseEnhancers, HumanEnhancersCohn, HumanEnhancersEnsembl, HumanNontataPromoters, HumanOcrEnsembl

test_data = {
    "demo_coding_vs_intergenomic_seqs": DemoCodingVsIntergenomicSeqs(split="test", version=0),
    "demo_human_or_worm": DemoHumanOrWorm(split="test", version=0),
    "drosophila_enhancers_stark": DrosophilaEnhancersStark(split="test", version=0),
    "dummy_mouse_enhancers_ensembl": DemoMouseEnhancers(split="test", version=0),
    "human_enhancers_cohn": HumanEnhancersCohn(split="test", version=0),
    "human_enhancers_ensembl": HumanEnhancersEnsembl(split="test", version=0),
    "human_ocr_ensembl": HumanOcrEnsembl(split="test", version=0),
    "human_nontata_promoters": HumanNontataPromoters(split="test", version=0),
}
