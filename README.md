# SEISMOS: A Statistical Signal Detection Framework for Semantic Chunking

**Ishtiak Mahmud Saad**¹, **Mominul Islam**², **Shahriyar Zaman Ridoy**³˒⁵, **Md Manjurul Ahsan**⁴˒⁵, **Azmine Toushik Wasi**⁵

¹ Rajshahi University of Engineering & Technology, ² Tulane University, ³ North South University, ⁴ University of Oklahoma, ⁵ Computational Intelligence and Operations Laboratory (CIOL)

* **Project website:** https://seismos-sc.github.io/
* **GitHub repository:** https://github.com/SEISMOS-sc/SEISMOS-sc.github.io/
* **Conference:** NeurIPS 2026


**SEISMOS** is a statistical framework for semantic document chunking that identifies topic boundaries by detecting statistically significant valleys in consecutive sentence similarities. It uses a normalized discrete Laplacian to distinguish meaningful semantic transitions from shallow fluctuations in the similarity signal.

SEISMOS uses a single decision threshold, requires no large language model calls for boundary detection, and runs in linear time with respect to the number of sentences.

## Overview

Document chunking determines how information is divided before it is indexed, retrieved, and passed to a language model in a retrieval augmented generation (RAG) system. Fixed length chunking can split related information across boundaries, while semantic chunkers can introduce unnecessary fragmentation or incur substantial preprocessing costs.

SEISMOS treats consecutive sentence embedding similarities as a document level statistical signal. Rather than splitting at every local drop in similarity, it measures the curvature of the signal and normalizes the resulting statistic by its variability.

The resulting detector identifies sharp semantic transitions using one interpretable threshold, without requiring document specific threshold tuning.

### Key Results

According to our reported experiments:

| Finding | Result |
|--|--|
| BEIR retrieval benchmarks | Best chunking performance on all 5 evaluated datasets |
| RAG evidence retrieval | Chunk nDCG@5 of 0.190, compared with 0.106 to 0.113 for the evaluated baselines |
| Preprocessing efficiency | 15 to 23 times faster than production semantic chunkers in the reported comparison |
| Decision threshold | One fixed threshold, k = 2.5 |
| Long document evaluation | Competitive retrieval performance on documents containing tens of thousands of tokens |
| Multilingual evaluation | Competitive performance across Japanese, German, and Arabic without language specific tuning |

These results reflect the experiments reported on the project website. Performance may vary with the embedding model, corpus, retrieval configuration, and downstream generation model.

## Experimental Evaluation

We evaluate SEISMOS on retrieval benchmarks, long document tasks, multilingual datasets, and an end to end RAG setting.

### BEIR Retrieval

The reported evaluation covers five BEIR datasets:

* ArguAna
* FiQA
* NFCorpus
* SciFact
* TREC COVID

The experiments compare SEISMOS with fixed length chunking, recursive chunking, and production semantic chunkers. The same BGE base embedding model and indexing configuration are used for the reported comparison.

TREC COVID is held out from threshold tuning. SEISMOS retains the same decision threshold across the evaluated datasets.

### Long Documents and Multilingual Retrieval

The evaluation also considers long documents and multilingual retrieval.

Reported results show competitive performance on documents containing tens of thousands of tokens and across Japanese, German, and Arabic. No language specific threshold tuning is required.

### End to End RAG

The RAG evaluation examines evidence retrieval, preprocessing cost, and answer quality.

Reported findings include:

* **Evidence retrieval:** Chunk nDCG@5 of 0.190 on NarrativeQA, compared with 0.106 for LangChain semantic chunking, 0.108 for LlamaIndex semantic chunking, and 0.113 for the evaluated fixed length and recursive baselines.
* **Preprocessing speed:** Approximately 0.133 seconds per document of roughly 27,000 tokens in the reported setup, compared with 2.036 seconds for LangChain semantic chunking and 3.105 seconds for LlamaIndex semantic chunking.
* **Answer faithfulness:** An HHEM faithfulness score of 0.687, the highest among the five evaluated chunkers.
* **Human rated correctness:** A score of 0.952 in the reported evaluation of 21 items assessed by three annotators.

The results suggest that boundary detection can affect both the quality of retrieved evidence and the cost of preparing documents for retrieval.

For complete tables, ablations, and visualizations, see the [project website](https://seismos-sc.github.io/).

## Citation

If you use SEISMOS in your research, please cite the paper.

```bibtex
@inproceedings{saad2026seismos,
  title     = {{SEISMOS}: A Statistical Signal Detection Framework
               for Semantic Chunking},
  author    = {Saad, Ishtiak Mahmud and Islam, Mominul and
               Ridoy, Shahriyar Zaman and Ahsan, Md Manjurul and
               Wasi, Azmine Toushik},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
