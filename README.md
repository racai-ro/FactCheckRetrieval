# FactCheckRetrieval

This repository contains the implementation of the paper **"RACAI at SemEval-2025 Task 7: Efficient adaptation of Large Language Models for Multilingual and Crosslingual Fact-Checked Claim Retrieval"**.

## Paper Abstract

This paper presents our approach to SemEval 2025 Shared Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval. We investigate how large language models (LLMs) designed for general-purpose retrieval can be adapted for fact-checked claim retrieval across multiple languages. This includes cases where the original claim and the fact-checked claim are in different languages. The experiments involve fine-tuning with a contrastive objective, resulting in notable gains in accuracy over the baseline. We evaluate cost-effective techniques such as LoRA, QLoRA and Prompt Tuning. Additionally, we demonstrate the benefits of Matryoshka embeddings in minimizing the memory footprint of stored embeddings, reducing the system requirements for a fact-checking engine. The final solution, using a LoRA adapter, achieved 4th place for the monolingual track (0.937 S@10) and 3rd place for crosslingual (0.825 S@10).

## Results

Our approach achieved competitive results in SemEval-2025 Task 7:
- **Monolingual Track**: 4th place with 0.937 S@10
- **Crosslingual Track**: 3rd place with 0.825 S@10

## Key Features

- Efficient adaptation of large language models for fact-checked claim retrieval using a constrastive learning objective
- Support for multilingual and crosslingual scenarios
- Implementation of cost-effective fine-tuning techniques (LoRA, QLoRA, Prompt Tuning)
- Matryoshka embeddings for reduced memory footprint

## Code Availability

!!!!**Code will be added soon.**

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{semeval2025task7_racai,
	title={RACAI at SemEval-2025 Task 7: Efficient adaptation of Large Language Models for Multilingual and Crosslingual Fact-Checked Claim Retrieval},
	author={Radu Chivereanu, Dan Tufis},
	booktitle = {Proceedings of the 19th International Workshop on Semantic Evaluation},
	series = {SemEval 2025},
	year = {2025},
	address = {Vienna, Austria},
	month = {July},
}
```

## Contact

For questions or issues, please contact the authors or open an issue in this repository.
