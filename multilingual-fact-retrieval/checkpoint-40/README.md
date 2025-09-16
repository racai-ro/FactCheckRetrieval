---
language:
- en
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:25491
- loss:CachedMultipleNegativesRankingLoss
base_model: BAAI/bge-multilingual-gemma2
widget:
- source_sentence: '<instruct>Given a social media post as a query, retrieve fact
    checks that verify or debunk the post.

    <query>Hard to believe we are breathing the same oxygen...more than 70 million
    people.... #MillionMoronMarch STATES 来降降速後 -ON WE 3 ***** ***** * + ★ IN *****
    *****Tru 20 RUMA 202 MAKE LIBER CRY AGE CHCKEVL4X4 TRUMP- Great 2020 God Bless
    Americk 2 Amend ment! ADANGERTRUMP TRUMP TRU 202 a One of the worst parts of the
    Trump era is the inescapable realization of just how many truly awful people walk
    among us. [USER]'
  sentences:
  - False warnings of Sri Lanka knifepoint muggings show video from Indonesia. Video
    shows an ATM robbery in suburban Sri Lankan
  - Tweets falsely claim photo of Nazi flag is from pro-Trump March. Nazi flag was
    sold at a stall at the march for Trump on November 14, 2020
  - El candidato peruano Lescano se casó en Chile y tiene número de cédula, pero no
    la nacionalidad. El candidato presidencial peruano Yonhy Lescano tiene nacionalidad
    chilena
- source_sentence: '<instruct>Given a social media post as a query, retrieve fact
    checks that verify or debunk the post.

    <query>Curva de votos de Trump y Biden. Veis algo raro en la curva azul? El claro
    indicio del fraude electoral que están haciendo los demócratas con los votos por
    correo sobretodo. Esto no os recuerda a las pasadas elecciones en España? Por
    ese deben intervenir organizaciones internacionales y no empresas compradas por
    los grandes magnates #EEUU #election2020 Votes 3000000 2000000 1000000 0 The state
    of the race in Michigan Total presidential votes for each party so far, with 86
    percent of Michigan''s expected vote counted as of 7:17 a.m. on Nov.4 MI Nov 03
    18:00 Nov 04 00:00 Time rump Votes Biden votes Nov 04 06:00 Nov 04 12:'
  sentences:
  - No, estas imágenes no demuestran un fraude electoral en Michigan o en Wisconsin.
    Gráficos que indican que hubo fraude electoral en Wisconsin y Michigan
  - Post falsely claims there were no US flu deaths during COVID-19 crisis. No Americans
    have died from the flu in 2020
  - No, this image has been manipulated to include an offensive slogan. Periodic table
    assembled to say “F**k Boris” in front of the UK PM while he attends a press conference
- source_sentence: '<instruct>Given a social media post as a query, retrieve fact
    checks that verify or debunk the post.

    <query>“So let me get this straight about this virus It’s “new” yet it was lab
    created and patented in 2015 (in development since 03’) The patent expired (today)
    on the day the first case is announced in the US The patent also tells us the
    CDC helped make this! “This invention was made by the Centers for Disease Control
    and Prevention, an agency of the United States Government. Therefore, the U.S.
    Government has certain rights in this invention.” And now magically a vaccine
    is in the works for it already? Yet the patent in 2015 already references a vaccine
    for it. But what do I know I’m just a conspiracy theorist.” - Chris Kirckof [URL]
    CORONAVIRUS BREAKING NEWS FIRST U.S. CASE OF DEADLY MYSTERY VIRUS CONFIRMED I
    NIGHT NEW'
  sentences:
  - 'Fake News: Patent For Coronavirus That Expired Is NOT For New Strain Killing
    People In China. Patent For Coronavirus That Expired Is For New Strain Killing
    People In China'
  - Este meteorito no fue registrado en México en 2020, sino en Holanda en 2009. Meteorito
    visto en el norte de México el 6 de octubre de 2020
  - Doctored advert does not show 'Delhi chief minister asking citizens to donate
    coal' following shortage. Delhi chief minister asked people to donate coal
- source_sentence: '<instruct>Given a social media post as a query, retrieve fact
    checks that verify or debunk the post.

    <query>If you were sick around Thanksgiving and Christmas, and you drunk boxes
    of theraflu, herbal tea, used essential oils, took 2 bottles of pills, had a random
    ass cough that wouldn''t leave, bags of cough drops couldn''t stop it, high fever
    and nothing could break your sickness, and the doctor told you that you tested
    negative for strep and flu, and said you "MAY" have an "Upper Respiratory Infection"
    because it was hard to breath and that you would have to let it run its course,
    and it took a minimum of 10+ days.. You my friend have already had the Coronavirus
    🤷🏼 ♂️ '
  sentences:
  - Non, cette photo ne montre pas "l'avenue du Tourisme" à Kinshasa "à l'horizon
    2023". Cette photo montre l'autoroute du Tourisme à Kinshasa "à l'horizon 2023"
  - No evidence novel coronavirus was spreading in US as early as November 2019. COVID-19
    was in US in November 2019
  - Non, une carte météo de la France n'a pas été rougie pour "manipuler" l'opinion.
    Ce visuel montre la manipulation du public via les cartes météo de France2
- source_sentence: '<instruct>Given a social media post as a query, retrieve fact
    checks that verify or debunk the post.

    <query>یہ ہے آزادی مارچ سرینگر میں یاسین ملک کو رہا کرو رہا کرو اس کو کہتے ہیں
    آزادی مارچ نام نہاد والو '
  sentences:
  - Un oignon coupé conservé jusqu'au lendemain n'est pas plus "toxique" ou "dangereux"
    que d'autres légumes. Un oignon coupé devient toxique et dangereux au bout de
    quelques heures
  - El vídeo de una manifestación de mujeres contra los talibanes es de Irán. Un vídeo
    muestra una numerosa manifestación de mujeres contra los talibanes en Afganistán
  - Photo from 2018 US rally misrepresented as protest in Indian-controlled Kashmir
    in May 2022. Photo of protest in Srinagar over sentencing of Kashmiri independence
    leader by Indian court
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: Multilingual Fact Retrieval Gemma
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: gemma
      type: gemma
    metrics:
    - type: cosine_accuracy@1
      value: 0.9227272727272727
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.9590909090909091
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.9681818181818181
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.9818181818181818
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.9227272727272727
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.36212121212121207
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.22181818181818183
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.11272727272727275
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.8636363636363636
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.9534090909090909
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.9670454545454545
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.9818181818181818
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.9523427640798445
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.9434199134199135
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.9429893785631489
      name: Cosine Map@100
---

# Multilingual Fact Retrieval Gemma

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2). It maps sentences & paragraphs to a 3584-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2) <!-- at revision 992e13d8984fde2c31ef8a3cb2c038aeec513b8a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 3584 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
- **Language:** en
- **License:** apache-2.0

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: Gemma2Model 
  (1): Pooling({'word_embedding_dimension': 3584, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': True, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '<instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.\n<query>یہ ہے آزادی مارچ سرینگر میں یاسین ملک کو رہا کرو رہا کرو اس کو کہتے ہیں آزادی مارچ نام نہاد والو ',
    'Photo from 2018 US rally misrepresented as protest in Indian-controlled Kashmir in May 2022. Photo of protest in Srinagar over sentencing of Kashmiri independence leader by Indian court',
    'Un oignon coupé conservé jusqu\'au lendemain n\'est pas plus "toxique" ou "dangereux" que d\'autres légumes. Un oignon coupé devient toxique et dangereux au bout de quelques heures',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 3584]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `gemma`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.9227     |
| cosine_accuracy@3   | 0.9591     |
| cosine_accuracy@5   | 0.9682     |
| cosine_accuracy@10  | 0.9818     |
| cosine_precision@1  | 0.9227     |
| cosine_precision@3  | 0.3621     |
| cosine_precision@5  | 0.2218     |
| cosine_precision@10 | 0.1127     |
| cosine_recall@1     | 0.8636     |
| cosine_recall@3     | 0.9534     |
| cosine_recall@5     | 0.967      |
| cosine_recall@10    | 0.9818     |
| **cosine_ndcg@10**  | **0.9523** |
| cosine_mrr@10       | 0.9434     |
| cosine_map@100      | 0.943      |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 25,491 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                               | positive                                                                            |
  |:--------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                               | string                                                                              |
  | details | <ul><li>min: 28 tokens</li><li>mean: 150.95 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 40.45 tokens</li><li>max: 204 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | positive                                                                                                                                                                                                                                          |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code><instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.<br><query>O Exército brasileiro prendendo caminhões com madeiras ilegais no Pará. Pasmem vocês, mas o dono de toda essa madeira é fundador da principal ONG de preservação ambiental da Amazônia, cabeça do MST-PA e esquerdista radical com vários títulos recebidos na Europa, como cidadão defensor e protetor da Amazônia. *RESPOSTA DO VICE-MOURÃO* PARA BOLSONARO: *”MISSÃO DADA MISSÃO CUMPRIDA”* 👮🏻 ♂️🤝 A Europa critica duramente o desmatamento, mas é a maior compradora da madeira oriunda da Amazônia, dá pra entender uma MERDA dessas? Entenderam agora o interesse surreal de Macron com a "preservação" da Amazônia? HIPÓCRITAS!! </code> | <code>Foto de caminhões com toras de madeira não é de apreensão do Exército na Amazônia. Foto mostra caminhões com madeira ilegal apreendidos pelo Exército</code>                                                                                |
  | <code><instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.<br><query>O Exército brasileiro prendendo caminhões com madeiras ilegais no Pará. Pasmem vocês, mas o dono de toda essa madeira é fundador da principal ONG de preservação ambiental da Amazônia, cabeça do MST-PA e esquerdista radical com vários títulos recebidos na Europa, como cidadão defensor e protetor da Amazônia. *RESPOSTA DO VICE-MOURÃO* PARA BOLSONARO: *”MISSÃO DADA MISSÃO CUMPRIDA”* 👮🏻 ♂️🤝 A Europa critica duramente o desmatamento, mas é a maior compradora da madeira oriunda da Amazônia, dá pra entender uma MERDA dessas? Entenderam agora o interesse surreal de Macron com a "preservação" da Amazônia? HIPÓCRITAS!! </code> | <code>Estas toras não foram apreendidas em uma ação contra desmatamento ilegal, mas doadas para festa de igreja. Madeira ilegal apreendida pelo Exército que era do fundador da principal ONG de preservação da Amazônia e “cabeça” do MST</code> |
  | <code><instruct>Given a social media post as a query, retrieve fact checks that verify or debunk the post.<br><query>Voilà la face cachée de la France à Kidal (Mali) où les militaires français sont en train ravager l'or. C'est ça la guerre contre les djihadistes au Mali ? Quelle honte ! </code>                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>Non, ces photos ne montrent pas l’armée française en train de piller de l’or dans le nord du Mali. Des photos montrant le pillage de l'or au Mali par l'armée française</code>                                                              |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 1024
- `per_device_eval_batch_size`: 1024
- `learning_rate`: 0.0002
- `num_train_epochs`: 20
- `lr_scheduler_type`: cosine
- `warmup_ratio`: 0.1
- `bf16`: True
- `load_best_model_at_end`: True
- `optim`: adamw_torch_fused
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 1024
- `per_device_eval_batch_size`: 1024
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 0.0002
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 20
- `max_steps`: -1
- `lr_scheduler_type`: cosine
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: True
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch | Step | Training Loss | gemma_cosine_ndcg@10 |
|:-----:|:----:|:-------------:|:--------------------:|
| 0.375 | 3    | 1.4351        | -                    |
| 0.75  | 6    | 0.6192        | -                    |
| 1.0   | 8    | -             | 0.9129               |
| 1.125 | 9    | 0.342         | -                    |
| 1.5   | 12   | 0.2698        | -                    |
| 1.875 | 15   | 0.2292        | -                    |
| 2.0   | 16   | -             | 0.9328               |
| 2.25  | 18   | 0.1619        | -                    |
| 2.625 | 21   | 0.1211        | -                    |
| 3.0   | 24   | 0.1057        | 0.9277               |
| 3.375 | 27   | 0.0692        | -                    |
| 3.75  | 30   | 0.0587        | -                    |
| 4.0   | 32   | -             | 0.9504               |
| 4.125 | 33   | 0.062         | -                    |
| 4.5   | 36   | 0.0529        | -                    |
| 4.875 | 39   | 0.0459        | -                    |
| 5.0   | 40   | -             | 0.9523               |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 3.4.0.dev0
- Transformers: 4.48.1
- PyTorch: 2.6.0a0+df5bbc09d1.nv24.12
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### CachedMultipleNegativesRankingLoss
```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->