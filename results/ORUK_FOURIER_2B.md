# Paired SHALLOW Evaluation: Oruk-Fourier-2B vs. Qwen3-ASR-1.7B-hf

This report provides paired 4-axis SHALLOW hallucination metrics (**WER**, **LF**, **PF**, **ME**, **SE**) for **[`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b)** (fine-tuned from `Qwen/Qwen3-ASR-1.7B` by Oruk, [`oruk.ai`](https://oruk.ai)) alongside a matched native Hugging Face baseline (**[`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)**) across all **6 publicly accessible, exact-split English SHALLOW corpora** (`39,415` utterances total).

Machine-readable summary: [`results/oruk_fourier_2b_paired_summary.json`](./oruk_fourier_2b_paired_summary.json).

---

## Models, Checkpoints & Disclosures

| Model | Hugging Face Checkpoint | Revision / Integrity | Precision & Decoding | License & Scope |
|---|---|---|---|---|
| **Oruk-Fourier-2B** | [`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b) | `8c2a8871f1304aa2f634da4fb95e24438c4945b9` (`model.safetensors` `4,076,193,080` B, SHA-256 `e64765ff...`) | BF16, published `generation_config.json` (`num_beams=3`, `no_repeat_ngram_size=8`, `do_sample=False`, `max_new_tokens=512`, `32,253` `bad_words_ids`, `language="en"`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial), English-only |
| **Qwen3-ASR-1.7B-hf** (paired base) | [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | `main` (`2026-11-25`) | BF16, native HF greedy default (`num_beams=1`, `do_sample=False`, `max_new_tokens=512`, `language="en"`) | Apache-2.0, multilingual (evaluated on English) |

- **Disclosure**: Submitted by the Oruk team (`oruk.ai`). Because `oruk/oruk-fourier-2b` ships with 3-beam search and repetition/non-English constraints in its published `generation_config.json` whereas `Qwen/Qwen3-ASR-1.7B-hf` defaults to greedy decoding, both configurations are disclosed explicitly above.
- **Coverage disclosure (6 of 10 SHALLOW corpora)**: All 6 corpora below match the exact test utterance counts reported in the main [README](../README.md) (`39,415` utterances total). The remaining 4 SHALLOW corpora could not be reproduced publicly without restricted/unavailable sources and are intentionally omitted from the macro average (which is computed strictly over the **exact same 6 corpora** for all models):
  - **CHiME-6** (`11,027` utts): full corpus is not publicly redistributed on Hugging Face Hub.
  - **MyST Child** (`13,180` utts): paid LDC release (`LDC2021S05`).
  - **CV16-Accented** (`2,197` utts): `mozilla-foundation/common_voice_16_1` is no longer hosted on HF Hub (`404`) and the 2,197-utterance accented selection list is not published.
  - **CORAAL** (`5,000` utts): the author-sampled 5,000-utterance test ID list is not published (the `anonymous.4open.science/r/SHALLOW/` link has expired).

---

## Paired Results Across All 6 Public SHALLOW Corpora (`39,415` Utterances)

Lower is better across all five metrics (`WER`, `LF`, `PF`, `ME`, `SE`). Best value between the two paired models is shown in **bold**; values marked with `*` set a new best across all 14 models (the 12 published SHALLOW systems plus these two).

| Dataset (# Utts) | Metric | Oruk-Fourier-2B (3-beam) | Qwen3-ASR-1.7B-hf (paired greedy) | Delta (Fourier − Base) | Best Published SHALLOW System (README) |
|---|---|---:|---:|---:|---|
| **LibriSpeech (other)** (`2,939`) | **WER** | **2.59*** | 3.37 | `-0.78` | Parakeet (`2.62`) |
| | **LF** | **0.93*** | 1.24 | `-0.31` (`-25.0%`) | Parakeet (`1.00`) |
| | **PF** | **3.98** | 4.83 | `-0.85` (`-17.6%`) | Parakeet (`3.54`) |
| | **ME** | **3.88*** | 5.07 | `-1.19` (`-23.5%`) | Parakeet (`4.35`) |
| | **SE** | **2.94*** | 3.88 | `-0.94` (`-24.2%`) | Parakeet (`2.95`) |
| **GigaSpeech** (`25,619`) | **WER** | **11.16*** | 12.57 | `-1.41` (`-11.2%`) | Parakeet (`11.37`) |
| | **LF** | **9.16** | 15.10 | `-5.94` (`-39.3%`) | Qwen2Audio (`5.26`) / Canary (`5.31`) |
| | **PF** | **21.26** | 27.23 | `-5.97` (`-21.9%`) | Canary (`16.12`) |
| | **ME** | **11.88** | 15.36 | `-3.48` (`-22.7%`) | Qwen2Audio (`10.09`) |
| | **SE** | **18.91** | 27.02 | `-8.11` (`-30.0%`) | Qwen2Audio (`12.32`) |
| **TEDLIUM** (`1,469`)† | **WER** | **10.80** | 10.81 | `-0.01` | Kimi (`8.12`) |
| | **LF** | **5.34*** | 5.61 | `-0.27` | Parakeet (`5.37`) |
| | **PF** | 25.69 | **23.47** | `+2.22` | Parakeet (`22.42`) |
| | **ME** | 11.02 | **10.96*** | `+0.06` | Whisper-v3 (`11.45`) |
| | **SE** | 21.97 | **19.94*** | `+2.03` | Whisper-v2 (`20.73`) |
| **SpeechOcean** (`2,500`) | **WER** | **11.02*** | 11.14 | `-0.12` | Phi4 (`12.88`) |
| | **LF** | **3.64*** | 3.67 | `-0.03` | Phi4 (`4.27`) |
| | **PF** | **8.16*** | **8.16*** | `0.00` | Phi4 (`9.32`) |
| | **ME** | 9.84 | **9.68*** | `+0.16` | Phi4 (`10.92`) |
| | **SE** | 12.51 | **12.37*** | `+0.14` | Phi4 (`13.76`) |
| **GLOBE-v2** (`5,046`) | **WER** | 1.66 | **1.56** | `+0.10` | Parakeet (`1.17`) |
| | **LF** | 0.58 | **0.54** | `+0.04` | Parakeet (`0.46`) |
| | **PF** | 2.01 | **1.86** | `+0.15` | Parakeet (`1.24`) |
| | **ME** | 1.97 | **1.90** | `+0.07` | Parakeet (`1.50`) |
| | **SE** | 1.79 | **1.69** | `+0.10` | Parakeet (`1.18`) |
| **VoxPopuli** (`1,842`) | **WER** | 7.33 | **6.28** | `+1.05` | Parakeet (`5.42`) |
| | **LF** | 2.42 | **2.04** | `+0.38` | Parakeet (`1.75`) |
| | **PF** | 15.89 | **13.02** | `+2.87` | Parakeet (`11.59`) |
| | **ME** | 7.46 | **6.48** | `+0.98` | Parakeet (`5.80`) |
| | **SE** | 5.41 | **4.92** | `+0.49` | Parakeet (`4.11`) |

### Matched 6-Corpus Macro Average (`39,415` Utterances)

Computed strictly across the exact same 6 corpora (`LibriSpeech-other`, `TEDLIUM`, `GigaSpeech`, `GLOBE-v2`, `SpeechOcean`, `VoxPopuli`) for every system:

| Model | 6-Corpus WER↓ | 6-Corpus LF↓ | 6-Corpus PF↓ | 6-Corpus ME↓ | 6-Corpus SE↓ |
|---|---:|---:|---:|---:|---:|
| **Oruk-Fourier-2B** (2B, 3-beam) | **7.43** | 3.68 | 12.83 | **7.67** | 10.59 |
| **Qwen3-ASR-1.7B-hf** (1.7B, paired greedy) | 7.62 | 4.70 | 13.09 | 8.24 | 11.64 |
| Phi4-MM-Instruct (5.6B, README) | 7.80 | 4.58 | 14.50 | 8.96 | 11.43 |
| Qwen2.5-Omni (10.7B, README) | 7.92 | 3.55 | 12.52 | 8.40 | **10.09** |
| Kimi-Audio (9.7B, README) | 8.84 | 4.93 | 16.23 | 9.76 | 12.48 |
| Qwen2Audio (8.4B, README) | 8.85 | **3.48** | 13.39 | 8.74 | 10.96 |
| Parakeet (1.1B, README) | 9.11 | 3.85 | **12.07** | 8.46 | 11.06 |
| Whisper-Large-v3 (1.5B, README) | 10.18 | 5.42 | 15.63 | 9.71 | 13.18 |
| Canary (1B, README) | 10.46 | 4.10 | 13.00 | 9.30 | 12.35 |
| Granite-Speech (8.6B, README) | 10.65 | 4.21 | 12.68 | 8.34 | 11.48 |

### Key Takeaways & Honest Trade-Offs

1. **5/5 Sweep vs. Paired Base Checkpoint**: Across the 6-corpus macro average, `Oruk-Fourier-2B` improves over its base checkpoint `Qwen/Qwen3-ASR-1.7B-hf` on **all five metrics**: WER (`7.43` vs. `7.62`), **LF (`-21.7%`: `3.68` vs. `4.70`)**, PF (`12.83` vs. `13.09`), **ME (`-6.9%`: `7.67` vs. `8.24`, new `#1` overall)**, and **SE (`-9.0%`: `10.59` vs. `11.64`)**.
2. **Non-Speech Hallucination Suppression on GigaSpeech (`5,762` empty-GT segments)**: After `utils.clean_transcript_gigaspeech` strips `<SIL>`, `<MUSIC>`, `<NOISE>`, and `<OTHER>` tags, `5,762` of `25,619` GigaSpeech test rows have an empty normalized reference (`""`). On these non-speech segments, `Oruk-Fourier-2B` cleanly emits empty output on **`2,202 / 5,762` (`38.2%`)** vs. **`261 / 5,762` (`4.5%`)** for `Qwen/Qwen3-ASR-1.7B-hf` (an **8.4×** improvement in non-speech abstinence, emitting `2,950` fewer hallucinated words: `14,963` vs. `17,913`). This drives the large paired reductions on GigaSpeech: **`-39.3%` LF (`9.16` vs. `15.10`)** and **`-30.0%` SE (`18.91` vs. `27.02`)** alongside a new `#1` dataset WER (`11.16%`).
3. **†TEDLIUM `ignore_time_segment_in_scoring` Diagnostic (`314 / 1,469` rows)**: The standard `sanchit-gandhi/tedlium-data` `test` split (`1,469` rows) includes `314` `inter_segment_gap` clips whose literal text label is `"ignore_time_segment_in_scoring"`. On those `314` gap clips, `Oruk-Fourier-2B` emits empty `""` on **`126 / 314` (`40.1%`)** vs. **`0 / 314` (`0.0%`)** for `Qwen/Qwen3-ASR-1.7B-hf`. Because `semantic.py` assigns `0.0` similarity (`SE = 100`) whenever the hypothesis is empty against a non-empty dummy label, silence abstinence lowers LF (`5.34` vs. `5.61`) while inflating PF/SE on the unfiltered `1,469` split. Excluding the `314` dummy rows (`n = 1,155` scorable speech segments) gives:
   - **Oruk-Fourier-2B (`n=1,155`)**: `WER 2.30` · `LF 0.80` · `PF 5.97` · `ME 3.11` · `SE 2.28`
   - **Qwen3-ASR-1.7B-hf (`n=1,155`)**: `WER 2.27` · `LF 0.79` · `PF 5.99` · `ME 3.06` · `SE 2.34`
4. **Honest Domain Trade-Offs**: On **VoxPopuli** (European Parliament speeches, `n=1,842`), `Qwen/Qwen3-ASR-1.7B-hf` outperforms `Oruk-Fourier-2B` (`6.28%` vs. `7.33%` WER; LF `2.04` vs. `2.42`, SE `4.92` vs. `5.41`), and slightly edges it on **GLOBE-v2** (`1.56%` vs. `1.66%` WER).

---

## Scoring Protocol & Parity Verification

All scores use SHALLOW's unmodified preprocessing (`src/utils.py::load_gt_pred_transcriptions`) and exact aggregation formulas (`src/shallow.py`) under the exact package pins from `requirements.txt` (`transformers==4.51.3`, `benepar==0.2.0`, `spacy==3.8.5`, `sentence_transformers==4.1.0`, `jiwer==3.1.0`, `evaluate==0.4.3`, `jellyfish==1.2.0`, `language_tool_python==2.9.3`, `whisper_normalizer==0.1.11`).

To evaluate `39,415 × 2 = 78,830` utterance pairs without hitting `LanguageToolPublicAPI`'s public HTTP rate limit (`RateLimitError`) or reloading `roberta-large` on every utterance:
- Local `language_tool_python.LanguageTool('en-US')` (OpenJDK 17) replaced `LanguageToolPublicAPI('en-US')`.
- `evaluate.load("bertscore")` was initialized once per worker process and unique `bert-base-uncased` n-gram windows were embedded with attention-masked mean pooling.
- **Verified parity gate**: Compared against unmodified `ShallowBenchmark` across 25 stratified non-identical pairs spanning all 7 categories of `synthetic_eval_dataset/synth_eval_dataset.csv`: **`overall_max_abs_diff = 1.8e-7`** (`0.0` exact on `wer_score`, `LF`, `PF`, `ME` structural divergence & grammatical errors, and `SE` global cosine/coherence; `<= 1.8e-7` FP32 epsilon on `SE` local windows).
