# Generative AI for End-to-End Limit Order Book Modelling
## A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network

This repository provides the implementation for the
paper: Generative AI for End-to-End Limit Order Book Modelling. The preprint is available [here](https://arxiv.org/abs/xxxx.xxxxx).

The repository is a fork of [the original S5 repository](https://github.com/lindermanlab/S5).

Developing a generative model of realistic order flow in financial markets is a challenging open problem, with numerous applications for market participants. Addressing this, we propose the first end-to-end autoregressive generative model that generates tokenized limit order book (LOB) messages. These messages are interpreted by a Jax-LOB simulator, which updates the LOB state. To handle long sequences efficiently, the model employs \emph{simplified structured state-space layers} to process sequences of order book states and tokenized messages. Using LOBSTER data of NASDAQ equity LOBs, we develop a custom tokenizer for message data, converting groups of successive digits to tokens, similar to tokenization in large language models.

## Repository Structure

The `lob/` package is organized **by pipeline stage** — each stage of the model
lifecycle lives in its own subpackage. Start here.

```
lob/                          LOB model source (S5), organized by pipeline stage
├── encode/                   Message tokenization (LOBSTER message <-> token ids)
│   ├── encoding.py               Canonical tokenizer: Message_Tokenizer, Vocab (26-token mode)
│   ├── encoding_1tok.py          Per-field vocab sizes + local<->global token mapping
│   ├── encoding_22tok.py         Legacy 22-token format (selectable in run_inference.py)
│   ├── encoding_26tok.py         26-token format (selectable in run_inference.py)
│   └── encoding_23tok.py …       Archived variants (incl. encoding_24tok.py) — not imported
├── preprocess/               Raw LOBSTER data -> arrays -> batched tensors
│   ├── preproc.py                .csv -> .npy preprocessing (run as: python -m lob.preprocess.preproc)
│   ├── lobster_dataloader.py     LOBSTER_Dataset (PyTorch), file caching, masking
│   └── dataloading.py            Dataset factory + DistributedSampler
├── model/                    Network definition
│   └── lob_seq_model.py          PaddedLobPredModel: stacked S5 layers (message + book fusion)
├── train/                    Training loop + distributed infrastructure
│   ├── train.py                  train(): epoch loop, validation, checkpointing, W&B
│   ├── train_helpers.py          JIT train/eval steps, cross-entropy loss, LR schedule, shard_map
│   ├── init_train.py             Model/TrainState init, Orbax checkpoint load/save
│   ├── sharding_utils.py         JAX mesh (1D flat / 2D hierarchical), data & param shardings
│   └── sweep.py                  W&B hyperparameter sweep
├── infer/                    Autoregressive generation
│   ├── inference.py              Generation with error correction
│   └── inference_no_errcorr.py   Generation without error correction
└── evaluate/                 Metrics
    ├── evaluation.py             Evaluation logic
    └── validation_helpers.py     Validation utilities

models/                           Sequence-model backbones: S5 / Mamba3 / GDN / FLA  (formerly s5/)
models/mamba3_cuda_kernels/       CUDA FFI kernels for the Mamba3 state scan
models/mamba3_triton_kernels/     Triton kernels for the Mamba3 SISO state scan (fwd+bwd)
bin/                          Experiment launch scripts
run_train.py                  Training entry point      -> lob.train.train.train()
run_eval.py                   Evaluation entry point
run_inference.py              Inference entry point
generate_data.py              Synthetic message generation from a trained model
train_full_autoreg.batch      SLURM launch script (see "Model & Training Reference")
```

### Subpackage responsibilities

| Subpackage | Responsibility | Key public symbols |
|:-----------|:---------------|:-------------------|
| `lob.encode` | Tokenize messages <-> ids | `Message_Tokenizer`, `Vocab`, `encode_msgs` |
| `lob.preprocess` | Raw data, datasets, loaders | `LOBSTER_Dataset`, `create_lobster_prediction_dataset`, `preproc` (CLI) |
| `lob.model` | S5 sequence model | `PaddedLobPredModel` (+ batched variants) |
| `lob.train` | Optimisation + multi-GPU | `train`, `create_train_state`, `init_train_state`, `initialize_mesh` |
| `lob.infer` | Autoregressive generation | `inference`, `inference_no_errcorr` |
| `lob.evaluate` | Metrics + validation | `evaluation`, `validation_helpers` |

### Import paths

Imports follow the subpackage layout:

```python
from lob.encode.encoding import Message_Tokenizer, Vocab
from lob.preprocess.dataloading import create_lobster_prediction_dataset
from lob.model.lob_seq_model import PaddedLobPredModel
from lob.train.train import train                  # the train() function
from lob.train.train_helpers import create_train_state
from lob.infer import inference_no_errcorr
from lob.evaluate import evaluation
```

## Data

The data used is NASDAQ LOB data from [LOBSTER](https://lobsterdata.com/index.php).
After downloading and unpacking, the raw files must be pre-processed into the
arrays the model consumes (see Quickstart step 2).

## Quickstart

```bash
# 1. Install (verified package versions are listed in the manifest below)
pip install -r requirements.txt

# 2. Preprocess LOBSTER data.
#    Run as a MODULE (-m): preproc.py now lives inside the lob.preprocess package,
#    so `python lob/preprocess/preproc.py` would not find `import lob`.
python -m lob.preprocess.preproc \
    --data_dir /path/to/LOBS5/data/GOOG/ \
    --save_dir /path/to/LOBS5/data/GOOG/ \
    --n_tick_range 500 --use_raw_book_repr

# 3. Train: single process, or launch the multi-node SLURM job
python run_train.py                  # see `python run_train.py --help` for args
sbatch train_full_autoreg.batch

# 4. Evaluate / generate
python run_eval.py
python run_inference.py
```

> 📖 **The "Model & Training Reference" section below documents the data format,
> model architecture, loss/optimization (with rendered math), distributed
> training, and reported metrics. The SLURM script `train_full_autoreg.batch`
> points back here.**

## Requirements & Installation

To install required packages, run `pip install -r requirements.txt`.

The GPU installation of JAX can cause problems, further instructions are available [here](https://github.com/google/jax#installation).

### Verified Base Conda Environment Package Manifest

The codebase has been verified on a base Conda environment (Python 3.12.11, verified on Job 2451587 for 2N/8GPU training + eval).

<details>
<summary><b>Click to expand the full package manifest</b></summary>

#### Critical ML Packages

| Package | Version | Notes |
|---------|---------|-------|
| jax | 0.9.0.1 | Main compute framework |
| jaxlib | 0.9.0.1 | |
| jax-cuda12-pjrt | 0.9.0.1 | CUDA 12 PJRT plugin |
| jax-cuda12-plugin | 0.9.0.1 | |
| jax-triton | 0.3.0 | Patched for JAX 0.9 + Triton 3.4 compat |
| flax | 0.12.2 | Neural network library |
| optax | 0.2.6 | Optimizer library |
| orbax-checkpoint | 0.11.32 | Checkpointing |
| chex | 0.1.91 | JAX testing utilities |
| ml_dtypes | 0.5.3 | ML data types (bf16 etc.) |
| tensorstore | 0.1.80 | Tensor storage for checkpoints |
| triton | 3.4.0 | GPU kernel compiler |

#### CUDA / GPU

| Package | Version | Notes |
|---------|---------|-------|
| nvidia-nccl-cu12 | 2.29.3 | NCCL (used via LD_LIBRARY_PATH override) |
| nvidia-cudnn-cu12 | 9.18.1.3 | |
| nvidia-cublas-cu12 | 12.9.1.4 | |
| nvidia-cuda-nvcc-cu12 | 12.9.86 | |
| nvidia-cuda-runtime-cu12 | 12.9.79 | |
| nvidia-cusolver-cu12 | 11.7.5.82 | |
| nvidia-cusparse-cu12 | 12.5.10.65 | |
| nvidia-cufft-cu12 | 11.4.1.4 | |
| nvidia-cuda-cupti-cu12 | 12.9.79 | |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | |
| nvidia-nvjitlink-cu12 | 12.9.86 | |
| nvidia-nvshmem-cu12 | 3.5.19 | |
| nvidia-cuda-cccl-cu12 | 12.9.27 | |

#### Data & Logging

| Package | Version | Notes |
|---------|---------|-------|
| wandb | 0.21.3 | Weights & Biases |
| tensorboard | 2.20.0 | |
| tensorboard-plugin-profile | 2.15.0 | XLA profiling |
| xprof | 2.20.7 | |

#### Scientific Computing

| Package | Version | Notes |
|---------|---------|-------|
| numpy | 2.3.3 | |
| scipy | 1.16.3 | |
| pandas | 2.3.2 | |
| scikit-learn | 1.8.0 | |
| matplotlib | 3.10.8 | |
| seaborn | 0.13.2 | |
| plotly | 6.5.1 | |
| statsmodels | 0.14.6 | |

#### PyTorch Ecosystem (coexists with JAX)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.8.0+cu129 | |
| torchvision | 0.23.0 | |
| flash_attn | 2.8.3 | Flash Attention |
| deepspeed | 0.17.6 | |
| transformer_engine | 2.11.0+c188b533 | |
| transformers | 4.56.0 | HuggingFace |
| accelerate | 1.10.1 | |
| peft | 0.17.1 | Parameter-efficient fine-tuning |
| megatron-core | 0.12.2 | |
| timm | 1.0.22 | |
| datasets | 4.1.1 | HuggingFace datasets |

#### Other Notable

| Package | Version | Notes |
|---------|---------|-------|
| ray | 2.49.2 | Distributed computing |
| tensorflow | 2.20.0 | |
| keras | 3.11.3 | |
| einops | 0.8.1 | Tensor operations |
| safetensors | 0.6.2 | Safe model serialization |
| polars | 1.34.0 | Fast dataframes |
| openai | 2.14.0 | |

#### Migration Notes (lob → base)

| Package | lob env | base env | Impact |
|---------|---------|----------|--------|
| jax/jaxlib | 0.6.1 | 0.9.0.1 | Major upgrade, no code changes needed |
| numpy | 1.26.4 | 2.3.3 | Major upgrade, compatible |
| flax | 0.11.2 | 0.12.2 | Minor upgrade |
| optax | 0.2.4 | 0.2.6 | Patch upgrade |
| nvidia-nccl-cu12 | 2.28.9 | 2.29.3 | Fixes ARM CAS hang bug |
| torch | 2.9.1 | 2.8.0+cu129 | lob was newer |
| triton | 3.5.1 | 3.4.0 | lob was newer |

</details>

## Model & Training Reference

This reference is migrated from the documentation block in `train_full_autoreg.batch`, with math rendered in LaTeX and all code locations updated to the subpackage layout above. It describes the default multi-node training configuration.

## Dataset

### Partitioning

| Partition | Source | Days | Method |
|:----------|:-------|:-----|:-------|
| Train | GOOG 2022 (full) | 249 | All files in main directory |
| Validation | GOOG 2022 (subset) | ~25 | Date-level random sampling from train (~10%) |
| Test | GOOG Jan 2023 (separate dir) | 9 | Temporal extrapolation |

**Key design.** The validation set is sampled at the *date* level from the training
set (not by random message sampling) to preserve time-series independence. The test
set uses independent Jan 2023 data to measure temporal extrapolation.

### Message Encoding (26 tokens)

Each LOB message is encoded as **26 tokens**: 8 new-event fields (16 tok) plus 4
reference fields (10 tok).

```
+----------------- New Event Fields (16 tok) -----------------++------- Reference Fields (10 tok) -------+
| idx:  0    1    2-4     5-6   7    8-10   11-12  13-15       || 16-18    19-20  21-22  23-25           |
|      evt  dir  price   size dt_s  dt_ns  time_s time_ns      || p_ref   sz_ref ts_ref tns_ref         |
+-------------------------------------------------------------++----------------------------------------+
                                       ^---- ignore_times removes idx 11-15 (time_s, time_ns)
```

| Token Index | Field | Count | Type | Range | Description |
|:------------|:------|:------|:-----|:------|:------------|
| 0 | event_type | 1 | event_type | {1,2,3,4} | 1=New, 2=Cancel, 3=Delete, 4=Trade |
| 1 | direction | 1 | direction | {0,1} | 0=Ask, 1=Bid |
| 2-4 | price | 3 | sign+price | $\text{sign}\in\{-1,+1\}$, $2\times[0,999]$ | Base-1000 relative price (ticks) |
| 5-6 | size | 2 | size_digit | $2\times[0,99]$ | Base-100 order quantity (0-9999) |
| 7 | delta_t_s | 1 | time | $[0,999]$ | Interval (seconds) |
| 8-10 | delta_t_ns | 3 | time | $3\times[0,999]$ | Interval (nanoseconds) |
| 11-12 | time_s | 2 | time | $2\times[0,999]$ | Exchange time sec (removed by `ignore_times`) |
| 13-15 | time_ns | 3 | time | $3\times[0,999]$ | Exchange time ns (removed by `ignore_times`) |
| 16-18 | price_ref | 3 | sign+price | same as price | Reference price |
| 19-20 | size_ref | 2 | size_digit | $2\times[0,99]$ | Reference quantity (base-100) |
| 21-22 | time_s_ref | 2 | time | $2\times[0,999]$ | Reference time sec |
| 23-25 | time_ns_ref | 3 | time | $3\times[0,999]$ | Reference time ns |

- `TOK_LENS = (1, 1, 3, 2, 1, 3, 2, 3, 3, 2, 2, 3)`, summing to **26**.
- `NEW_MSG_LEN = 16` (new-event fields only, excluding `_ref` fields).
- Encoding logic: `lob/encode/encoding.py` (`Message_Tokenizer`, `Vocab`, `encode_msg`, `decode_msg`).

### Vocabulary

| Domain | Size | Token Range | Description |
|:-------|:-----|:------------|:------------|
| Special tokens | 4 | 0-3 | MASK=0, HIDDEN=1, NA=2, START=3 |
| time | 1000 | 4-1003 | Digits 0-999, delimiters at [3,6,9,12] |
| event_type | 4 | 1004-1007 | Types 1-4 |
| size_digit | 100 | 1008-1107 | Base-100 digits 0-99 |
| price | 1000 | 1108-2107 | Price digits 0-999 |
| sign | 2 | 2108-2109 | Sign {-1, +1} |
| direction | 2 | 2110-2111 | Direction {0, 1} |
| **Total `n_classes`** | **2112** | | Returned by `Vocab.__len__()` |

### Sequence Dimensions

| Parameter | Value | Formula | Description |
|:----------|:------|:--------|:------------|
| `msg_seq_len` | 500 | (CLI arg) | History messages per sample |
| `MSG_LEN` | 26 | (constant) | Tokens per message |
| `seq_len` (total) | 13000 | $500 \times 26$ | Total tokens per sample |
| Effective loss tokens | 10500 | $500 \times 21$ | After removing time_s/ns (5 tok/msg) |
| `n_classes` | 2112 | vocab size | Output softmax classes |

---

## Model Architecture

A 75M-parameter S5 (Simplified Structured State Space) sequence model. Implemented
in `lob/model/lob_seq_model.py` (`PaddedLobPredModel`).

| Parameter | Value | Description |
|:----------|:------|:------------|
| `d_model` | 1024 | Hidden dimension $H$ |
| `n_layers` | 12 | Number of S5 blocks |
| `n_message_layers` | 2 | Message encoder layers |
| `n_book_pre_layers` | 1 | Order-book pre-processing layers |
| `n_book_post_layers` | 1 | Order-book post-processing layers |
| `blocks` | 16 | SSM discretization blocks $J$ |
| `ssm_size_base` | 1024 | SSM latent state size $P$ |
| `block_size` | 64 | $P / J = 1024 / 16$ |
| `conj_sym` | True | Conjugate-symmetry constraint |
| `clip_eigs` | True | Force stability |
| `bidirectional` | False | Unidirectional S5 |
| `activation_fn` | half_glu1 | GLU variant |
| `C_init` | trunc_standard_normal | Output initialization |
| `discretization` | zoh | Zero-order hold |
| `mode` | pool / none | Aggregation method |
| `prenorm` | True | Pre-LayerNorm |
| `batchnorm` | False | Uses LayerNorm |
| **Total params** | **~75M** | |

---

## Loss & Optimization

### Token-Level Cross-Entropy

The model emits per-position log-probabilities via log-softmax over the
$n_\text{classes}=2112$ vocabulary. For a token position with true label $y$ and
log-softmax output vector $\ell$, the per-token cross-entropy is

$$
\text{cross\\_entropy\\_loss}(\ell, y) = -\,\ell_y = -\log p(y)
\quad\text{where}\quad p(y) = \mathrm{softmax}(z)_y .
$$

Defined in `lob/train/train_helpers.py` (`cross_entropy_loss`); vectorized over the
batch and sequence dimensions. The batch loss is the mean over all (non-ignored)
positions:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \big(-\log p(y_i)\big).
$$

### Token Accuracy

Defined in `lob/train/train_helpers.py` (`compute_accuracy`):

$$
\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\!\left[\arg\max_{c}\, z_{i,c} = y_i\right].
$$

### `ignore_times` Mechanism

`ignore_times=True` does **not** change the model input (still 26 tokens); it only
masks the loss and accuracy computation. The 5 absolute-time tokens (idx 11-12
`time_s`, idx 13-15 `time_ns`) are removed before averaging.

$$
\mathcal{L}_{\text{full}} = \operatorname{mean}\big(\text{CE}[0:26]\big),
\qquad
\mathcal{L}_{\text{ignore\\_times}} = \operatorname{mean}\big(\text{CE}[0:11] \,\Vert\, \text{CE}[16:26]\big),
$$

where $\Vert$ denotes concatenation, giving 21 retained positions per message.

**Motivation.** Absolute timestamps (HH:MM:SS) are essentially unpredictable and
would otherwise contribute $5/26 \approx 19.2\%$ of the loss from noise. The
*interval* tokens `delta_t_s`/`delta_t_ns` are **kept**, since intervals are
informative.

Implementation (all in `lob/train/train_helpers.py`):

| Context | Function | Purpose |
|:--------|:---------|:--------|
| Training loss | `train_step()` -> `loss_fn()` | Remove idx 11-15 from CE loss |
| Hierarchical train loss | `sharded_step()` | Same, shard_map version |
| Validation/test loss | `eval_step()` | Remove idx 11-15 from CE loss |
| Validation/test accuracy | `eval_step()` | Remove idx 11-15 from accuracy |

Impact on the loss (input dimensions are unchanged; time-token gradients become 0):

| Dimension | No ignore_times | With ignore_times |
|:----------|:----------------|:------------------|
| Loss tokens / message | 26 | 21 ($-19.2\%$) |
| Loss tokens / sample | 13000 | 10500 ($500\times26 \to 500\times21$) |
| Model input tokens | 13000 | 13000 (unchanged) |
| Gradient dimensions | full | full (time grads = 0) |

### Optimizer

| Parameter | Value | Description |
|:----------|:------|:------------|
| `opt_config` | standard | SSM params use Adam, others use AdamW |
| `ssm_lr_base` | 5e-4 | SSM-param LR ($B$, $\Lambda$, `log_step`, ...) |
| `lr_factor` | 1 | Global LR $=$ `lr_factor` $\times$ `ssm_lr_base` |
| Global LR | 5e-4 | For $C$, dense, encoder, ... |
| `weight_decay` | 0.05 | Applies only to the regular param group |
| `p_dropout` | 0.0 | No dropout |
| `jax_seed` | 42 | Random seed |
| LR schedule | optax | Computed inside JIT (see below) |

### Learning-Rate Schedule (warmup + cosine)

A linear warmup over 1 epoch followed by cosine decay over the remaining 39 epochs.
Built in `lob/train/train_helpers.py` (LR-schedule creation). Let $t$ be the global
step, $\eta_{\max}=5\times10^{-4}$ the peak LR, $\eta_{\min}=0$, $T_w$ the warmup
steps, and $T_c$ the cosine steps.

$$
\eta(t) =
\begin{cases}
\eta_{\max}\,\dfrac{t+1}{T_w}, & 0 \le t < T_w \quad\text{(linear warmup)}\\[2.2ex]
\eta_{\min} + (\eta_{\max}-\eta_{\min})\cdot \dfrac{1}{2}\!\left(1 + \cos\!\left(\pi\,\dfrac{t-T_w}{T_c}\right)\right), & T_w \le t \le T_w + T_c \quad\text{(cosine decay)}
\end{cases}
$$

32-node specific values:

| Parameter | Value | Calculation |
|:----------|:------|:------------|
| `steps_per_epoch` | ~549 | $\text{train\\_size} / (\text{micro\\_bsz}\times \text{num\\_devices}\times \text{proc\\_count})$ |
| `warmup_steps` $T_w$ | ~549 | 1 epoch |
| `cosine_steps` $T_c$ | ~21,411 | $39 \times 549$ |
| `total_steps` | ~21,960 | $40 \times 549$ |

The LR is computed **inside JIT** from `state.step` via an `optax` schedule (rather
than `inject_hyperparams`), which avoids the NCCL deadlock observed with Python-side
scalar LR injection on multi-node runs.

---

## Distributed Training

### Hardware (32 nodes / 128 GPUs)

| Parameter | Value | Description |
|:----------|:------|:------------|
| `--nodes` | 32 | SLURM node count |
| `--gres=gpu` | 4 | GPUs per node |
| Total GPUs | 128 | $32 \times 4$ |
| GPU model | NVIDIA GH200 (sm_90a) | 96 GB HBM3, Grace CPU |
| Available memory | ~85.5 GB / GPU | 96 GB SKU |
| Intra-node interconnect | NV6 | ~159 GB/s per GPU pair |
| Per-GPU bandwidth | ~478 GB/s | $18 \text{ NVLinks} \times 26.56\,\text{GB/s}$ |
| Inter-node interconnect | Slingshot | HPE Cray |
| `--ntasks-per-node` | 1 | 1 process per node (4 GPUs) |
| Partition | workq | |

### Batch Size

| Parameter | Value | Formula |
|:----------|:------|:--------|
| `PER_GPU_BSZ` | 8 | env var, default 8 |
| `GPUS_PER_NODE` | 4 | fixed |
| `NNODES` | 32 | SLURM |
| `TOTAL_GPUS` | 128 | $32 \times 4$ |
| `GLOBAL_BSZ` | 1024 | $8 \times 128$ |
| `PER_PROCESS_BSZ` | 32 | $8 \times 4$ (DataLoader batch size) |

### Sharding & Gradient Sync

Configured in `lob/train/sharding_utils.py`; shard_map step in
`lob/train/train_helpers.py`.

| Parameter | 32N Value | Description |
|:----------|:----------|:------------|
| Mesh type | 2D (nodes, gpus) | Auto-enabled with `hierarchical=True` |
| Mesh shape | (32, 4) | 32 nodes $\times$ 4 GPUs |
| Data sharding | `P(('nodes','gpus'), None)` | Batch dim split across 128 GPUs |
| Param sharding | `P()` (replicated) | Each GPU holds a full copy |
| Gradient sync | Hierarchical AllReduce | `pmean(gpus)` then `pmean(nodes)` |
| Level 1 (NVLink) | `jax.lax.pmean` | Intra-node (4 GPUs), ~478 GB/s |
| Level 2 (Slingshot) | `jax.lax.pmean` | Across 32 nodes |
| `shard_autotuning` | False | Avoids 16N scaling regression |

Hierarchical AllReduce performs gradient averaging in two stages:

$$
g \leftarrow \operatorname{pmean}_{\text{gpus}}(g)
\quad\text{(NVLink, intra-node)}, \qquad
g \leftarrow \operatorname{pmean}_{\text{nodes}}(g)
\quad\text{(Slingshot, inter-node)} .
$$

### NCCL Configuration

| Env Variable | Value | Description |
|:-------------|:------|:------------|
| `NCCL_BUFFSIZE` | 2097152 (2 MB) | Critical: 16N efficiency 34.8% -> 83.3% |
| `NCCL_TIMEOUT` | 3600 (1h) | Prevents false timeouts |
| `NCCL_MIN_NCHANNELS` | 4 | Slingshot best practice |
| `NCCL_NCHANNELS_PER_NET` | 4 | GH200-cluster recommended |
| `NCCL_DEBUG` | INFO | Set to WARN after debugging |
| `NCCL_P2P_DISABLE` | 0 (default) | Setting to 1 causes ~25x slowdown |

### XLA / JAX Environment

| Env / Flag | 32N Value | Description |
|:-----------|:----------|:------------|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | true | GPU memory pooling |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.80 | Dual NCCL communicator limit |
| `JAX_PLATFORMS` | cuda | Force CUDA backend |
| `TF_GPU_ALLOCATOR` | cuda_malloc | Async GPU memory |
| `CUDA_MODULE_LOADING` | EAGER | Load modules at startup |
| `--xla_gpu_all_reduce_combine_threshold` | 128MB | Combine AllReduce ops |
| `--xla_gpu_enable_latency_hiding_scheduler` | true | Overlap compute/comm |
| `--xla_gpu_enable_highest_priority_async` | true | High-priority streams |
| `--xla_gpu_shard_autotuning` | false | Avoid autotuner overhead |

### Data Loading

DataLoader factory in `lob/preprocess/dataloading.py`; dataset in
`lob/preprocess/lobster_dataloader.py`; preprocessing in `lob/preprocess/preproc.py`.

| Parameter | Value | Description |
|:----------|:------|:------------|
| dataset | lobster-prediction | Dataset type |
| `msg_seq_len` | 500 | History length per sample |
| `use_book_data` | True | Include order book |
| `use_simple_book` | False | Not volume-image representation |
| `book_transform` | True | On-the-fly transformation |
| `book_depth` | 500 | Tick levels |
| `merging` | padded | Message + book merge method |
| `masking` | none | Full autoregressive |
| `n_data_workers` | 12 | CPU loading processes |
| `random_offsets_train` | True | Random offset start per epoch |
| `shuffle_train` | True | Shuffle training batches |
| `ignore_times` | True | Branch core: remove absolute time |
| Distributed sampler | DistributedSampler | Rank-based auto-sharding |

### Training Loop, Checkpoints, W&B

Driven by `lob/train/train.py`; init/checkpoint logic in `lob/train/init_train.py`.

| Parameter | Value | Notes |
|:----------|:------|:------|
| `epochs` | 40 | Production epochs |
| `early_stop_patience` | 1000 | Effectively disabled |
| `cosine_anneal` | True | Cosine schedule |
| `warmup_end` | 1 (epoch) | Linear warmup |
| `curtail_epochs` | None / 10 | Full / truncated testing |
| Checkpoint save interval | Every epoch | `save_interval_steps=1` |
| Max kept | 10 | `max_to_keep=10` |
| Permanent keeps | Every 5th | Epoch 0, 5, 10, ... |
| Async write | False | `fork()` breaks NCCL |
| Format | Orbax PyTree | `CheckpointManager`, rank 0 writes |
| `USE_WANDB` | True | rank 0 online, other ranks offline |
| `wandb_project` | lobs5-75M-B1 | |

---

## Training Metrics

### Per-Epoch W&B Metrics

Logged from `lob/train/train.py`.

| W&B Metric | Type | Computation | Description |
|:-----------|:-----|:------------|:------------|
| Training Loss | Train | `mean(batch_losses)` | Mean CE over epoch |
| Val loss | Val | `mean(concat_loss)` | Validation CE loss |
| Val Accuracy | Val | `mean(accuracies)` | Token-level accuracy |
| Test Loss | Test | `mean(concat_loss)` | Jan 2023 CE loss |
| Test Accuracy | Test | `mean(accuracies)` | Jan 2023 accuracy |
| count | Internal | early-stop counter | val_loss patience |
| Learning rate count | Internal | LR-decay counter | patience counter |
| Opt acc | Internal | best val accuracy | tracked best acc |
| lr | Hyperparam | `lr_schedule(step)` | Current global LR |
| ssm_lr | Hyperparam | `ssm_lr_schedule(step)` | Current SSM LR |
| block_size | Config | `ssm_size / blocks` | Logged at epoch 0 |

### W&B Run Summary (Final)

| Summary Metric | Description |
|:---------------|:------------|
| Best Val Loss | Best validation loss across all epochs |
| Best Val Accuracy | Best validation accuracy across all epochs |
| Best Epoch | Epoch achieving best Val Accuracy |
| Best Test Loss | Test loss at the best Val-Acc epoch |
| Best Test Accuracy | Test accuracy at the best Val-Acc epoch |

### Checkpoint-Stored Metrics

| Key | Type | Description |
|:----|:-----|:------------|
| `metrics.loss_train` | float | Last-epoch training loss |
| `metrics.loss_val_ar` | float | Validation loss (AR mode) |
| `metrics.loss_test_rnn` | float | Test loss |
| `metrics.acc_val_ar` | float | Validation accuracy (AR mode) |
| `metrics.acc_test_rnn` | float | Test accuracy |
| `config` | dict | Full args dictionary |
| `model` | PyTree | Full TrainState (params + opt) |

### Optional Per-Token CE Table (`log_ce_tables=True`)

Columns: `tok` (token index in $[0,\text{seq\\_len})$), and per-epoch
`val_ce_{epoch}`, `test_ce_{epoch}`, `val_acc_{epoch}`, `test_acc_{epoch}`,
`train_ce_{epoch}`.

---

## Performance Benchmarks

### Scaling Efficiency (verified)

| Config | Nodes | GPUs | Step Time (s/step) | Efficiency | Key Config |
|:-------|:------|:-----|:-------------------|:-----------|:-----------|
| 1N baseline | 1 | 4 | ~0.29 | 100% | 1D mesh |
| 4N hier. | 4 | 16 | — | 114.6% | 2D mesh |
| 16N hier. | 16 | 64 | 0.66 | 83.3% | BUFF=2MB |
| **32N hier.** | **32** | **128** | **0.94** | **58.5%** | **BUFF=2MB** |

### 32N Throughput

Token throughput is $\text{GLOBAL\\_BSZ}\times\text{seq\\_len}$ per step divided by
step time:

$$
\text{tokens/step} = 1024 \times 13000 = 13{,}312{,}000,
\qquad
\text{throughput} = \frac{13{,}312{,}000}{0.94} \approx 1.42\times10^{7}\ \text{tok/s}.
$$

| Metric | Value | Formula |
|:-------|:------|:--------|
| Step time | 0.94 s/step | |
| Tokens per step | 13,312,000 | $1024 \times 13000$ |
| **Token throughput** | **~14.2M tok/s** | $13{,}312{,}000 / 0.94$ |
| Samples per step | 1024 | `GLOBAL_BSZ` |
| **Sample throughput** | **~1089 samp/s** | $1024 / 0.94$ |
| Steps per epoch | ~549 | |
| **Time per epoch** | **~8.6 min** | $549 \times 0.94 / 60$ |
| **40-epoch total** | **~5.7 h** | $8.6 \times 40 / 60$ |
| **GPU-hours** | **~731 GPU-h** | $5.7 \times 128$ |

---

## Data Flow

```
+----------------------------------------------------------------------+
|                        DATA PIPELINE (32 Nodes)                      |
+----------------------------------------------------------------------+

     GOOG/2022/*.npy (249 days)          JAN2023/*.npy (9 days)
             |                                   |
             v                                   v
     +------------------------------------------------------+
     |              LOBSTER.setup()                         |
     |  train_files = 249 days (all)                        |
     |  val_files   = ~25 days (random 10% from train)      |
     |  test_files  = 9 days (separate directory)           |
     +------------------------------------------------------+
             |                    |                   |
             v                    v                   v
     +-------------+    +-------------+    +-------------+
     | LOBSTER_    |    | LOBSTER_    |    | LOBSTER_    |
     | Dataset     |    | Dataset     |    | Dataset     |
     | (train)     |    | (val)       |    | (test)      |
     | mask=none   |    | mask=none   |    | mask=none   |
     | rand_offset |    |             |    |             |
     +------+------+    +------+------+    +------+------+
            |                  |                   |
            v                  v                   v
     +-----------------------------------------------------+
     |        torch DataLoader (per rank)                  |
     |   batch_size=micro_bsz*num_devices, workers=12      |
     |   DistributedSampler (epoch-aware shuffle)          |
     |   Output: (messages[per_proc_bsz,500,26], book[...])|
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |  prep_batch() + make_array_from_process_local_data  |
     |  -> inputs:  (msg[per_proc_bsz,13000], book[...])   |
     |  -> labels:  [per_proc_bsz, 13000]                  |
     |  -> times:   [per_proc_bsz, 13000]                  |
     |  Sharding: P(('nodes','gpus'), None)                |
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |     shard_map(sharded_step, mesh=2D(32,4))          |
     |  Per-shard (1 GPU, micro_bsz=8):                    |
     |    logits = model(msg[8,13000], book[8,...])        |
     |    ce = cross_entropy_loss(logits, labels)          |
     |    if ignore_times:                                 |
     |       ce = ce[:, 0:11] || ce[:, 16:26]  (per msg)   |
     |    loss  = mean(ce);  grads = grad(loss)            |
     |    grads = pmean(grads, 'gpus')   <- NVLink         |
     |    grads = pmean(grads, 'nodes')  <- Slingshot      |
     |    state = apply_gradients(grads)                   |
     +-----------------------+-----------------------------+
                             |
                             v
     +-----------------------------------------------------+
     |  Per Epoch: W&B log + Orbax checkpoint (rank 0)     |
     +-----------------------------------------------------+
```

---

## Code Map

All `lob/` references use the refactored subpackage layout. Files are referenced by
**path + function/symbol name** (line numbers omitted, as the refactor shifted them).

| Feature | File | Symbol |
|:--------|:-----|:-------|
| Batch script (SLURM + env + params) | `train_full_autoreg.batch` | full file |
| Data path / `ignore_times=True` config | `train_full_autoreg.batch` | `DATA_DIR`, `TEST_DIR`, `--ignore_times` |
| Argparse parameter definitions | `run_train.py` | argparse block |
| Multi-node init (JAX distributed + SLURM) | `run_train.py` | `jax.distributed.initialize` |
| Training main loop | `lob/train/train.py` | `train()` |
| W&B per-epoch logging | `lob/train/train.py` | epoch-logging block |
| W&B run summary (best metrics) | `lob/train/train.py` | summary block |
| Checkpoint saving (Orbax + sync) | `lob/train/train.py` / `lob/train/init_train.py` | `CheckpointManager` save |
| LR schedule creation (warmup + cosine) | `lob/train/train_helpers.py` | LR-schedule builder |
| `cross_entropy_loss` ($-\ell_y$) | `lob/train/train_helpers.py` | `cross_entropy_loss` |
| `compute_accuracy` ($\arg\max = y$) | `lob/train/train_helpers.py` | `compute_accuracy` |
| JIT training step | `lob/train/train_helpers.py` | `train_step` / `loss_fn` |
| ignore_times (train) | `lob/train/train_helpers.py` | inside `train_step` |
| Hierarchical (shard_map) train step | `lob/train/train_helpers.py` | `sharded_step` |
| ignore_times (hierarchical) | `lob/train/train_helpers.py` | inside `sharded_step` |
| Eval (validation/test) step | `lob/train/train_helpers.py` | `eval_step` |
| ignore_times (eval loss + accuracy) | `lob/train/train_helpers.py` | inside `eval_step` |
| `TIME_START_I` / `TIME_END_I` constants | `lob/train/train_helpers.py` | module constants |
| Mesh / sharding construction | `lob/train/sharding_utils.py` | mesh + sharding helpers |
| Sweep driver | `lob/train/sweep.py` | sweep entry |
| Message tokenizer | `lob/encode/encoding.py` | `Message_Tokenizer` |
| Vocab construction | `lob/encode/encoding.py` | `Vocab` |
| `encode_msg` / `decode_msg` | `lob/encode/encoding.py` | `encode_msg`, `decode_msg` |
| Model (PaddedLobPredModel) | `lob/model/lob_seq_model.py` | `PaddedLobPredModel` |
| Dataset (LOBSTER_Dataset) | `lob/preprocess/lobster_dataloader.py` | `LOBSTER_Dataset` |
| DataLoader factory / setup | `lob/preprocess/dataloading.py` | dataset factory |
| Raw preprocessing | `lob/preprocess/preproc.py` | preprocessing entry |
| Inference (with error correction) | `lob/infer/inference.py` | autoregressive generation |
| Inference (no error correction) | `lob/infer/inference_no_errcorr.py` | autoregressive generation |
| Evaluation pipeline | `lob/evaluate/evaluation.py` | evaluation entry |
| Validation helpers | `lob/evaluate/validation_helpers.py` | validation utilities |

> **Refactor note (module layout).** The flat `lob/` package was reorganized into
> subpackages: `lob/encode/`, `lob/preprocess/`, `lob/model/`, `lob/train/`,
> `lob/infer/`, `lob/evaluate/`. Path mapping: `lob/encoding*.py` ->
> `lob/encode/encoding*.py`; `lob/lobster_dataloader.py`,
> `lob/dataloading.py`, and root `preproc.py` -> `lob/preprocess/`;
> `lob/lob_seq_model.py` -> `lob/model/`; `lob/train.py`, `lob/train_helpers.py`,
> `lob/init_train.py`, `lob/sharding_utils.py`, `lob/sweep.py` -> `lob/train/`;
> `lob/inference*.py` -> `lob/infer/`; `lob/evaluation.py`,
> `lob/validation_helpers.py` -> `lob/evaluate/`.

---

## Appendix: Legacy Encoding Archive (22 / 24-token modes)

The active pipeline supports **26-token mode only**. The following layouts are
preserved for reference.

### Legacy 22-Token Mode

22 tokens per message: 8 new-event fields (18 tok) + 4 reference fields (4 tok).

```
+--------------------- New Event Fields (18 tok) ---------------------++-- Reference Fields (4 tok) --+
| idx:  0    1    2-3   4    5    6-8    9-10   11-13                  | 14-15  16   17-18  19-21      |
|      evt  dir  price size dt_s dt_ns  time_s time_ns                | p_ref sz_r ts_ref tns_ref    |
+--------------------------------------------------------------------++------------------------------+
                                     ^---- ignore_times removes idx 9-13 (time_s, time_ns)
```

| Token Index | Field | Count | Type | Range | Description |
|:------------|:------|:------|:-----|:------|:------------|
| 0 | event_type | 1 | event_type | {1,2,3,4} | 1=New, 2=Cancel, 3=Delete, 4=Trade |
| 1 | direction | 1 | direction | {0,1} | 0=Ask, 1=Bid |
| 2-3 | price | 2 | sign+price | $\text{sign}\pm1$, val 999 | Relative price (ticks) |
| 4 | size | 1 | size | $[0,9999]$ | Order quantity (shares) |
| 5 | delta_t_s | 1 | time | $[0,999]$ | Interval (seconds) |
| 6-8 | delta_t_ns | 3 | time | $3\times[0,999]$ | Interval (nanoseconds) |
| 9-10 | time_s | 2 | time | $2\times[0,999]$ | Exchange time sec (removed) |
| 11-13 | time_ns | 3 | time | $3\times[0,999]$ | Exchange time ns (removed) |
| 14-15 | price_ref | 2 | sign+price | same as price | Reference price |
| 16 | size_ref | 1 | size | $[0,9999]$ | Reference quantity |
| 17-18 | time_s_ref | 2 | time | $2\times[0,999]$ | Reference time sec |
| 19-21 | time_ns_ref | 3 | time | $3\times[0,999]$ | Reference time ns |

Legacy 22-tok vocabulary uses a base-10000 size domain (`size` 0-9999 -> 10000
classes), giving `n_classes` $\approx 12012$. Sequence: `MSG_LEN=22`,
`seq_len` $= 500 \times 22 = 11000$, effective loss tokens $= 500 \times 17 = 8500$.

### Legacy 24-Token Mode

24-tok uses a base-1000 price and base-100 size, but only 1 token for `price` and
`price_ref` (restricting the range); vocab size is **2112**.

```
24tok: [evt:0, dir:1, price:2-3, size:4-5, dt_s:6, dt_ns:7-9, time_s:10-11,
        time_ns:12-14, price_ref_sign:15, price_ref:16, size_ref:17-18, time_ref:19-23]
```

### Legacy Performance (G2 experiment, 32N, 40 epochs)

| Tokenization | Val Acc | Test Acc |
|:-------------|:--------|:---------|
| 24tok | 78.87% | 75.89% |
| 22tok | 76.61% | 73.45% |

### Legacy Inference Note

At the time of the 26-tok migration, the inference modules
(`lob/infer/inference.py` and `lob/infer/inference_no_errcorr.py`) still used
22-token decode indices. Training/test loss were unaffected (the training pipeline
never calls decode/inference), but these modules must be updated before running
autoregressive generation.

## S5 Background: Simplified State Space Layers for Sequence Modeling

_By Jimmy Smith, Andrew Warrington & Scott Linderman._

This section is adapted from the S5 preprint (Smith et al. [2022], available [here](https://arxiv.org/pdf/2208.04933.pdf)) and the accompanying blog post. Code for the paper is available [here](https://github.com/lindermanlab/S5).

### TL;DR
In our preprint we demonstrate that we can build a state-of-the-art deep sequence-to-sequence model by stacking many dense, multi-input, multi-output (MIMO) state space models (SSMs) as a layer. This replaces the many single-input, single-output (SISO) SSMs used by the _structured state space sequence_ (S4) model [Gu et al, 2021]. This allows us to make use of efficient parallel scan to achieve the same computational efficiency of S4, without the need to use frequency domain and convolutional methods. We show that S5 achieves the same, if not better, performance than S4 on a range of long-range sequence modeling tasks.

### S4 is Epically Good. So... Why?
The performance of S4 is unarguable. Transformer-based methods were clawing for single percentage point gains on the long range arena benchmark dataset [Tay et al, 2021]. S4 beat many SotA transformer methods by as much as twenty percentage points. AND, to top it off, could process sequences with complexity linear in the sequence length, and sublinear in parallel time (with a reasonable number of processors).

However, the original S4 is a very involved method. It required specific matrix parameterizations, decompositions, mathematical identities, Fourier transforms, and more. As a research group, we spent several weeks trying to understand all the intricacies of the method. This left us asking: is there a different way of using the same core concepts, retaining performance and complexity, but, maybe, making it simpler?

Enter S5.

### From SISO to MIMO. From Convolution to Parallel Recurrence.
The S5 layer shifts from S4's many independent SISO systems to a single dense MIMO state space model. Instead of relying on frequency domain convolution for efficiency, S5 uses an associative parallel scan over the recurrence relation, achieving the same sublinear parallel time complexity while keeping the formulation simple and direct.

### S4 and Its Variants
Since publishing the original S4 model, the original authors have released three further papers studying the S4 model. Most significant of those papers are S4D [Gu, 2022] and DSS [Gupta, 2022]. These papers explore using diagonal state spaces, similar to what we use. S4D provided a proof as to why the (diagonalizable) normal matrix, from the normal-plus-low-rank factorization of the HiPPO-LegS matrix, provides such a good initialization for SISO systems. We show that using this initialization in the MIMO case enjoys similar characteristics. We note, however, that S4D and DSS provide computationally simpler implementations of S4; but do not perform quite as strongly. Most importantly, S5 isn't the only simplification to S4.

### Other Resources
- Much of our understanding and early code was based on the _excellent_ blog post, _The Annotated S4_, by [Rush and Karamcheti [2022]](https://srush.github.io/annotated-s4/).
- Full code for the original S4 implementation, and many of its forerunners and derivatives, is available [here](https://github.com/HazyResearch/state-spaces).
- Instructions for obtaining the LRA dataset are [here](https://openreview.net/pdf?id=qVyeW-grC2k).

### Awesome Other Work
There are obviously many other great researchers working on adapting, extending, and understanding S4. We outline some very recent work here:
- **Mega**, by Ma et al [2022], combines linear state space layers with transformer heads for sequence modeling. The main Mega method has $O(L^2)$ complexity. A second method, Mega-chunk, is presented that has $O(L)$, but does not achieve the same performance as Mega. Combining SSMs with transformer heads is a great avenue for future work.
- **Liquid-S4**, by Hasani et al [2022], extends S4 by adding a dependence on the input signal into the state matrix. When expanded, this is equivalent to adding cross-terms between the $k^{th}$ input and all previous inputs. Evaluating all previous terms is intractable, and so this sequence is often truncated. Extending the linear SSM, such that it is conditionally linear, is a really exciting opportunity for making the more model of linear state space layers more expressive.

### Bibliography
- Smith, Jimmy TH, Andrew Warrington, and Scott W. Linderman. "Simplified State Space Layers for Sequence Modeling." arXiv preprint arXiv:2208.04933 (2022). [Link](https://arxiv.org/pdf/2208.04933.pdf).
- Gu, Albert, Karan Goel, and Christopher Re. "Efficiently Modeling Long Sequences with Structured State Spaces." International Conference on Learning Representations (2021). [Link](https://openreview.net/pdf?id=uYLFoz1vlAC).
- Rush, Sasha, and Sidd Karamcheti. "The Annotated S4." Blog Track at ICLR 2022 (2022). [Link](https://srush.github.io/annotated-s4/).
- Yi Tay, et al. "Long Range Arena : A Benchmark for Efficient Transformers ." International Conference on Learning Representations (2021). [Link](https://openreview.net/pdf?id=qVyeW-grC2k).
- Ma, Xuezhe, et al. "Mega: Moving Average Equipped Gated Attention." arXiv preprint arXiv:2209.10655 (2022). [Link](https://arxiv.org/pdf/2209.10655).
- Hasani, Ramin, et al. "Liquid Structural State-Space Models." arXiv preprint arXiv:2209.12951 (2022). [Link](https://web10.arxiv.org/pdf/2209.12951.pdf).

## Citation

Citation details are omitted to preserve anonymity during double-blind review.

```
@article{anonymous,
  author  = {Anonymous Authors},
  title   = {Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network},
  note    = {Under review. Author and venue details withheld for double-blind review.},
  year    = {}
}
```
