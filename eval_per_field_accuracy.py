#!/usr/bin/env python3
"""Per-field accuracy evaluation for LOBS5 24-token encoding.

Loads a trained checkpoint, runs validate() with log_ce_tables=True,
and maps the per-token-position accuracy/CE back to semantic fields.

Usage (via SLURM):
    RESTORE_PATH=/path/to/checkpoint sbatch eval_per_field.batch

Or directly (on a compute node with GPUs):
    python eval_per_field_accuracy.py \
        --restore /path/to/checkpoint \
        --ignore_times True \
        --d_model 1024 --n_layers 12 --blocks 16 --ssm_size_base 1024 \
        --micro_bsz 10
"""

import os
import sys
import numpy as onp

# --- Environment setup (must happen before JAX import) ---
if __name__ != "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"

from lob.preprocess.dataloading import Datasets

# LOBSTER_Dataset.__init__ calls seq_offsets.share_memory_() unconditionally.
# torch's default 'file_descriptor' strategy backs that segment in /dev/shm,
# which is tiny/full on some compute nodes -> "No space left on device (28)".
# Switch to the 'file_system' strategy so the segment is a regular temp file in
# $TMPDIR instead. Harmless for n_data_workers=0 (no real cross-process sharing).
try:
    import torch.multiprocessing as _tmp
    _tmp.set_sharing_strategy("file_system")
except Exception:
    pass

if __name__ == "__main__":
    import argparse
    from models.utils.util import str2bool

    _n_gpus = int(os.environ.get('GPUS_PER_NODE', '4'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_n_gpus))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("NCCL_TIMEOUT", "600")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")

    parser = argparse.ArgumentParser(description="Per-field accuracy eval for LOBS5")

    # Checkpoint
    parser.add_argument("--restore", type=str, required=True,
                        help="Checkpoint directory to restore from")
    parser.add_argument("--restore_step", type=int, default=None,
                        help="Step to restore (default: latest)")

    # Data
    parser.add_argument("--dir_name", type=str, default='./data/',
                        help="Data directory")
    parser.add_argument("--test_dir_name", type=str, default=None,
                        help="Separate test data directory")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for multi-ticker data")
    parser.add_argument("--train_date_range", type=str, default=None,
                        help="Train date range (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--test_date_range", type=str, default=None,
                        help="Test date range (YYYY-MM-DD,YYYY-MM-DD)")

    # Model config
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--ssm_size_base", type=int, default=1024)
    parser.add_argument("--ssm_type", type=str, default="s5")
    parser.add_argument("--micro_bsz", type=int, default=10)
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument("--msg_seq_len", type=int, default=500)
    parser.add_argument("--n_message_layers", type=int, default=2)
    parser.add_argument("--n_book_pre_layers", type=int, default=1)
    parser.add_argument("--n_book_post_layers", type=int, default=1)
    parser.add_argument("--activation_fn", type=str, default="half_glu1")
    parser.add_argument("--C_init", type=str, default="trunc_standard_normal")
    parser.add_argument("--conj_sym", type=str2bool, default=True)
    parser.add_argument("--clip_eigs", type=str2bool, default=True)
    parser.add_argument("--bidirectional", type=str2bool, default=False)
    parser.add_argument("--prenorm", type=str2bool, default=True)
    parser.add_argument("--batchnorm", type=str2bool, default=False)
    parser.add_argument("--bn_momentum", type=float, default=0.95)
    parser.add_argument("--dt_min", type=float, default=0.001)
    parser.add_argument("--dt_max", type=float, default=0.1)
    parser.add_argument("--dt_global", type=str2bool, default=False)
    parser.add_argument("--discretization", type=str, default="zoh")
    parser.add_argument("--mode", type=str, default="none")
    parser.add_argument("--p_dropout", type=float, default=0.0)
    parser.add_argument("--token_mode", type=str, default="24tok")

    # Mamba3-specific args
    parser.add_argument("--mamba3_d_state", type=int, default=128)
    parser.add_argument("--mamba3_expand", type=int, default=2)
    parser.add_argument("--mamba3_headdim", type=int, default=64)
    parser.add_argument("--mamba3_chunk_size", type=int, default=64)
    parser.add_argument("--mamba3_rope_fraction", type=float, default=0.5)
    parser.add_argument("--mamba3_use_triton", type=str2bool, default=False)

    # Training config (needed for model init)
    parser.add_argument("--ignore_times", type=str2bool, default=True)
    parser.add_argument("--ssm_lr_base", type=float, default=5e-4)
    parser.add_argument("--lr_factor", type=float, default=1.0)
    parser.add_argument("--warmup_end", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--opt_config", type=str, default="standard")
    parser.add_argument("--cosine_anneal", type=str2bool, default=True)
    parser.add_argument("--lr_min", type=float, default=0)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--reduce_factor", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=1000)
    parser.add_argument("--jax_seed", type=int, default=42)

    # Data loading
    parser.add_argument("--use_book_data", type=str2bool, default=True)
    parser.add_argument("--use_simple_book", type=str2bool, default=False)
    parser.add_argument("--book_transform", type=str2bool, default=True)
    parser.add_argument("--book_depth", type=int, default=500)
    parser.add_argument("--masking", type=str, default="none")
    parser.add_argument("--merging", type=str, default="padded")
    parser.add_argument("--dataset", type=str, default="lobster-prediction")
    parser.add_argument("--n_data_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--random_offsets_train", type=str2bool, default=True)
    parser.add_argument("--shuffle_train", type=str2bool, default=True)
    parser.add_argument("--val_split", type=float, default=0.01)
    parser.add_argument("--debug_overfit", type=str2bool, default=False)

    # Eval-specific
    parser.add_argument("--curtail_epochs", type=int, default=None,
                        help="Limit eval batches (None = full eval)")
    parser.add_argument("--hierarchical", type=str2bool, default=True)

    # Unused but needed for init_train_state compatibility
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_wd", type=float, default=None)
    parser.add_argument("--mini_epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--local_steps_k", type=int, default=0)

    args = parser.parse_args()

    # Post-parse: multi-ticker
    if args.tickers is not None:
        args.tickers = [t.strip() for t in args.tickers.split(',')]
    if args.train_date_range is not None:
        parts = args.train_date_range.split(',')
        args.train_date_range = (parts[0].strip(), parts[1].strip())
    if args.test_date_range is not None:
        parts = args.test_date_range.split(',')
        args.test_date_range = (parts[0].strip(), parts[1].strip())

    # ========== JAX distributed init ==========
    import jax
    import jax.numpy as jnp
    from jax import random

    process_count = int(os.environ.get('SLURM_NNODES', '1'))
    is_distributed = process_count > 1

    if is_distributed:
        jax.distributed.initialize()
        args.process_index = jax.process_index()
        args.process_count = jax.process_count()
        args.is_distributed = True
        print(f"[Distributed] Process {jax.process_index()}/{jax.process_count()}")
    else:
        args.process_index = 0
        args.process_count = 1
        args.is_distributed = False

    is_main = args.process_index == 0

    # ========== Load checkpoint metadata and merge with CLI args ==========
    # This ensures model config matches the checkpoint exactly
    from lob.train.init_train import load_metadata
    try:
        ckpt_args = load_metadata(args.restore)
        # Merge: CLI args override checkpoint metadata, but checkpoint fills in
        # any args not provided on the CLI (like mamba3-specific config)
        for k, v in vars(ckpt_args).items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
        # Force key model config from checkpoint (not CLI defaults)
        for k in ['ssm_type', 'd_model', 'n_layers', 'blocks', 'ssm_size_base',
                   'n_message_layers', 'n_book_pre_layers', 'n_book_post_layers',
                   'mamba3_d_state', 'mamba3_expand', 'mamba3_headdim',
                   'mamba3_chunk_size', 'mamba3_rope_fraction', 'mamba3_use_triton',
                   'gdn_num_heads', 'gdn_head_dim', 'gdn_expand_v', 'gdn_chunk_size',
                   'gdn_use_conv', 'activation_fn']:
            if hasattr(ckpt_args, k):
                setattr(args, k, getattr(ckpt_args, k))
        # Force standard optimizer for eval — avoids Muon/optax tree structure mismatch
        args.opt_config = "standard"
        # tp_size from checkpoint is preserved — TP inference supported via shard_map
        if is_main:
            print(f"[Metadata] Loaded config from checkpoint: ssm_type={args.ssm_type}, "
                  f"d_model={args.d_model}, n_layers={args.n_layers}")
    except Exception as e:
        print(f"[Metadata] WARNING: Could not load checkpoint metadata: {e}")
        print(f"[Metadata] Using CLI args only")

    # ========== Create dataset ==========
    from lob.preprocess.lobster_dataloader import LOBSTER_Dataset
    from lob.preprocess.dataloading import create_lobster_prediction_dataset

    mask_fn = LOBSTER_Dataset.no_mask

    process_rank = args.process_index
    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            micro_bsz=args.micro_bsz,
            num_devices=args.num_devices,
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            n_data_workers=args.n_data_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            val_split=args.val_split,
            test_dir_name=args.test_dir_name,
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=args.process_count,
            tickers=args.tickers,
            data_root=args.data_root,
            train_date_range=args.train_date_range,
            test_date_range=args.test_date_range,
            token_mode=args.token_mode,
        )

    per_ticker_test_loaders = aux_dataloaders.get('per_ticker_test', {})

    # ========== Init model ==========
    from lob.train.init_train import init_train_state, load_checkpoint, remap_train_state_step
    from lob.train.sharding_utils import initialize_mesh, create_state_shardings
    from lob.train.train_helpers import (validate, create_jit_eval_step)
    from lob.encode.encoding import Message_Tokenizer

    ssm_size = args.ssm_size_base

    # ====== Legacy-checkpoint Mamba3 BCNorm/out_norm epsilon compatibility ======
    # Checkpoints saved before commit ffe45d66 / 13f04e52 (2026-04-22) were trained
    # with nn.RMSNorm(N) for B_norm/C_norm (epsilon=d_state=1024) and
    # nn.RMSNorm(d_inner) for out_norm (epsilon=d_inner=2048). Those positional args
    # set epsilon to the feature size, silently near-disabling the norm; the weights
    # were learned under that regime. The current models/mamba3.py (read-only here) uses
    # the corrected epsilon=1e-6, which scrambles B/C/out for the old weights and
    # yields near-random logits (overall CE ~8 instead of ~0.6). models/mamba3.py is not
    # writable from this account, so we reproduce the training-time normalizers via a
    # scoped runtime monkeypatch of Mamba3SSM.setup. Opt-in via MAMBA3_LEGACY_NORM=1
    # (the eval batch sets it for the 2026-03-28 78M baseline ckpt).
    if getattr(args, 'ssm_type', None) == 'mamba3' and os.environ.get('MAMBA3_LEGACY_NORM'):
        import models.mamba3 as _m3mod
        from flax import linen as _nn
        _orig_setup = _m3mod.Mamba3SSM.setup

        def _legacy_setup(self):
            # During the original setup, B_norm/C_norm are the ONLY nn.RMSNorm
            # constructions inside Mamba3SSM.setup (out_norm uses _FullScaleParam).
            # Temporarily swap nn.RMSNorm for a version that injects epsilon=d_state
            # so the created B_norm/C_norm match the legacy normalizer.
            _real_rmsnorm = _m3mod.nn.RMSNorm
            _legacy_eps_bc = float(self.d_state)

            def _rmsnorm_legacy(*a, **kw):
                kw.setdefault('epsilon', _legacy_eps_bc)
                return _real_rmsnorm(*a, **kw)

            _m3mod.nn.RMSNorm = _rmsnorm_legacy
            try:
                _orig_setup(self)
            finally:
                _m3mod.nn.RMSNorm = _real_rmsnorm
            # out_norm epsilon was d_inner under the legacy nn.RMSNorm(d_inner).
            self._out_norm_eps = float(self.d_inner)

        _m3mod.Mamba3SSM.setup = _legacy_setup
        if is_main:
            print("[LegacyNorm] Mamba3 BCNorm eps=d_state, out_norm eps=d_inner "
                  "(pre-ffe45d66 checkpoint compatibility ENABLED)")

    state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        train_size=train_size,
        print_shapes=is_main
    )

    # Mesh setup
    use_hierarchical = args.hierarchical if is_distributed else False
    if jax.process_count() > 1:
        mesh = initialize_mesh(jax.device_count(), hierarchical=use_hierarchical)
    else:
        mesh = initialize_mesh(args.num_devices)

    # ========== Restore checkpoint ==========
    # Optional diagnostic: capture L2 norms of a few params BEFORE restore so we
    # can compare against AFTER restore. Equal norms => restore did not land
    # trained values; different norms => restore applied (suspect B vs C bisect).
    def _diag_param_norms(params, tag):
        import jax.numpy as _jnp
        flat = jax.tree_util.tree_leaves(params)
        total = float(sum(float(_jnp.sum(_jnp.asarray(x).astype('float32') ** 2)) for x in flat)) ** 0.5
        print(f"[DIAG {tag}] total param L2 = {total:.6f}  (n_leaves={len(flat)})")
        # Probe a specific fused_s5 leaf if present
        def _walk(d, path=()):
            if isinstance(d, dict):
                for k, v in d.items():
                    yield from _walk(v, path + (k,))
            else:
                yield path, d
        for kp, arr in _walk(params):
            name = '/'.join(str(k) for k in kp)
            if 'fused_s5' in name and ('A_log' in name or 'in_proj' in name or 'out_proj' in name):
                a = jax.numpy.asarray(arr).astype('float32')
                print(f"[DIAG {tag}]   {name}: shape={tuple(a.shape)} L2={float((a**2).sum())**0.5:.6f}")
                break

    if os.environ.get('DIAG_PARAM_NORM'):
        _diag_param_norms(state.params, 'BEFORE')

    print(f"[*] Restoring from: {args.restore}")
    try:
        ckpt = load_checkpoint(
            state,
            args.restore,
            step=args.restore_step,
            mesh=mesh,
            train=False,
            partial_restore=True,
        )
        state = ckpt['model']
        print(f"[Restore] state.step = {int(state.step)}")
    except (ValueError, TypeError) as _ckpt_err:
        # Muon optimizer state tree != AdamW tree — use inference-style fallback
        print(f"[Restore] StandardRestore failed: {_ckpt_err}")
        print("[Restore] Falling back to inference-style param-only restore...")
        import orbax.checkpoint as ocp
        from lob.train.init_train import deduplicate_trainstate

        _abs_path = os.path.abspath(args.restore)
        _mngr = ocp.CheckpointManager(
            _abs_path, item_names=('state', 'metadata'),
            options=ocp.CheckpointManagerOptions(),
        )
        _step = args.restore_step if args.restore_step else _mngr.latest_step()

        # Extract just the params subtree, ignoring optimizer state
        _restored = _mngr.restore(
            _step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(state.params)),
        )
        state = state.replace(params=_restored['state'])
        print(f"[Restore] Params loaded from step {_step} (optimizer state skipped)")

    if os.environ.get('DIAG_PARAM_NORM'):
        _diag_param_norms(state.params, 'AFTER')

    val_model = model_cls(training=False, step_rescale=1)

    # NOTE: the autoregressive eval path (apply_method='__call_ar__') processes
    # full sequences and never consumes a pre-built RNN carry. The old
    # initialize_carry() call here was dead code that also crashed for GDN
    # (its carry kwargs, e.g. head_dim, were not forwarded), so it is removed.

    # Shard state to mesh
    state = jax.device_get(state)
    if jax.process_count() > 1:
        from jax.experimental.multihost_utils import broadcast_one_to_all
        state = broadcast_one_to_all(state)
    state_shardings = create_state_shardings(state, mesh)
    state = jax.device_put(state, state_shardings)

    jit_eval_step = create_jit_eval_step(mesh, state, has_book_data=args.use_book_data)

    # ========== Diagnostic: eager __call_ar__ vs JIT eval_step on ONE batch ==========
    # Bisects "is the JIT/sharded eval path corrupting Mamba3 logits vs the proven
    # eager closed-loop forward?" Pull one test batch, run BOTH ways, compare CE.
    if os.environ.get('DIAG_EAGER_VS_JIT'):
        from lob.train.train_helpers import prep_batch, repeat_book, _compute_ce_unified
        _loader = testloader if testloader is not None else valloader
        _b = next(iter(_loader))
        _inp, _lab, _its = prep_batch(_b, seq_len, args.num_devices)
        # Host-side single copy for eager run (take first micro example).
        host_state = jax.device_get(state)
        _inp_h = tuple(onp.asarray(x) for x in _inp)
        _lab_h = onp.asarray(_lab)
        _its_h = tuple(onp.asarray(x) for x in _its)
        print(f"[DIAG E-vs-J] batch inputs shapes: {[x.shape for x in _inp_h]} labels {_lab_h.shape}")
        # --- EAGER path: mirror extract_ssd_closedloop (disable_jit, __call_ar__) ---
        import jax as _jax
        _eager_inp = _inp_h
        if len(_eager_inp) > 1:
            _eager_inp = repeat_book(*_eager_inp, True)
        with _jax.disable_jit():
            _logits_e, _ = val_model.apply({"params": host_state.params},
                                           *_eager_inp, *_its_h,
                                           rngs={"dropout": _jax.random.key(0)},
                                           mutable=["intermediates"],
                                           method='__call_ar__')
        _ce_e = _compute_ce_unified(onp.asarray(_logits_e), _lab_h, args.ignore_times)
        print(f"[DIAG E-vs-J] EAGER  logits shape={tuple(onp.asarray(_logits_e).shape)} "
              f"mean={float(onp.mean(_logits_e)):.4f} std={float(onp.std(_logits_e)):.4f} "
              f"CE={float(onp.mean(_ce_e)):.5f}")
        # --- JIT path: exactly what validate() runs ---
        from lob.train.sharding_utils import get_data_shardings_for_batch
        _ish, _lsh, _tsh = get_data_shardings_for_batch(mesh, has_book_data=(len(_inp) > 1))
        _inp_j = tuple(jax.make_array_from_process_local_data(sh, x) for x, sh in zip(_inp, _ish))
        _lab_j = jax.make_array_from_process_local_data(_lsh, _lab)
        _its_j = tuple(jax.make_array_from_process_local_data(sh, x) for x, sh in zip(_its, _tsh))
        _loss_j, _acc_j, _logits_j = jit_eval_step(
            _inp_j, _lab_j, _its_j, state, val_model.apply, args.batchnorm,
            '__call_ar__', (onp.array([0])), args.ignore_times)
        print(f"[DIAG E-vs-J] JIT    logits shape={tuple(onp.asarray(_logits_j).shape)} "
              f"mean={float(onp.mean(onp.asarray(_logits_j))):.4f} "
              f"std={float(onp.std(onp.asarray(_logits_j))):.4f} "
              f"CE={float(onp.mean(onp.asarray(_loss_j))):.5f}")

    # ========== Define field mapping ==========
    # 24-token layout per message:
    #   idx 0:       event_type    (1 tok)
    #   idx 1:       direction     (1 tok)
    #   idx 2-3:     price         (2 tok: sign + value)
    #   idx 4-5:     size          (2 tok: high + low, base-100)
    #   idx 6:       delta_t_s     (1 tok)
    #   idx 7-9:     delta_t_ns    (3 tok)
    #   idx 10-11:   time_s        (2 tok) -- excluded by ignore_times
    #   idx 12-14:   time_ns       (3 tok) -- excluded by ignore_times
    #   idx 15-16:   price_ref     (2 tok: sign + value)
    #   idx 17-18:   size_ref      (2 tok: high + low, base-100)
    #   idx 19-20:   time_s_ref    (2 tok)
    #   idx 21-23:   time_ns_ref   (3 tok)

    # Detect encoding from Message_Tokenizer.MSG_LEN
    from lob.encode.encoding import Message_Tokenizer as _MT
    _msg_len = _MT.MSG_LEN
    if _msg_len == 26:
        FIELDS = [
            ('event_type', 1),
            ('direction', 1),
            ('price', 3),        # 26tok: sign + high + low (base-1000)
            ('size', 2),
            ('delta_t_s', 1),
            ('delta_t_ns', 3),
            ('time_s', 2),
            ('time_ns', 3),
            ('price_ref', 3),    # 26tok: sign + high + low (base-1000)
            ('size_ref', 2),
            ('time_s_ref', 2),
            ('time_ns_ref', 3),
        ]
        print(f"[Encoding] 26tok detected (MSG_LEN={_msg_len}), using 26-tok field layout")
    else:
        FIELDS = [
            ('event_type', 1),
            ('direction', 1),
            ('price', 2),        # 24tok: sign + value
            ('size', 2),
            ('delta_t_s', 1),
            ('delta_t_ns', 3),
            ('time_s', 2),
            ('time_ns', 3),
            ('price_ref', 2),    # 24tok: sign + value
            ('size_ref', 2),
            ('time_s_ref', 2),
            ('time_ns_ref', 3),
        ]
        print(f"[Encoding] 24tok detected (MSG_LEN={_msg_len}), using 24-tok field layout")

    TIME_FIELDS = {'time_s', 'time_ns'}  # excluded by ignore_times

    def map_positions_to_fields(values, ignore_times):
        """Map per-position accuracy/CE array to per-field averages.

        values: 1D array of length tpm (21 if 26tok+ignore_times, 19 if 24tok+ignore_times)
        Returns: list of (field_name, n_tokens, mean_value, per_token_values)
        """
        results = []
        pos = 0
        for field_name, n_tok in FIELDS:
            if ignore_times and field_name in TIME_FIELDS:
                continue
            field_vals = values[pos:pos + n_tok]
            results.append((field_name, n_tok, float(onp.mean(field_vals)),
                           [float(v) for v in field_vals]))
            pos += n_tok
        assert pos == len(values), f"Position mismatch: {pos} != {len(values)}"
        return results

    # ========== Run eval ==========
    ignore_times = args.ignore_times

    for split_name, loader in [("val", valloader), ("test", testloader)]:
        if loader is None:
            continue
        print(f"\n{'='*60}")
        print(f"  Evaluating: {split_name} split (ignore_times={ignore_times})")
        print(f"{'='*60}")

        (avg_loss, avg_acc, ce_means, acc_means,
         last_order_loss, last_order_acc,
         last_order_nll, all_orders_nll) = validate(
            state,
            val_model.apply,
            loader,
            seq_len,
            in_dim,
            args.batchnorm,
            args.num_devices,
            epoch=0,
            curtail_epoch=args.curtail_epochs,
            ignore_times=ignore_times,
            apply_method='__call_ar__',
            log_ce_tables=True,
            mesh=mesh,
            jit_eval_step_fn=jit_eval_step,
            silent=False)

        print(f"\n  Overall {split_name}:")
        print(f"    Avg Loss: {avg_loss:.5f}   Avg Acc: {avg_acc:.4f}")
        print(f"    Last-Order Loss: {last_order_loss:.5f}  Acc: {last_order_acc:.4f}")
        print(f"    All-Orders NLL: {all_orders_nll:.4f}  Last-Order NLL: {last_order_nll:.4f}")

        if acc_means is not None and ce_means is not None:
            tpm = len(acc_means)
            expected_tpm = 19 if ignore_times else 24
            print(f"\n  Per-position array length: {tpm} (expected {expected_tpm})")

            field_acc = map_positions_to_fields(acc_means, ignore_times)
            field_ce = map_positions_to_fields(ce_means, ignore_times)

            # Print per-field table
            print(f"\n  Per-Field Accuracy & CE ({split_name}):")
            print(f"  {'Field':<16} {'Tok':>3} {'Accuracy':>10} {'CE':>10} {'Per-Token Acc':>40}")
            print(f"  {'─'*16} {'─'*3} {'─'*10} {'─'*10} {'─'*40}")

            for (fname, ntok, facc, tok_accs), (_, _, fce, tok_ces) in zip(field_acc, field_ce):
                tok_acc_str = ', '.join(f'{a:.4f}' for a in tok_accs)
                print(f"  {fname:<16} {ntok:>3} {facc:>10.4f} {fce:>10.4f} [{tok_acc_str}]")

            # Grouped summary
            print(f"\n  Grouped Summary ({split_name}):")
            groups = {
                'event_type': ['event_type'],
                'direction': ['direction'],
                'price': ['price'],
                'size': ['size'],
                'delta_t': ['delta_t_s', 'delta_t_ns'],
                'price_ref': ['price_ref'],
                'size_ref': ['size_ref'],
                'time_ref': ['time_s_ref', 'time_ns_ref'],
            }
            if not ignore_times:
                groups['time_abs'] = ['time_s', 'time_ns']

            print(f"  {'Group':<16} {'Accuracy':>10} {'CE':>10}")
            print(f"  {'─'*16} {'─'*10} {'─'*10}")

            for group_name, group_fields in groups.items():
                g_accs = []
                g_ces = []
                for (fname, ntok, facc, tok_accs), (_, _, fce, tok_ces) in zip(field_acc, field_ce):
                    if fname in group_fields:
                        g_accs.extend(tok_accs)
                        g_ces.extend(tok_ces)
                if g_accs:
                    print(f"  {group_name:<16} {onp.mean(g_accs):>10.4f} {onp.mean(g_ces):>10.4f}")

            # Per-ticker test
            if split_name == "test" and per_ticker_test_loaders:
                print(f"\n  Per-Ticker Test Accuracy:")
                print(f"  {'Ticker':<10} {'Avg Loss':>10} {'Avg Acc':>10}")
                print(f"  {'─'*10} {'─'*10} {'─'*10}")
                for ticker, ticker_loader in per_ticker_test_loaders.items():
                    (t_loss, t_acc, _, _, _, _, _, _) = validate(
                        state, val_model.apply, ticker_loader,
                        seq_len, in_dim, args.batchnorm, args.num_devices, 0,
                        curtail_epoch=args.curtail_epochs,
                        apply_method='__call_ar__',
                        ignore_times=ignore_times,
                        log_ce_tables=False,
                        mesh=mesh,
                        jit_eval_step_fn=jit_eval_step,
                        silent=True)
                    print(f"  {ticker:<10} {t_loss:>10.5f} {t_acc:>10.4f}")

    print("\n[*] Per-field evaluation complete.")
