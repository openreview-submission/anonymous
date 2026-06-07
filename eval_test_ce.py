#!/usr/bin/env python3
"""Test CE evaluation for Mamba3 SISO scaling law checkpoints.

Iterates over all checkpoints in a directory, evaluates per-ticker
test CE on Jan 2026 data, and writes a CSV for scaling law fitting.

Based on eval_per_field_accuracy.py (known working).

Usage:
    python eval_test_ce.py \
        --restore checkpoints/j3443014_zkrtl2ef_3443014 \
        --output_csv test_ce_8m.csv --model_label 8m
"""

import os
import sys
import numpy as onp

# --- Environment setup (must happen before JAX import) ---
if __name__ != "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"

from lob.preprocess.dataloading import Datasets

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
    parser.add_argument("--output_csv", type=str, default="test_ce.csv",
                        help="Output CSV path")
    parser.add_argument("--model_label", type=str, default="unknown",
                        help="Model label for CSV (e.g. 8m, 14m)")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated checkpoint steps to eval (default: all)")
    parser.add_argument("--per_position_csv", type=str, default=None,
                        help="If set, write per-position (token-index) CE + accuracy to this CSV. "
                             "Requires log_ce_tables=True in validate().")
    parser.add_argument("--per_position_final_only", type=str2bool, default=False,
                        help="If True together with --per_position_csv, only emit per-position rows "
                             "for the final (largest-step) checkpoint in the eval run, skipping "
                             "intermediate checkpoints. Used to keep per-position output size "
                             "tractable across the SP500 sweep (one ckpt per run × 487 tickers × "
                             "13,000 positions ≈ 215M rows total).")
    parser.add_argument("--ood_tickers", type=str, default=None,
                        help="Comma-separated OOD tickers (test-only, no training data)")
    parser.add_argument("--eval_only_ood", type=str2bool, default=False,
                        help="Only eval OOD tickers (skip in-distribution)")
    parser.add_argument("--ticker_index_json", type=str, default=None,
                        help="Path to a SquashFS index_<month>.json. If set, the union of "
                             "tickers seen in the index 'shapes' map is used as --tickers, "
                             "overriding any --tickers value. Used for SP500 eval where "
                             "passing 488 tickers on the CLI is unwieldy.")

    # Unused but needed for init_train_state compatibility
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_wd", type=float, default=None)
    parser.add_argument("--mini_epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--local_steps_k", type=int, default=0)

    args = parser.parse_args()

    # Post-parse: load tickers from squashfs index if provided.
    if args.ticker_index_json is not None:
        import json as _json
        with open(args.ticker_index_json) as _f:
            _idx = _json.load(_f)
        _shapes = _idx.get('shapes', {})
        _tk_set = set()
        for _key in _shapes.keys():
            # Layout: '<TICKER>/<TICKER>_<DATE>_..._proc.npy'
            _parts = _key.split('/')
            if len(_parts) >= 2:
                _tk_set.add(_parts[0])
        args.tickers = sorted(_tk_set)
        print(f"[ticker_index_json] loaded {len(args.tickers)} tickers from "
              f"{args.ticker_index_json}")

    # Post-parse: multi-ticker
    if args.tickers is not None and isinstance(args.tickers, str):
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
        if is_main:
            print(f"[Metadata] Loaded config from checkpoint: ssm_type={args.ssm_type}, "
                  f"d_model={args.d_model}, n_layers={args.n_layers}")
    except Exception as e:
        print(f"[Metadata] WARNING: Could not load checkpoint metadata: {e}")
        print(f"[Metadata] Using CLI args only")

    # ========== Create dataset ==========
    from lob.preprocess.lobster_dataloader import LOBSTER_Dataset
    from lob.preprocess.dataloading import create_lobster_prediction_dataset

    per_ticker_test_loaders = {}

    if getattr(args, 'eval_only_ood', False):
        # OOD-only mode: skip full dataset creation, use hardcoded 26tok dims
        from lob.encode.encoding import Message_Tokenizer as _MT, Vocab as _Vocab
        n_classes = len(_Vocab())  # 2112 for 26tok
        seq_len = args.msg_seq_len * _MT.MSG_LEN  # 500 * 26 = 13000
        in_dim = n_classes
        book_dim = args.book_depth + 3  # 503
        book_seq_len = args.msg_seq_len  # 500
        train_size = 0
        print(f"[*] OOD-only mode: n_classes={n_classes}, seq_len={seq_len}, "
              f"book_dim={book_dim}, book_seq_len={book_seq_len}")
    else:
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

    # OOD tickers: test-only, no training data — build loaders directly
    if args.ood_tickers:
        from torch.utils.data import DataLoader as TorchDataLoader
        from lob.preprocess.lobster_dataloader import discover_ticker_files, LOBSTER
        ood_list = [t.strip() for t in args.ood_tickers.split(',')]
        ood_test_dr = args.test_date_range
        ood_msg, ood_book = discover_ticker_files(
            args.data_root, ood_list, date_range=ood_test_dr)
        for tk in ood_list:
            ood_ds = LOBSTER_Dataset(
                message_files=ood_msg[tk],
                n_messages=args.msg_seq_len,
                mask_fn=LOBSTER_Dataset.no_mask,
                seed=args.jax_seed,
                n_cache_files=0,
                randomize_offset=False,
                book_files=ood_book.get(tk),
                book_transform=args.book_transform,
                book_depth=args.book_depth,
                token_mode=args.token_mode,
            )
            per_ticker_test_loaders[tk] = TorchDataLoader(
                ood_ds, batch_size=args.micro_bsz * args.num_devices,
                shuffle=False, drop_last=True, num_workers=0,
                collate_fn=LOBSTER._collate_fn,
            )
            print(f"[OOD] {tk}: {len(ood_msg[tk])} days, {len(ood_ds)} seqs, "
                  f"{len(per_ticker_test_loaders[tk])} batches")

    # ========== Init model ==========
    from lob.train.init_train import init_train_state, load_checkpoint, remap_train_state_step
    from lob.train.sharding_utils import initialize_mesh, create_state_shardings
    from lob.train.train_helpers import (validate, create_jit_eval_step)
    from lob.encode.encoding import Message_Tokenizer

    ssm_size = args.ssm_size_base

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

    val_model = model_cls(training=False, step_rescale=1)

    # Shard state to mesh
    state = jax.device_get(state)
    if jax.process_count() > 1:
        from jax.experimental.multihost_utils import broadcast_one_to_all
        state = broadcast_one_to_all(state)
    state_shardings = create_state_shardings(state, mesh)
    state = jax.device_put(state, state_shardings)

    jit_eval_step = create_jit_eval_step(mesh, state, has_book_data=args.use_book_data)

    # ========== Discover checkpoint steps ==========
    import csv
    import time
    import orbax.checkpoint as ocp

    ckpt_dir = os.path.abspath(args.restore)
    _ckpt_mngr = ocp.CheckpointManager(
        ckpt_dir, item_names=('state', 'metadata'),
        options=ocp.CheckpointManagerOptions())
    if args.steps:
        ckpt_steps = sorted(int(s) for s in args.steps.split(","))
    else:
        ckpt_steps = sorted(int(d) for d in os.listdir(ckpt_dir) if d.isdigit())
    print(f"[*] Checkpoints: {len(ckpt_steps)} steps ({ckpt_steps[0]}...{ckpt_steps[-1]})")

    # Training config for CSV metadata
    ckpt_meta = load_metadata(ckpt_dir)
    gBSZ_train = (getattr(ckpt_meta, 'micro_bsz', 10) *
                  getattr(ckpt_meta, 'num_devices', 4) *
                  getattr(ckpt_meta, 'process_count', 4))
    local_steps_k = getattr(ckpt_meta, 'local_steps_k', 1)
    num_params = sum(x.size for x in jax.tree.leaves(state.params))

    ignore_times = args.ignore_times
    # If --ood_tickers provided with --eval_only_ood, only eval OOD tickers
    if args.ood_tickers and getattr(args, 'eval_only_ood', False):
        ood_set = set(t.strip() for t in args.ood_tickers.split(','))
        ticker_names = sorted(t for t in per_ticker_test_loaders.keys() if t in ood_set)
        print(f"[*] Eval-only-OOD mode: evaluating {ticker_names}")
    else:
        ticker_names = sorted(per_ticker_test_loaders.keys())

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # ========== Eval loop: checkpoints × tickers ==========
    results = []
    per_pos_results = []
    total_evals = len(ckpt_steps) * len(ticker_names)
    eval_count = 0

    for step_i, step in enumerate(ckpt_steps):
        print(f"\n{'='*60}")
        print(f"[Checkpoint {step_i+1}/{len(ckpt_steps)}] step={step}")
        t0 = time.time()

        # Load checkpoint — reuse the same fallback pattern as initial restore
        try:
            ckpt = load_checkpoint(
                state, ckpt_dir, step=step, train=False,
                mesh=mesh, partial_restore=True)
            cur_state = ckpt['model']
        except (ValueError, TypeError) as e:
            print(f"  [Restore] StandardRestore failed, trying param-only fallback...")
            _restored = _ckpt_mngr.restore(
                step, args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state.params)))
            cur_state = state.replace(params=_restored['state'])
            print(f"  [Restore] Params loaded from step {step}")

        # Shard to mesh (reuse shardings from initial state)
        cur_state = jax.device_get(cur_state)
        cur_state = jax.device_put(cur_state, state_shardings)

        # Per-ticker eval
        for ticker in ticker_names:
            ticker_loader = per_ticker_test_loaders[ticker]
            eval_count += 1
            t1 = time.time()

            _is_final_step = (step == ckpt_steps[-1])
            _want_per_pos = args.per_position_csv is not None and (
                not args.per_position_final_only or _is_final_step
            )
            (t_loss, t_acc, ce_means, acc_means, t_lo_loss, t_lo_acc, t_lo_nll, _) = validate(
                cur_state, val_model.apply, ticker_loader,
                seq_len, in_dim, args.batchnorm, args.num_devices, 0,
                curtail_epoch=args.curtail_epochs,
                apply_method='__call_ar__',
                ignore_times=ignore_times,
                log_ce_tables=_want_per_pos,
                mesh=mesh,
                jit_eval_step_fn=jit_eval_step,
                silent=True)

            elapsed = time.time() - t1
            results.append({
                'model_label': args.model_label,
                'params': num_params,
                'd_model': args.d_model,
                'step': step,
                'gBSZ_train': gBSZ_train,
                'local_steps_k': local_steps_k,
                'seq_len': seq_len,
                'ticker': ticker,
                'test_ce': float(t_loss),
                'test_acc': float(t_acc),
                'last_order_ce': float(t_lo_loss),
                'last_order_nll': float(t_lo_nll),
            })
            if _want_per_pos and ce_means is not None:
                for pos_idx, (ce_p, acc_p) in enumerate(zip(ce_means, acc_means)):
                    per_pos_results.append({
                        'model_label': args.model_label,
                        'params': num_params,
                        'd_model': args.d_model,
                        'step': step,
                        'ticker': ticker,
                        'position_idx': pos_idx,
                        'ce': float(ce_p),
                        'acc': float(acc_p),
                    })
            print(f"  [{eval_count}/{total_evals}] {ticker}: "
                  f"CE={float(t_loss):.5f}  Acc={float(t_acc):.4f}  ({elapsed:.1f}s)")

        print(f"  Step {step} total: {time.time()-t0:.1f}s")
        write_csv(args.output_csv, results)
        if args.per_position_csv is not None and per_pos_results:
            write_csv(args.per_position_csv, per_pos_results)

    # ========== Summary ==========
    write_csv(args.output_csv, results)
    if args.per_position_csv is not None and per_pos_results:
        write_csv(args.per_position_csv, per_pos_results)
    print(f"\n[*] Done! {len(results)} rows → {args.output_csv}")

    final_rows = [r for r in results if r['step'] == ckpt_steps[-1]]
    if final_rows:
        print(f"\n[Final step {ckpt_steps[-1]}] Per-ticker test CE:")
        for r in final_rows:
            print(f"  {r['ticker']}: {r['test_ce']:.5f}")
        print(f"  MEAN: {onp.mean([r['test_ce'] for r in final_rows]):.5f}")
