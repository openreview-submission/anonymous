import os
import sys
import time
import json
import jax
from jax import random
from jax.experimental.multihost_utils import sync_global_devices
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import wandb
import gc

from lob.train.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate, remap_train_state_step
from lob.preprocess.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader
from lob.preprocess.lobster_dataloader import LOBSTER_Dataset
from lob.train.train_helpers import reduce_lr_on_plateau, train_epoch, validate, \
    create_jit_train_step, create_jit_eval_step, create_lobs5_learning_rate_schedule, \
    StepWatchdog, TIME_START_I, TIME_END_I, LR_MIN_FRACTION, DiLoCoState
from lob.encode.encoding import Message_Tokenizer
from lob.train.sharding_utils import initialize_mesh, create_state_shardings




def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0
    best_test_last_order_loss = 100000000
    best_test_last_order_acc = -10000.0
    best_test_last_order_nll = 100000000
    best_test_all_orders_nll = 100000000

    #for parameter sweep: get args from wandb server
    is_main_process = getattr(args, 'process_index', 0) == 0
    if args is None:
        args = wandb.config
    else:
        if args.USE_WANDB and is_main_process:
            # Rank 0: online sync to wandb cloud
            slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
            wandb_name = f"j{slurm_job_id}" if slurm_job_id else None
            run = wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity, name=wandb_name)
        elif args.USE_WANDB:
            # Non-rank-0: local logging only, no duplicate cloud runs
            run = wandb.init(mode='offline')
        else:
            run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    ds = 'lobster-prediction'
    #create_dataset_fn =  Datasets[ds]

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    mask_fn=None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
         mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
         mask_fn = LOBSTER_Dataset.no_mask
    else:
        ValueError('Issue with mask function: logic for '+args.masking+' not implemented.')

    is_distributed = getattr(args, 'is_distributed', False)
    process_rank = getattr(args, 'process_index', 0)
    process_count = getattr(args, 'process_count', 1)
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)

    (lobster_dataset, trainloader, _, _, _,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=args.jax_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            micro_bsz=args.micro_bsz,
            num_devices=args.num_devices,
            tp_size=getattr(args, 'tp_size', 1),
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            book_ablation=getattr(args, 'book_ablation', 'real'),
            n_data_workers=args.n_data_workers,
            prefetch_factor=getattr(args, 'prefetch_factor', 2),
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            val_split=0.0,
            test_split=0.0,
            test_dir_name=getattr(args, 'test_dir_name', None),
            use_distributed_sampler=is_distributed,
            process_rank=process_rank,
            process_count=process_count,
            tickers=getattr(args, 'tickers', None),
            data_root=getattr(args, 'data_root', None),
            train_date_range=getattr(args, 'train_date_range', None),
            test_date_range=getattr(args, 'test_date_range', None),
            token_mode=getattr(args, 'token_mode', '24tok'),
        )

    print(f"[*] Starting S5 Training on {ds} =>> Initializing...")
    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
    else:
        state, model_cls = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=seq_len,
            book_dim=book_dim,
            book_seq_len=book_seq_len,
            train_size=train_size,
            print_shapes=True
        )

        # Initialize mesh first (needed for restore and JIT step functions)
        # Multi-node: global mesh over ALL devices for cross-node gradient sync
        # Single-node: local mesh over num_devices GPUs
        use_hierarchical = getattr(args, 'hierarchical', False)
        tp_size = getattr(args, 'tp_size', 1)
        if jax.process_count() > 1:
            mesh = initialize_mesh(jax.device_count(),
                                   hierarchical=use_hierarchical,
                                   tp_size=tp_size)
        else:
            if use_hierarchical and tp_size <= 1:
                print("[Sharding] Single-node: forcing hierarchical=False (1D mesh)")
            use_hierarchical = False if tp_size <= 1 else use_hierarchical
            mesh = initialize_mesh(args.num_devices, tp_size=tp_size)

        # DiLoCoState wrapping must match the saved checkpoint structure:
        #   - DiLoCo-era ckpts (post-988fb8b)  → wrap BEFORE restore
        #   - Pre-DiLoCo ckpts                 → wrap AFTER restore (fresh outer_momentum)
        # Probe the saved _METADATA to choose the right order.
        diloco_enabled = getattr(args, 'diloco_outer', 'none') == 'nesterov'
        restore_active = (args.restore is not None and args.restore != '')
        ckpt_has_outer = False
        if diloco_enabled and restore_active:
            try:
                _restore_step = args.restore_step
                if _restore_step is None:
                    _step_dirs = [d for d in os.listdir(args.restore) if d.isdigit()]
                    _restore_step = max(int(d) for d in _step_dirs) if _step_dirs else None
                if _restore_step is not None:
                    _md_path = os.path.join(args.restore, str(_restore_step),
                                             'state', '_METADATA')
                    if os.path.exists(_md_path):
                        with open(_md_path) as _f:
                            _md = json.load(_f)
                        ckpt_has_outer = any(
                            'outer_momentum' in k
                            for k in _md.get('tree_metadata', {}).keys())
                        print(f"[DiLoCo] Probed ckpt at step {_restore_step}: "
                              f"outer_momentum "
                              f"{'present' if ckpt_has_outer else 'absent'}")
            except Exception as _e:  # noqa: BLE001
                print(f"[DiLoCo] Could not probe ckpt structure ({_e}); "
                      "assuming pre-DiLoCo")

        if diloco_enabled and ckpt_has_outer:
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer "
                  f"(lr={getattr(args, 'diloco_outer_lr', 0.7)}, "
                  f"β={getattr(args, 'diloco_outer_momentum', 0.9)}). "
                  f"Wrapped BEFORE restore (matches DiLoCo-era ckpt).")
        elif diloco_enabled and not restore_active:
            # Fresh DiLoCo run, no restore — wrap now with zeros
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer (fresh start). "
                  f"Wrapped state in DiLoCoState with zero outer_momentum.")

        restored_metrics = {}
        if restore_active:
            print(f"[*] Restoring weights from {args.restore}")
            ckpt = load_checkpoint(
                state,
                args.restore,
                # args.__dict__,
                step=args.restore_step,
                mesh=mesh,
                partial_restore=getattr(args, 'partial_restore', False),
            )
            state = ckpt['model']
            # Debug: verify restored state
            print(f"[Restore] state.step = {int(state.step)}")
            print(f"[Restore] Restored metrics: {ckpt.get('metrics', {})}")
            # Check optimizer momentum is non-zero (proves Adam state restored)
            # Handle both plain multi_transform and chain(clip, multi_transform) structures
            _opt = state.opt_state
            if isinstance(_opt, tuple):
                _opt = _opt[-1]  # unwrap chain → last element is MultiTransformState
            ssm_inner = _opt.inner_states['ssm'].inner_state
            adam_state = ssm_inner[0]  # ScaleByAdamState (optax schedule mode)
            mu_leaves = jax.tree_util.tree_leaves(adam_state.mu)
            nu_leaves = jax.tree_util.tree_leaves(adam_state.nu)
            mu_norms = [float(jnp.linalg.norm(m)) for m in mu_leaves[:3]]
            nu_norms = [float(jnp.linalg.norm(n)) for n in nu_leaves[:3]]
            print(f"[Restore] Adam mu norms (first 3 params): {mu_norms}")
            print(f"[Restore] Adam nu norms (first 3 params): {nu_norms}")
            schedule_count = ssm_inner[1].count  # ScaleByScheduleState
            print(f"[Restore] Schedule count = {int(schedule_count)}")
            restored_metrics = ckpt.get('metrics', {})

            # --- Elastic Resume: remap step when device count or grad_accum changes ---
            ckpt_config = ckpt.get('config', {})
            original_process_count = ckpt_config.get('process_count', process_count)
            original_grad_accum = ckpt_config.get('grad_accum_steps', 1)
            if original_process_count != process_count or original_grad_accum != grad_accum_steps:
                # Compute original optimizer_steps_per_epoch (respecting curtail if checkpoint used it)
                original_curtail = ckpt_config.get('curtail_epochs', None)
                raw_original_micro_spe = train_size // (args.micro_bsz * args.num_devices * original_process_count)
                original_micro_spe = min(raw_original_micro_spe, original_curtail + 1) if original_curtail is not None else raw_original_micro_spe
                original_spe = original_micro_spe // max(original_grad_accum, 1)

                # Compute new optimizer_steps_per_epoch (respecting current curtail setting)
                raw_new_micro_spe = train_size // (args.micro_bsz * args.num_devices * process_count)
                new_micro_spe = min(raw_new_micro_spe, args.curtail_epochs + 1) if args.curtail_epochs is not None else raw_new_micro_spe
                new_spe = new_micro_spe // max(grad_accum_steps, 1)

                restored_epoch = int(state.step) // max(original_spe, 1)
                step_within_epoch = int(state.step) % max(original_spe, 1)
                scaled_step_within = round(step_within_epoch * new_spe / original_spe) if original_spe > 0 else 0
                remapped_step = restored_epoch * new_spe + scaled_step_within
                print(f"[Elastic Resume] process_count changed: {original_process_count} → {process_count}")
                print(f"[Elastic Resume] grad_accum changed: {original_grad_accum} → {grad_accum_steps}")
                print(f"[Elastic Resume] optimizer_steps/epoch: {original_spe} → {new_spe}")
                print(f"[Elastic Resume] intra-epoch: {step_within_epoch}/{original_spe} → {scaled_step_within}/{new_spe} ({step_within_epoch/max(original_spe,1)*100:.1f}%)")
                print(f"[Elastic Resume] state.step {int(state.step)} → {remapped_step} (epoch {restored_epoch})")
                state = remap_train_state_step(state, remapped_step)

            # --- Curriculum / Fine-tune: reset step to 0 so LR schedule starts fresh ---
            if getattr(args, 'restore_reset_schedule', False):
                print(f"[Restore Reset] state.step {int(state.step)} → 0 (fresh LR schedule)")
                state = remap_train_state_step(state, 0)

        # Pre-DiLoCo ckpt + DiLoCo enabled: wrap AFTER restore with fresh outer_momentum
        # (the saved state had no outer_momentum subtree, so wrapping before restore
        # would have produced a pytree-mismatch error).
        if diloco_enabled and not ckpt_has_outer and restore_active:
            _outer_m = jax.tree.map(jnp.zeros_like, state.params)
            state = DiLoCoState(train_state=state, outer_momentum=_outer_m)
            print(f"[DiLoCo] Enabled Nesterov outer optimizer "
                  f"(lr={getattr(args, 'diloco_outer_lr', 0.7)}, "
                  f"β={getattr(args, 'diloco_outer_momentum', 0.9)}). "
                  f"Wrapped AFTER restore with fresh zero outer_momentum "
                  f"(pre-DiLoCo ckpt).")

        val_model = model_cls(training=False, step_rescale=1)
        gdn_hd = getattr(args, 'gdn_head_dim', 128)
        gdn_nh = getattr(args, 'gdn_num_heads', None) or max(1, args.d_model // gdn_hd)
        gdn_hvd = gdn_hd * getattr(args, 'gdn_expand_v', 2)
        init_hidden = model_cls().initialize_carry(
            batch_size=args.micro_bsz,
            hidden_size=0,
            n_message_layers=args.n_message_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            n_fused_layers=args.n_layers,
            h_size_ema=args.d_model,
            num_heads=gdn_nh, head_dim=gdn_hd, head_v_dim=gdn_hvd,
            d_book=book_dim)

        # Move state to host numpy (device-agnostic) then shard to global mesh.
        # This handles both init (jax array on local device) and restore (numpy from checkpoint).
        state = jax.device_get(state)
        # Fix: ensure all processes have identical state before device_put.
        # numpy.linalg.eigh in make_DPLR_HiPPO (ssm_init.py:69) can produce
        # bit-level differences across nodes (LAPACK non-determinism).
        # device_put uses strict assert_equal, so broadcast rank 0's state.
        if jax.process_count() > 1:
            from jax.experimental.multihost_utils import broadcast_one_to_all
            state = broadcast_one_to_all(state)
        state_shardings = create_state_shardings(state, mesh)
        state = jax.device_put(state, state_shardings)
        total_devices = jax.device_count() if jax.process_count() > 1 else args.num_devices
        print(f"[*] State distributed via sharding (replicated across {total_devices} devices)")

        local_steps_k = getattr(args, 'local_steps_k', 0)
        if local_steps_k > 0:
            assert use_hierarchical or tp_size > 1, \
                "Local Steps (--local_steps_k>0) requires --hierarchical=True or --tp_size>1"
            print(f"[*] Local Steps enabled: sync params every {local_steps_k} steps "
                  f"(inner optimizer unchanged, only intra-node grad sync per step)")

        jit_train_step = create_jit_train_step(
            mesh, state, has_book_data=args.use_book_data,
            hierarchical=use_hierarchical,
            batchnorm=args.batchnorm, ignore_times=args.ignore_times,
            local_steps_k=local_steps_k,
            grad_accum_steps=grad_accum_steps,
            diloco_outer=getattr(args, 'diloco_outer', 'none'),
            diloco_outer_lr=getattr(args, 'diloco_outer_lr', 0.7),
            diloco_outer_momentum=getattr(args, 'diloco_outer_momentum', 0.9))
        jit_eval_step = create_jit_eval_step(mesh, state, has_book_data=args.use_book_data)

    # ── Compute FLOPs for throughput tracking ──
    try:
        from models.flops import compute_flops_per_step, print_flops_summary
        _num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        _tokens_per_step = args.micro_bsz * total_devices * seq_len
        # Mamba3 has a measured hardware-correction factor (~1.8-2.5x);
        # for other SSM types (gdn, s5, kda) leave correction at 1.0 to
        # avoid silently inflating MFU.
        _is_mamba3 = getattr(args, 'ssm_type', 'gdn') == 'mamba3'
        _flops_info = compute_flops_per_step(
            _num_params, _tokens_per_step,
            correction=None if _is_mamba3 else 1.0,
            d_model=getattr(args, 'd_model', 1024),
            seq_len=seq_len,
            chunk_size=getattr(args, 'mamba3_chunk_size', 64),
            d_state=getattr(args, 'mamba3_d_state', 128),
            headdim=getattr(args, 'mamba3_headdim', 64),
            expand=getattr(args, 'mamba3_expand', 2),
        )
        _flops_per_step = _flops_info["flops_per_step"]
        if is_main_process:
            print_flops_summary(_flops_info, num_gpus=total_devices)
            if args.USE_WANDB:
                wandb.config.update({
                    "flops_per_step": _flops_per_step,
                    "flops_correction": _flops_info["correction_factor"],
                    "num_params": _num_params,
                    "tokens_per_step": _tokens_per_step,
                }, allow_val_change=True)
    except Exception as e:
        print(f"[FLOPs] Warning: could not compute FLOPs: {e}")
        _flops_per_step = None

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = int(state.step)  # for per step learning rate decay (restored from checkpoint or 0)

    # Restore best metrics from checkpoint if available
    if restored_metrics:
        best_loss = restored_metrics.get('loss_val_ar', best_loss)
        best_acc = restored_metrics.get('acc_val_ar', best_acc)
        best_val_loss = restored_metrics.get('loss_val_ar', best_val_loss)
        best_test_loss = restored_metrics.get('loss_test_rnn', best_test_loss)
        best_test_acc = restored_metrics.get('acc_test_rnn', best_test_acc)
        print(f"[Restore] Best metrics restored: val_loss={best_loss:.5f}, val_acc={best_acc:.4f}, "
              f"test_loss={best_test_loss:.5f}, test_acc={best_test_acc:.4f}")
    micro_steps_per_epoch = int(train_size / (args.micro_bsz * args.num_devices * process_count)) if args.curtail_epochs is None else args.curtail_epochs+1
    # steps_per_epoch in optimizer updates (= micro_steps // K)
    steps_per_epoch = micro_steps_per_epoch // grad_accum_steps
    if grad_accum_steps > 1:
        print(f"[GradAccum] K={grad_accum_steps}: micro_steps/epoch={micro_steps_per_epoch}, "
              f"optimizer_steps/epoch={steps_per_epoch}, "
              f"effective_bsz={args.micro_bsz * args.num_devices * process_count * grad_accum_steps}")

    # Scaling-law training jobs are train-only. Held-out loss is computed later
    # by a separate fixed-eval job, so no validation/test callback is wired into
    # the training loop.
    train_only_validate_every_n_steps = 0

    # Create LR schedule functions for wandb logging (optax manages LR inside JIT)
    total_steps = steps_per_epoch * args.epochs
    # COSINE_STEPS override: decouple cosine period from dataset size.
    # When training << 1 epoch, set COSINE_STEPS to your actual training budget
    # so the LR decays meaningfully (e.g. COSINE_STEPS=200000 for ~200k steps).
    cosine_steps_override = int(os.environ.get('COSINE_STEPS', '0'))
    if cosine_steps_override > 0:
        total_steps = cosine_steps_override
        print(f"[Schedule] COSINE_STEPS override: cosine period = {total_steps} "
              f"(dataset epoch = {steps_per_epoch * args.epochs})")
    warmup_end_step = int(steps_per_epoch * args.warmup_end)

    effective_lr_min = args.lr_min if args.lr_min > 0 else lr * LR_MIN_FRACTION
    effective_ssm_lr_min = args.lr_min if args.lr_min > 0 else ssm_lr * LR_MIN_FRACTION

    lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_lr_min,
        use_cosine_anneal=args.cosine_anneal)
    ssm_lr_schedule_fn = create_lobs5_learning_rate_schedule(
        base_lr=ssm_lr, warmup_end_step=warmup_end_step,
        total_steps=total_steps, lr_min=effective_ssm_lr_min,
        use_cosine_anneal=args.cosine_anneal)

    # print("USING VERY INFREQUENT CHECKPOINTING FOR TINY EPOCH SIZE ")

    # Global mesh: ALL ranks must create CheckpointManager so Orbax barriers work.
    # Use SLURM_JOB_ID for consistent path across ranks (wandb run names differ per rank).
    # Orbax primary_host=0 ensures only rank 0 writes; others just participate in barriers.
    slurm_jid = os.environ.get("SLURM_JOB_ID", "local")
    ckpt_base = os.environ.get('CHECKPOINT_BASE_DIR', 'checkpoints')
    ckpt_dir = os.path.abspath(f'{ckpt_base}/{run.name}_{run.id}_{slurm_jid}/') if is_main_process else \
               os.path.abspath(f'{ckpt_base}/job_{slurm_jid}/')
    if process_count > 1:
        # Multi-node: broadcast rank 0's checkpoint dir to all ranks
        if is_main_process:
            # Encode path as fixed-length byte array
            path_bytes = ckpt_dir.encode('utf-8')
            path_arr = jnp.array(list(path_bytes) + [0] * (256 - len(path_bytes)), dtype=jnp.uint8)
        else:
            path_arr = jnp.zeros(256, dtype=jnp.uint8)
        path_arr = jax.make_array_from_process_local_data(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), path_arr
        )
        path_arr = jax.experimental.multihost_utils.broadcast_one_to_all(path_arr)
        path_bytes = bytes(jax.device_get(path_arr).tolist()).rstrip(b'\x00')
        ckpt_dir = path_bytes.decode('utf-8')
        print(f"[Rank {process_rank}] Checkpoint dir: {ckpt_dir}")

    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        create=True,
        max_to_keep=10,
        keep_period=5,
        # Disable async: forked subprocesses corrupt NCCL after cuInit in child
        enable_async_checkpointing=False,
    )
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_dir,
        item_names=('state', 'metadata'),
        options=mgr_options,
        metadata=vars(args) if is_main_process else {}
    )


    if args.ignore_times:
        # Removing the 5 abs time tokens from the length of the sequence.  
        dt = [[x] for (x,) in zip([*range(seq_len-5*args.msg_seq_len)])]
    else:
        dt = [[x] for (x,) in zip([*range(seq_len)])]
    ce_table=wandb.Table(columns=["tok"] ,data=dt)

    ignore_times=args.ignore_times
    batchnorm=args.batchnorm

    start_epoch = 0
    resume_from_step_auto = None
    if args.restore is not None and args.restore != '':
        # Always infer from state.step (works for both epoch-end and mid-epoch checkpoints)
        start_epoch = int(state.step) // max(steps_per_epoch, 1)
        # Auto-compute resume_from_step for mid-epoch checkpoints
        step_within_epoch = int(state.step) % max(steps_per_epoch, 1)
        if step_within_epoch > 0:
            resume_from_step_auto = step_within_epoch
            print(f"[Restore] Resuming from epoch {start_epoch}, batch_idx={resume_from_step_auto} "
                  f"(state.step={int(state.step)}, steps_per_epoch={steps_per_epoch})")
        else:
            print(f"[Restore] Resuming training from epoch {start_epoch} (of {args.epochs})")

    # Mid-epoch checkpoint: callback + resume state
    job_start_time = time.monotonic()
    # CLI arg overrides auto-computed value
    resume_from_step = getattr(args, 'resume_from_step', None) or resume_from_step_auto

    # Reshard via host roundtrip — avoids creating a new NCCL clique that can
    # deadlock on CXI (Slingshot) when memory registrations go stale.
    # Cherry-picked from K3 Mamba2 (dbb1e445).
    if is_distributed:
        from jax.experimental.multihost_utils import broadcast_one_to_all as _bcast

        def _reshard_for_ckpt(s):
            # device_get may produce bit-level FP differences across hosts
            # (Muon Newton-Schulz, NCCL reduction non-determinism, etc.).
            # Broadcast rank-0's copy so device_put's assert_equal passes.
            host_s = jax.device_get(s)
            host_s = _bcast(host_s)
            return jax.device_put(host_s, state_shardings)

        # On resume: force a dummy checkpoint save while Orbax CXI endpoints
        # are still warm from the restore.  Without this, the first real save
        # (even at 5 min) can hit stale CXI registrations → SIGABRT (RC:265).
        # Cherry-picked from K3 Mamba2 (0721c624).
        if args.restore is not None and args.restore != '':
            _warm_ckpt = {
                'model': _reshard_for_ckpt(state),
                'config': vars(args) if is_main_process else {},
                'metrics': {'loss_train': 0.0, 'epoch': 0, 'step_in_epoch': 0}
            }
            save_checkpoint(ckpt_mgr, _warm_ckpt, int(state.step))
            del _warm_ckpt
            gc.collect()
            if is_main_process:
                print("[*] CXI warm: dummy checkpoint save completed (freed)")

    def step_checkpoint_callback(cb_state, cb_epoch, cb_batch_idx, cb_loss, save_flag=True, **kwargs):
        """Mid-epoch: log to wandb and optionally save checkpoint."""
        global_step = int(cb_state.step)
        if is_main_process and args.USE_WANDB:
            log_dict = {
                "step_loss": float(cb_loss),
                "epoch": cb_epoch + 1,
                "step_in_epoch": cb_batch_idx + 1,
                "global_step": global_step,
            }
            # Add throughput metrics if available
            throughput = kwargs.get("throughput")
            if throughput:
                log_dict.update({
                    "throughput/step_time_s": throughput["step_time_s"],
                    "throughput/mfu_pct": throughput["mfu_pct"],
                    "throughput/tflops": throughput["achieved_tflops"],
                    "throughput/tokens_per_sec": throughput["tokens_per_sec"],
                })
            # Per-group gradient norms (LOG_GRAD_NORMS=1)
            grad_norms = kwargs.get("grad_norms")
            if grad_norms is not None:
                _gn_names = ['global', 'muon', 'ssm', 'regular', 'in_proj', 'out_proj']
                for _gn_name, _gn_val in zip(_gn_names, grad_norms):
                    log_dict[f"grad_norms/{_gn_name}"] = float(_gn_val)
                _max_gn = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
                if _max_gn > 0:
                    log_dict["grad_norms/clip_ratio"] = float(grad_norms[0]) / _max_gn
            wandb.log(log_dict, step=global_step)
        if save_flag:
            if is_distributed:
                ckpt_st = _reshard_for_ckpt(cb_state)
            else:
                ckpt_st = deduplicate_trainstate(cb_state)
            ckpt = {
                'model': ckpt_st,
                'config': vars(args) if is_main_process else {},
                'metrics': {
                    'loss_train': float(cb_loss),
                    'epoch': cb_epoch,
                    'step_in_epoch': cb_batch_idx,
                }
            }
            try:
                save_checkpoint(ckpt_mgr, ckpt, global_step)
                if is_main_process:
                    print(f"[Checkpoint] Mid-epoch save: epoch={cb_epoch}, "
                          f"step={cb_batch_idx}, global_step={global_step}")
            except (OSError, ValueError) as e:
                print(f"[Checkpoint] WARNING: mid-epoch save failed: {e}")

    for epoch in range(start_epoch, args.epochs):
        # Free residual memory before training.
        gc.collect()

        # Mid-epoch resume: rebuild trainloader with sampler-level skip
        # so DataLoader never calls __getitem__ for already-completed batches.
        if resume_from_step is not None and resume_from_step > 0:
            print(f"[Resume] Rebuilding trainloader with sampler skip: "
                  f"step={resume_from_step}, epoch={epoch}")
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
                resume_from_step=resume_from_step,
                resume_epoch=epoch,
            )
        else:
            # Update DistributedSampler epoch for proper cross-epoch shuffling
            if hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'set_epoch'):
                trainloader.sampler.set_epoch(epoch)

        print(f"[*] Starting Training Epoch {epoch + 1}...")
        print(f"[*] Step {step} - LR managed by optax schedules")
        print('Training on', args.num_devices, 'devices.')
        train_rng, skey = random.split(train_rng)

        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used.
        state, train_loss, ce_by_tok, interrupted_at_step = train_epoch(state,
                                              skey,
                                              trainloader,
                                              seq_len,
                                              batchnorm,
                                              None,  # lr_params=None → optax schedules
                                              args.num_devices,
                                              args.debug_loading,
                                              args.enable_profiler,
                                              args.curtail_epochs,
                                              init_hidden,
                                              epoch,
                                              ignore_times,
                                              args.log_ce_tables,
                                              mesh=mesh,
                                              jit_train_step_fn=jit_train_step,
                                              checkpoint_callback=step_checkpoint_callback,
                                              checkpoint_every_n_steps=getattr(args, 'checkpoint_every_n_steps', 'auto'),
                                              job_start_time=job_start_time,
                                              max_job_hours=getattr(args, 'max_job_hours', 24.0),
                                              save_before_timeout_minutes=getattr(args, 'save_before_timeout_minutes', 30),
                                              resume_from_step=resume_from_step,
                                              validate_callback=None,
                                              validate_every_n_steps=train_only_validate_every_n_steps,
                                              flops_per_step=_flops_per_step,
                                              num_gpus=total_devices,
                                              )
        # resume_from_step only applies to the first epoch after restore.
        # If we rebuilt the trainloader with sampler skip, restore the normal
        # loader for subsequent epochs (so DistributedSampler.set_epoch works).
        if resume_from_step is not None:
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                seed=args.jax_seed,
                per_process_bsz=args.micro_bsz * args.num_devices,
                num_workers=args.n_data_workers,
                reset_train_offsets=False,
                shuffle=args.shuffle_train,
                use_distributed_sampler=is_distributed,
                process_rank=process_rank,
                process_count=process_count,
            )
        resume_from_step = None
        step = int(state.step)

        # Handle timeout interrupt: skip validation + epoch-end checkpoint, exit
        if interrupted_at_step is not None:
            if is_main_process:
                print(f"[Train] Epoch {epoch+1} interrupted at step {interrupted_at_step} due to timeout")
                print(f"[Train] To resume: RESTORE_PATH={ckpt_dir} RESTORE_STEP={step} "
                      f"RESUME_FROM_STEP={interrupted_at_step}")
            break

        if args.random_offsets_train:
            # Refresh random offsets in-place without rebuilding DataLoader.
            # This keeps persistent workers alive across epochs.
            lobster_dataset.reset_train_offsets()

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(f"\tTrain Loss: {train_loss:.5f}")

        # Save checkpoint — ALL ranks must call save() for Orbax barrier sync.
        # Orbax primary_host=0 ensures only rank 0 writes to disk.
        # Multi-host: re-shard state to ensure consistent NamedSharding for Orbax.
        # Single-host: deduplicate to single device first.
        if is_distributed:
            ckpt_state = _reshard_for_ckpt(state)
        else:
            ckpt_state = deduplicate_trainstate(state)
        ckpt = {
            'model': ckpt_state,
            'config': vars(args) if is_main_process else {},
            'metrics': {
                'loss_train': float(train_loss),
            }
        }
        try:
            save_checkpoint(ckpt_mgr, ckpt, int(state.step))
            if is_main_process:
                print(f"[Checkpoint] Epoch-end save: epoch={epoch}, global_step={int(state.step)}")
        except (OSError, ValueError) as e:
            print(f"\n[FATAL] Checkpoint save failed at epoch {epoch}: {e}")
            print("[FATAL] Likely disk quota or serialization issue. Exiting.")
            if ckpt_mgr is not None:
                try:
                    ckpt_mgr.close()
                except Exception:
                    pass
            sys.exit(1)
        del ckpt
        del ckpt_state

        current_lr = float(lr_schedule_fn(step))
        current_ssm_lr = float(ssm_lr_schedule_fn(step))
        wandb.log(
            {
                "Training Loss": train_loss,
                "lr": current_lr,
                "ssm_lr": current_ssm_lr,
            }
        )
        wandb.run.summary["Final Training Loss"] = train_loss

        gc.collect()
        if is_distributed:
            sync_global_devices(f"post_epoch_{epoch}")
        continue

    # Wait for async checkpoint writes to complete before exiting
    if ckpt_mgr is not None:
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()
