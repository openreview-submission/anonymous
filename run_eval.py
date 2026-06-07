# >>> Read README.md first: repo structure + full model/training reference. <<<
import os


if __name__ == "__main__":
    pass
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # Forces all generated worker processes to not run on GPU.
    #  Required at this high level, because the init func in the 
    # worker spawn interface happens after init. of the CUDA process. 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
import wandb

from lob.train.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.preprocess.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader#, Datasets
from lob.preprocess.lobster_dataloader import LOBSTER_Dataset
from lob.train.train_helpers import reduce_lr_on_plateau, linear_warmup, \
    cosine_annealing, constant_lr, train_epoch, validate

from lob.train.init_train import load_metadata, load_args_from_checkpoint



def eval(eval_args):
    """
    Main function to evaluate a given checkpoint
    """

    args= load_metadata(eval_args.restore)
    for arg in vars(eval_args):
        print(arg)
        setattr(args,str(arg),getattr(eval_args, str(arg)))
    if args.USE_WANDB:
        # Make wandb config dictionary
        run = wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    else:
        run = wandb.init(mode='offline')

    ssm_size = args.ssm_size_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})


    # Set randomness...
    print("[*] Eval Setting Randomness...")
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
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            val_split=0.0,
            test_split=0.0,
        )

    jax.random.normal(jax.random.PRNGKey(0), (1,))  # init GPU
    print(jax.devices())
    

    print(f"[*] Starting S5 Eval on {ds} =>> Loading the states...")
    state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=seq_len,
        book_dim=book_dim,
        book_seq_len=book_seq_len,
        print_shapes=True
    )
    # print("State at init",jax.tree_util.tree_map(lambda x: x.shape,state))

    dt = [[x] for (x,) in zip([*range(seq_len)])]
    ce_table=wandb.Table(columns=["tok"] ,data=dt)


    for epoch in range(args.epochs):
        print(f"[*] Starting Val/Test of Epoch {args.restore_step + epoch + 1}...")

        print(f"[*] Restoring weights from {args.restore}")
        ckpt = load_checkpoint(
            state,
            args.restore,
            # args.__dict__,
            step=args.restore_step+epoch,
        )
        state = ckpt['model']
        eval_model = model_cls(training=False, step_rescale=1)
        # print("State at restore",jax.tree_util.tree_map(lambda x: x.shape,state))


        #Pass an initial hidden state to be used in case of the 'RNN' forward pass being used. 
        init_hidden=model_cls().initialize_carry(batch_size=args.micro_bsz,
                                                hidden_size=(ssm_size // pow(2,int(args.conj_sym))),
                                                n_message_layers=args.n_message_layers,
                                                n_book_pre_layers=args.n_book_pre_layers ,
                                                n_book_post_layers=args.n_book_post_layers,
                                                n_fused_layers=args.n_layers,
                                                h_size_ema=ssm_size)


        print(f"[*] Running Epoch {args.restore_step + epoch + 1} Validation on train set (With call)...")
        ar_loss, ar_acc,ar_ce_by_tok, ar_acc_by_tok = validate(state,
                                        #model_cls,
                                        eval_model.apply,
                                        trainloader,
                                        seq_len,
                                        in_dim,
                                        args.batchnorm,
                                        args.num_devices,
                                        epoch,
                                        curtail_epoch=args.curtail_epoch,
                                        ignore_times=args.ignore_times,
                                        apply_method='__call_ar__',
                                        log_ce_tables=True)

        print(f"[*] Running Epoch {args.restore_step + epoch + 1} Test on train set (With Scan RNN)...")
        rnn_loss, rnn_acc, rnn_ce_by_tok, rnn_acc_by_tok  = validate(state,
                                        #model_cls,
                                        eval_model.apply,
                                        trainloader,
                                        seq_len,
                                        in_dim,
                                        args.batchnorm,
                                        args.num_devices,
                                        epoch,
                                        curtail_epoch=args.curtail_epoch,
                                        ignore_times=args.ignore_times,
                                        apply_method='__call_rnn__',
                                        init_hiddens=init_hidden,
                                        log_ce_tables=True)

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f" Autoreg Loss: {ar_loss:.5f} --RNN Loss: {rnn_loss:.5f} --"
            f" Autoreg Accuracy: {ar_acc:.4f} -- RNN Accuracy: {rnn_acc:.4f}"
        )




        # ce_table.add_column(name="val_ce_"+str(epoch),data=val_ce_by_tok.tolist())
        ce_table.add_column(name="ar_ce_"+str(epoch),data=ar_ce_by_tok.tolist())
        # ce_table.add_column(name="val_acc_"+str(epoch),data=val_acc_by_tok.tolist())
        ce_table.add_column(name="ar_acc_"+str(epoch),data=ar_acc_by_tok.tolist())
        ce_table=wandb.Table(columns=ce_table.columns,data=ce_table.data)
        

        wandb.log(
            {
                "'Call' loss": ar_loss,
                "'Call' Accuracy": ar_acc,
                "'Call_Rnn' Loss": rnn_loss,
                "'Call_Rnn' Accuracy": rnn_acc,
                "Training CE by token":ce_table
            }
        )








if __name__ == "__main__":
    import argparse
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.85"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    from models.utils.util import str2bool


    parser = argparse.ArgumentParser()

    parser.add_argument("--restore", type=str, default=None,
                            help="Path to file of checkpoint")
    parser.add_argument("--curtail_epoch", type=int, default=None,
                    help="End epoch early at this step (Train & Val/Test)")
    parser.add_argument("--restore_step", type=int, default=None,
                help="How many epochs to eval from the restore")  
    parser.add_argument("--epochs", type= int, default=1,
                        help="How many epochs to run from the restore step.")
    parser.add_argument("--mask_time_losses", type=str2bool, default=False,
            help="Ignore the loss due to the tokens related to the exact time.")
    parser.add_argument("--dir_name", type=str, default=None,
                            help="Path to data")
    parser.add_argument("--wandb_project", type=str, default="LOBS5-Eval",
                    help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="anonymous",
                    help="wandb entity name, e.g. username")
    parser.add_argument("--n_data_workers", type=int, default=0,
                    help="number of workers used in DataLoader")
    parser.add_argument("--micro_bsz", type=int, default=16,
                    help="per-GPU (micro) batch size")
    parser.add_argument("--num_devices", type=int, default=1,
                    help="number of devices (GPUs) to use")
    parser.add_argument("--USE_WANDB", type=str2bool, default=True,
                    help="log with wandb?")
    parser.add_argument("--ignore_times", type=str2bool, default=True,
                    help="Ignore the loss due to predicting the time.")


    args = parser.parse_args()


    import torch
    torch.multiprocessing.set_start_method('spawn')

    eval(args)