# >>> Read README.md first: repo structure + full model/training reference. <<<
import argparse
import os
import sys
sys.path.append("/path/to/LOBS5/")
sys.path.append("/path/to/AlphaTrade/")
# # Add parent folder to path (to run this file from subdirectories)
# (parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
# sys.path.append(parent_folder_path)

# # add git submodule to path to allow imports to work
# submodule_name = 'AlphaTrade'
# sys.path.append(os.path.join(parent_folder_path, submodule_name))

# from gymnax_exchange.jaxob.jorderbook import OrderBook
# from AlphaTrade import gymnax_exchange
# from AlphaTrade.gymnax_exchange.jaxob.jorderbook import OrderBook

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"

import torch
torch.multiprocessing.set_start_method('spawn')



# Add parent folder to path (to run this file from subdirectories)
# (parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
# sys.path.append(parent_folder_path)

# # add git submodule to path to allow imports to work
# submodule_name = 'AlphaTrade'
# sys.path.append(os.path.join(parent_folder_path, submodule_name))

import jax
from lob.encode.encoding import Vocab, Message_Tokenizer

from lob.infer import inference_no_errcorr as inference
from lob.train.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint

print(os.path.abspath(''))


#############################################################################
parser = argparse.ArgumentParser()

parser.add_argument(
    "--stock", type=str)
parser.add_argument(
    # "--save_folder", type=str, default='/path/to/LOBS5/data_saved/m500/')
    "--save_folder", type=str, default='/path/to/LOBS5/GOOG_tmp/')
parser.add_argument(
    "--n_gen_msgs", type=int,
	help="how many messages to generate following each input sequence")
parser.add_argument(
    "--n_samples", type=int,
	help="how many messages sequences to generate")
parser.add_argument(
    "--batch_size", type=int, default=16,
	help="how many sequences to generate in parallel (vmap)")
parser.add_argument(
    "--model_size", type=str, default='large',)
parser.add_argument(
    "--data_dir", type=str, default='/path/to/LOBS5/GOOG_tiny/',)
    # "--data_dir", type=str, default='/path/to/LOBS5/data/test_set/',)

args = parser.parse_args()

#############################################################################

n_messages = 500  # length of input sequence to model
n_gen_msgs = args.n_gen_msgs # how many messages to generate into the future
n_samples = args.n_samples
batch_size = args.batch_size

v = Vocab()
n_classes = len(v)
seq_len = n_messages * Message_Tokenizer.MSG_LEN
book_dim = 501 #b_enc.shape[1]
book_seq_len = n_messages

n_eval_messages = args.n_gen_msgs  # how many to load from dataset 
eval_seq_len = n_eval_messages * Message_Tokenizer.MSG_LEN

rng = jax.random.key(42)
rng, rng_ = jax.random.split(rng)

stock = args.stock # 'GOOG', 'INTC'

if stock == 'GOOG':
    # ckpt_path = './checkpoints/treasured-leaf-149_84yhvzjt/' # 0.5 y GOOG, (full model)
    # ckpt_path = './checkpoints/denim-elevator-754_czg1ss71/' # large model
    ckpt_path = '/path/to/LOBS5/checkpoints/honest-oath-159_3kn3xbd5/' # large model
elif stock == 'INTC':
    # ckpt_path = './checkpoints/pleasant-cherry-152_i6h5n74c/' # 0.5 y INTC, (full model)
    # TODO:
    ckpt_path = '.'
else:
    raise ValueError(f'stock {stock} not recognized')

print('Loading metadata:', ckpt_path)
args_ckpt = load_metadata(ckpt_path)

# scale down to single GPU, single sample inference
args_ckpt.bsz = 1 #1, 10
args_ckpt.num_devices = 1

batchnorm = args_ckpt.batchnorm

# load train state from disk

print('Initializing model...')
new_train_state, model_cls = init_train_state(
    args_ckpt,
    n_classes=n_classes,
    seq_len=seq_len,
    book_dim=book_dim,
    book_seq_len=book_seq_len,
)

print('Loading model checkpoint...')
ckpt = load_checkpoint(
    new_train_state,
    ckpt_path,
    train=False,
)
state = ckpt['model']

model = model_cls(training=False, step_rescale=1.0)


# entire test set after training data

data_dir = args.data_dir + stock
# if stock == 'GOOG':
#     data_dir = args.data_dir + '/GOOG/'
# elif stock == 'INTC':
    # data_dir = '/path/to/LOBS5/data/test_set/INTC/'

ds = inference.get_dataset(data_dir, n_messages, n_eval_messages)

# rng, rng_ = jax.random.split(rng)

print('Generating...')
# m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = inference.sample_new(
# saves data to disk
inference.sample_new(
    n_samples,
    batch_size,
    ds,
    rng,
    seq_len,
    n_messages,
    n_gen_msgs,
    state,
    model,
    batchnorm,
    v.ENCODING,
    stock,
    save_folder=args.save_folder + '/' + stock + '/',
)
print('DONE.')
