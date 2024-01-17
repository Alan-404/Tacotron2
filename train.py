import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from src.tacotron2 import Tacotron2
from processor import Tacotron2Processor
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping
import torchsummary
from dataset import Tacotron2Dataset
from src.utils.masking import generate_mask
import wandb
from torch.cuda.amp import GradScaler, autocast

from argparse import ArgumentParser
parser = ArgumentParser()

# Text Processor Config
parser.add_argument("--phoneme_path", type=str, default="./phoneme.json")
parser.add_argument("--padding_token", type=str, default="<pad>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--bos_token", type=str, default="<s>")
parser.add_argument("--eos_token", type=str, default="</s>")
parser.add_argument("--space_token", type=str, default="|")

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sample_rate", type=int, default=22050)
parser.add_argument("--fft_size", type=int, default=1024)
parser.add_argument("--hop_length", type=int, default=256)
parser.add_argument("--win_size", type=int, default=1024)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)
parser.add_argument("--htk", type=bool, default=True)

# Checkpoint Config
parser.add_argument("--saved_checkpoint", type=str, default="./checkpoints")
parser.add_argument("--checkpoint", type=str, default=None)

# Training Config
parser.add_argument("--train_path", type=str, default="./datasets/train.csv")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_train", type=int, default=None)

parser.add_argument("--use_validation", type=bool, default=False)
parser.add_argument("--val_path", type=str, default=None)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--num_val", type=int, default=None)

parser.add_argument("--num_epochs", type=int, default=1)

parser.add_argument("--set_lr", type=bool, default=False)
parser.add_argument("--lr", type=float, default=3e-4)

# Model Config
parser.add_argument("--embedding_dim", type=int, default=512)
parser.add_argument("--encoder_kernel_size", type=int, default=5)
parser.add_argument("--attention_dim", type=int, default=128)
parser.add_argument("--attention_rnn_dim", type=int, default=1024)
parser.add_argument("--decoder_rnn_dim", type=int, default=1024)
parser.add_argument("--n_filters", type=int, default=32)
parser.add_argument("--prenet_dim", type=int, default=256)
parser.add_argument("--location_kernel_size", type=int, default=5)
parser.add_argument("--encoder_n_convolutions", type=int, default=3)
parser.add_argument("--postnet_n_convolutions", type=int, default=5)
parser.add_argument("--postnet_kernel_size", type=int, default=5)
parser.add_argument("--dropout_rate", type=int, default=0.1)

parser.add_argument("--early_stopping_patience", type=int, default=5)

parser.add_argument("--wandb_project_name", type=str, default="(TTS) Tacotron2")
parser.add_argument("--wandb_name", type=str, default="trind18")

args = parser.parse_args()

wandb.init(project=args.wandb_project_name, name=args.wandb_name)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

processor = Tacotron2Processor(
    vocab_path=args.phoneme_path,
    sampling_rate=args.sample_rate,
    num_mels=args.num_mels,
    n_fft=args.fft_size,
    hop_length=args.hop_length,
    win_length=args.win_size,
    fmin=args.fmin,
    fmax=args.fmax,
    htk=args.htk,
    bos_token=args.bos_token,
    eos_token=args.eos_token,
    word_delim_token=args.space_token,
    pad_token=args.padding_token,
    unk_token=args.unk_token
)

model = Tacotron2(
    token_size=len(processor.dictionary),
    n_mel_channels=processor.n_mel_channels,
    embedding_dim = args.embedding_dim,
    encoder_kernel_size = args.encoder_kernel_size,
    attention_dim = args.attention_dim,
    attention_rnn_dim = args.attention_rnn_dim,
    decoder_rnn_dim = args.decoder_rnn_dim,
    n_filters = args.n_filters,
    prenet_dim = args.prenet_dim,
    location_kernel_size = args.location_kernel_size,
    encoder_n_convolutions = args.encoder_n_convolutions,
    postnet_n_convolutions = args.postnet_n_convolutions,
    postnet_kernel_size = args.postnet_kernel_size,
    dropout_rate = args.dropout_rate,
).to(device)

scaler = GradScaler()

def get_batch(batch: list):
    graphemes, signals = zip(*batch)

    tokens, token_lengths = processor(graphemes)

    mels, mel_lengths = processor.mel_spectrogize(signals, return_length=True)

    return tokens, mels, token_lengths, mel_lengths

optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)

assert os.path.exists(args.train_path), f"NOT FOUND TRAIN DATASET AT {args.train_path}"
train_dataset = Tacotron2Dataset(args.train_path, processor=processor, num_examples=args.num_train)

if args.use_validation:
    if args.val_path is not None:
        assert os.path.exists(args.val_path) == False, f"NOT FOUND VALIDATION AT {args.val_path}"
        val_dataset = Tacotron2Dataset(args.val_path, processor=processor, num_examples=args.num_val)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=True, collate_fn=get_batch)
    else:
        assert args.val_size < 0.5
        train_dataset, val_dataset = random_split(train_dataset, lengths=[1 - args.val_size, args.val_size], generator=torch.Generator().manual_seed(41))

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=True, collate_fn=get_batch)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=get_batch)

# Setup Functions

def loss_function(postnet_mel_out: torch.Tensor, mel_out: torch.Tensor, mel_target: torch.Tensor, gate_out: torch.Tensor, gate_target: torch.Tensor):
    mel_target.requires_grad = False
    gate_target.requires_grad = False

    mel_mask = (gate_target == False).unsqueeze(1)
        
    mel_out = mel_out.masked_fill(mel_mask, -100)
    postnet_mel_out = postnet_mel_out.masked_fill(mel_mask, -100)
    mel_target = mel_target.masked_fill(mel_mask, -100)
    gate_target = gate_target.type(torch.float32)
        
    loss = F.mse_loss(mel_out, mel_target) + F.mse_loss(postnet_mel_out, mel_target) + F.binary_cross_entropy_with_logits(gate_out, gate_target)
    
    return loss

def train_step(engine: Engine, batch: list):
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
                
    token_lengths = batch[2].to(device)
    mel_lengths = batch[3].to(device)
    gate = generate_mask(mel_lengths)

    optimizer.zero_grad()

    with autocast():
        postnet_mel_outputs, mel_outputs, gate_outputs = model(inputs, labels, token_lengths)
        loss = loss_function(postnet_mel_outputs, mel_outputs, labels, gate_outputs, gate)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)

    scaler.step()

    return loss.item()

def val_step(engine: Engine, batch: list):
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
                
    lengths = batch[2].to(device)
    mask = batch[3].to(device)
    gate = batch[4].to(device)

    with torch.no_grad():
        postnet_mel_outputs, mel_outputs, gate_outputs = model(inputs, labels, lengths, mask)
    
    loss = loss_function(postnet_mel_outputs, mel_outputs, labels, gate_outputs, gate)
    
    return loss.item()

def early_stopping_condition(engine: Engine):
    return -engine.state.metrics['loss']

# Setup Trainer
trainer = Engine(train_step)
ProgressBar().attach(trainer)
train_loss = RunningAverage(output_transform=lambda x: x)
train_loss.attach(trainer, 'loss')

validator = Engine(val_step)
val_loss = RunningAverage(output_transform=lambda x: x)
val_loss.attach(validator, 'loss')
ProgressBar().attach(validator)

early_stopping_handler = EarlyStopping(patience=args.early_stopping_patience, score_function=early_stopping_condition, trainer=trainer)

# Checkpoint Setup
to_save = {
    'model': model,
    'optimizer': optimizer,
    'lr_scheduler': scheduler,
    'scaler': scaler
}

checkpoint_manager = Checkpoint(to_save=to_save ,save_handler=DiskSaver(args.saved_checkpoint, create_dir=True, require_empty=False), n_saved=args.early_stopping_patience)

# Trainer Events
@trainer.on(Events.STARTED)
def start_training(engine: Engine):
    torchsummary.summary(model)

    if args.set_lr:
        optimizer.param_groups[0]['lr'] = args.lr

    print("\n")
    print("================== Training Information ==================")
    print(f"\tNumber of Samples: {len(engine.state.dataloader.dataset)}")
    print(f"\tBatch Size: {engine.state.dataloader.batch_size}")
    print(f"\tNumber of Batches: {len(engine.state.dataloader)}")
    print(f"\tCurrent Learning Rate: {optimizer.param_groups[0]['lr']}")
    print("==========================================================\n")

    if args.use_validation:
        print("================== Validation Information ==================")
        print(f"\tNumber of Samples: {len(val_dataset)}")
        print(f"\tBatch Size: {args.val_batch_size}")
        print(f"\tNumber of Batches: {len(val_dataloader)}")
        print("==========================================================\n")

    model.train()

@trainer.on(Events.EPOCH_STARTED)
def start_epoch(engine):
    print(f"====== Epoch {engine.state.epoch} =========")

@trainer.on(Events.EPOCH_COMPLETED)
def finish_epoch(engine: Engine):
    print(f"Training Loss: {engine.state.metrics['loss']}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    wandb.log({
        'train_loss': engine.state.metrics['loss'],
        'learning_rate': optimizer.param_groups[0]['lr']
    }, step=engine.state.epoch)
    scheduler.step()
    train_loss.reset()
    if args.use_validation:
        validator.run(data=val_dataloader, max_epochs=1)
    print(f"====== Done Epoch {engine.state.epoch} =========\n")

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_manager)

# Validator Events
@validator.on(Events.EPOCH_COMPLETED)
def end_validating(engine: Engine):
    print(f"Validation Loss: {(engine.state.metrics['loss']):.4f}")
    val_loss.reset()
    wandb.log({
        'val_loss': engine.state.metrics['loss']
    }, step=trainer.state.epoch)

# Early Stopping
if args.use_validation == True:
    validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

# Load Checkpoint 
if args.checkpoint is not None:
    if os.path.exists(args.checkpoint) == False:
        print(f"NOT FOUND CHECKPOINT {args.checkpoint}")
    else:
        print(f"LOADED CHECKPOINT: {args.checkpoint}")
        Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(args.checkpoint, map_location=device))

# Start Training
trainer.run(data=train_dataloader, max_epochs=args.num_epochs)