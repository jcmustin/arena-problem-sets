# %% (setup)
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import webbrowser
# (removed PyTorch Lightning from exercises)
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part1_transformers_from_scratch', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("mps" if t.backends.mps.is_available() else ("cuda" if t.cuda.is_available() else "cpu"))

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# %% === CLEAN TRANSFORMER IMPLEMENTATION ===
# %% (sorted vocab exploration)
sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()

lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")


print(sorted_vocab[-20:])


# %% (convert text to tokens)
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))

# %% (map tokens to logits)
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)
print(list(zip(reference_gpt2.to_str_tokens(tokens)[1:], reference_gpt2.to_str_tokens(logits.argmax(dim=-1))[:-1])))


# %% (softmax)
probs = logits.softmax(dim=-1)
print(probs.shape)
print(probs)


# %% (Config)
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# %% (tests)
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")
# %% LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # note: unbiased is about sample variance, I think
        var = t.var(residual, dim=-1, unbiased=True, keepdim=True)
        mean = t.mean(residual, dim=-1, keepdim=True)
        normalized = (residual - mean) / (var + self.cfg.layer_norm_eps).sqrt()
        return normalized * self.w + self.b


rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])


# %% Embedding
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)
# %% Positional Embedding
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)


# %% Attention
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        queries = self.b_Q + einops.einsum(normalized_resid_pre, self.W_Q, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head")
        keys = self.b_K + einops.einsum(normalized_resid_pre, self.W_K, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head")
        values = self.b_V + einops.einsum(normalized_resid_pre, self.W_V, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head")
        attn_scores = einops.einsum(queries, keys, "batch q_pos n_heads d_head, batch k_pos n_heads d_head -> batch n_heads q_pos k_pos")
        attn_scores = attn_scores/math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_scores = attn_scores.softmax(dim=-1)
        weighted_values = einops.einsum(attn_scores, values, "batch n_heads q_pos k_pos, batch k_pos n_heads d_head -> batch q_pos n_heads d_head")
        out = einops.einsum(weighted_values, self.W_O, "batch q_pos n_heads d_head, n_heads d_head d_model -> batch q_pos d_model")
        return self.b_O + out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        all_ones = t.ones(*attn_scores.shape[-2:], device=attn_scores.device)
        mask = all_ones.triu(diagonal=1).bool()
        return attn_scores.masked_fill(mask, self.IGNORE)


rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])


# %% MLP
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        fc1_out = self.b_in + einops.einsum(
            normalized_resid_mid,
            self.W_in,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        )
        nonlinear_out = gelu_new(fc1_out)
        return self.b_out + einops.einsum(
            nonlinear_out,
            self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        )


rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])


# %% TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid_mid = resid_pre + self.attn(self.ln1(resid_pre))
        return resid_mid + self.mlp(self.ln2(resid_mid))


rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])


# %% Unembed
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return self.b_U + einops.einsum(
            normalized_resid_final,
            self.W_U,
            "batch position d_model, d_model d_vocab -> batch position d_vocab"
        )


rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])


# %% Full Transformer
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for module in self.blocks:
            resid = module(resid)
        return self.unembed(self.ln_final(resid))


rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
# %% (get_log_probs)
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], 
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens


# %% === TRAINING A TRANSFORMER ===
# %% (create model)
model_cfg = Config(
    debug=False, 
    d_model=256, 
    n_heads=4, 
    d_head=64, 
    d_mlp=1024, 
    n_layers=2, 
    n_ctx=256, 
    d_vocab=reference_gpt2.cfg.d_vocab
)
model = DemoTransformer(model_cfg)


# %% (TransformerTrainingArgs)
dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2
    wandb_project: Optional[str] = "day1-demotransformer"
    wandb_name: Optional[str] = None


args = TransformerTrainingArgs()


# %% (create data)
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
print(dataset)
print(dataset[0]['text'][:100])

tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model.cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

first_batch = train_loader.dataset[:args.batch_size]

print(first_batch.keys())
print(first_batch['tokens'].shape)


# %% wandb init
wandb.login()
# %% TransformerTrainer
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandb.watch(self.model)


    def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        tokens = batch['tokens'].to(device)
        predictions = self.model(tokens)

        # tokens_one_hot = F.one_hot(tokens, num_classes=self.model.cfg.d_vocab).float()
        # # note: I think cross entropy is the correct choice rather than nll_loss,
        # # since Unembed in DemoTransformer doesn't run softmax on the output.
        # loss = F.cross_entropy(predictions[:, :-1, :], tokens_one_hot[:, 1:, :])

        # edit: I must've mis-implemented the commented code above,
        # because I'm getting different results than get_log_probs. Not yet sure why.

        loss = -get_log_probs(predictions, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        wandb.log({'loss': loss.item()}, step=self.step)
        return loss

    @t.inference_mode()
    def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
        the whole validation set).
        '''
        tokens = batch['tokens'].to(device)
        predictions = self.model(tokens)[:, :-1].argmax(dim=-1)
        return (predictions==tokens[:, 1:]).reshape(-1).float()

    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        progress_bar = tqdm(total = self.args.max_steps_per_epoch * self.args.epochs)

        accuracy = 0

        for epoch in range(self.args.epochs):
            for step, batch in enumerate(self.train_loader()):
                if step > self.args.max_steps_per_epoch:
                    break
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")
                self.step += 1
            accuracy = t.cat([self.validation_step(batch) for batch in self.test_loader()]).mean().item()
            wandb.log({'accuracy': accuracy}, step=self.step)

    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


# %% (train)
model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()


# %% generating text via argmax
text = "I'm going to take over the world"
tokens = reference_gpt2.to_tokens(text).to(device)
logits = model(tokens).argmax(dim=-1)
print(list(zip(reference_gpt2.to_str_tokens(tokens)[1:], reference_gpt2.to_str_tokens(logits)[:-1])))
# %% === SAMPLING FROM A TRANSFORMER ===
# %% TransformerSampler (contains answers for all remaining problems in 1.1.4)
model_cfg = Config()
model = DemoTransformer(model_cfg).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)

tokenizer = reference_gpt2.tokenizer


class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how 
        new tokens are chosen.
        '''
        self.model.eval()

        # note: there's this dance you have to do in this function between storing
        # the list of tokens as a 1d thing vs a 2d with len(dim0)=1 sort of thing. 
        # sample_next_tokens expects a 1d index; self.model expects a batch dimension.
        # Squeezing (or t[0]) and unsqueezing (or t[None], or wrapping in []) accomplishes this.

        # `return_tensors="pt"` returns pytorch-type tensors
        running_tokens: Float[Tensor, "n_tokens"] = self.tokenizer.encode(prompt, return_tensors="pt").to(device).squeeze(0)
        for _ in range(max_tokens_generated):
            # -self.cfg.n_ctx ensures you don't overflow the model's max context size.
            # implicit choice here is to take the last n tokens, which makes sense given
            # that we're extending the text.
            next_logits: Float[Tensor] = self.model(running_tokens[None,-self.cfg.n_ctx:])[0, -1]
            next_token = t.tensor([self.sample_next_token(running_tokens, next_logits, **kwargs)], device=device)
            running_tokens = t.cat([running_tokens, next_token], dim=-1)
            if verbose:
                print(self.tokenizer.decode(running_tokens))
            if next_token == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(running_tokens)

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str, 
        num_return_sequences: int, 
        num_beams: int, 
        max_new_tokens: int, 
        no_repeat_ngram_size: int = 0,
        verbose=False
    ) -> List[Tuple[float, t.Tensor]]:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how 
        new tokens are chosen.
        '''
        pass

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"], 
        logits: Float[Tensor, "seq_len d_vocab"], 
        temperature=1.0, 
        top_k=0, 
        top_p=0.0, 
        frequency_penalty=0.0,
        seed=None
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Returns the most likely token (as an int).
        '''
        out = logits.argmax().item()
        return out

    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies temperature scaling to the logits.
        '''
        assert temperature > 0
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies a frequency penalty to the logits.
        '''
        (d_vocab,) = logits.shape
        return logits - freq_penalty * t.bincount(input_ids, minlength=d_vocab)

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Samples from the distribution defined by the logits.
        '''
        distribution = t.distributions.categorical.Categorical
        return distribution(logits.softmax(dim=-1)).sample().item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        '''
        Samples from the top k most likely tokens.
        '''
        top_logits, top_idx = t.topk(logits, k)
        idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
        return top_idx[..., idx].item()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        '''
        Samples from the most likely tokens which make up at least p cumulative probability.
        '''
        probs = logits.softmax(dim=-1)
        sorted_probs, sorted_indices = t.sort(probs, dim=-1, descending=True, stable=True)
        cumulative_p = t.cumsum(sorted_probs, dim=-1)
        n_take = max((cumulative_p < top_p).sum().item()+1, min_tokens_to_keep)
        index = t.distributions.categorical.Categorical(sorted_probs[:n_take]).sample()
        return sorted_indices[index].item()


# %% (sample)
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Greedy decoding with prompt: {prompt!r}\n")

output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output!r}\n")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
# %% (sample_basic testing)
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097
}
frequency_of_top_5 = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
    print(token)
    frequency_of_top_5[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word]
    observed_freq = frequency_of_top_5[word] / N
    print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."

print("Tests passed!")


# %% (apply_temperature testing)
logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")


# %% (apply_frequency_penalty testing)
bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt")
logits = t.ones(tokenizer.vocab_size)
penalized_logits = TransformerSampler.apply_frequency_penalty(input_ids.squeeze(), logits, 2.0)

assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space, 1-2*6=-11"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space, 1-2*3=-5"

print("Tests passed!")
# %% (manual testing)
sampler = TransformerSampler(model, tokenizer)

N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(frequency_penalty=100.0)),
    ("Negative freq penalty", dict(frequency_penalty=-3.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]

table = Table("Name", "Kwargs", "Output", title="Sampling - Manual Testing")

for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sampler.sample(your_prompt, max_tokens_generated=24, **kwargs)
        table.add_row(name, repr(kwargs), repr(output) + "\n")

rprint(table)


# %% (sample_top_k testing)
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097
}
topk_5_sum = sum(expected_top_5.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_k=5)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word] / topk_5_sum
    observed_freq = observed_freqs[word] / N
    print(f"Word: {word!r:<9}. Expected freq = {expected_freq:.4f}, observed freq = {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.015, "Try increasing N if this fails by a small amount."


# %% (further top-k testing)
sampler = TransformerSampler(model, tokenizer)

your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
output = sampler.sample(your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
rprint(f"Your model said:\n\n[bold dark_orange]{output}")


# %% (top_p testing)
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
    " church": 0.0648,
    " house": 0.0367, # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
    expected_freq = expected_top_10pct[word] / top_10pct_sum
    observed_freq = observed_freqs[word] / N
    print(f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}")
    assert abs(observed_freq - expected_freq) < 0.01, "Try increasing N if this fails by a small amount."


# %% beam_search
@dataclass
class Beams:
    '''Class to store beams during beam search.'''
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    # note: `batch` here is num_beams, pretty sure.
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        '''Creates a new Beams object with the same model and tokenizer.'''
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> "Beams":
        '''Allows you to take a slice of the beams object along the batch dimension.'''
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        '''Returns self as a list of logprob sums and completions (useful for getting final output).'''
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None) -> "Beams":
        '''
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        '''
        continuation_logits = self.model(self.tokens)[..., -1, :]
        top_logits, top_tokens = t.topk(continuation_logits, toks_per_beam)
        new_logprob_sums = einops.rearrange(top_logits.log_softmax(dim=-1), "batch k -> (batch k)") + einops.repeat(self.logprob_sums, "batch -> (batch k)", k=toks_per_beam)
        print(self.logprob_sums, new_logprob_sums)
        new_tokens = t.cat([
            einops.repeat(self.tokens, "batch seq -> (batch k) seq", k=toks_per_beam),
            einops.repeat(top_tokens, "batch k -> (batch k) 1")
        ], dim=-1)
        print(new_logprob_sums, new_tokens)
        return self.new_beams(new_logprob_sums, new_tokens)


    def filter(self, num_beams: int) -> Tuple["Beams", "Beams"]:
        '''
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing all best `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        '''
        pass

    def print(self, title="Best completions", max_print_chars=80) -> None:
        '''
        Prints out a set of sequences with their corresponding logitsums.
        '''
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str, 
    num_return_sequences: int, 
    num_beams: int, 
    max_new_tokens: int, 
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False
) -> List[Tuple[float, Tensor]]:
    '''
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    '''

    assert num_return_sequences <= num_beams
    self.model.eval()

    pass


# %% (Beams initializing)
beams = Beams(
    model, 
    tokenizer,
    logprob_sums = t.tensor([-10.0, -15.0, -20.0]).to(device),
    tokens = t.tensor([
        [5661, 318, 262, 2368],
        [5661, 318, 262, 1218],
        [5661, 318, 262, 717],
    ]).to(device)
)

beams.print()


# %% (Beams.generate testing)
print("Testing generate, without no_repeat_ngram_size argument:")
new_beams = beams.generate(toks_per_beam=2)
new_beams.print()
assert new_beams.logprobs_and_completions[0][1] == "this is the third time"

print("Testing generate, with no_repeat_ngram_size argument:")
bigram_beams = Beams(
    model, 
    tokenizer,
    logprob_sums = t.tensor([-0.0]).to(device),
    tokens = t.tensor([[530, 734, 530, 734]]).to(device)
    # tokens are " one two one two"
)

# With no_repeat_ngram_size=1, should not generate the token " one" or " two"
new_bigram_beams = bigram_beams.generate(toks_per_beam=3, no_repeat_ngram_size=1)
new_bigram_beams.print()
assert all([not (completion[1].endswith(" one") or completion[1].endswith(" two")) for completion in new_bigram_beams.logprobs_and_completions])

# With no_repeat_ngram_size=2, it can generate " two" (which it should), but not " one"
new_bigram_beams = bigram_beams.generate(toks_per_beam=3, no_repeat_ngram_size=2)
new_bigram_beams.print()
assert all([not completion[1].endswith(" one") for completion in new_bigram_beams.logprobs_and_completions])
assert any([not completion[1].endswith(" two") for completion in new_bigram_beams.logprobs_and_completions])

print("All tests for `generate` passed!")


# %% (Beams.filter testing)
