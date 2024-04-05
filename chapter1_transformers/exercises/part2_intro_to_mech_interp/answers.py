# %% (setup)
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_intro_to_mech_interp', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("mps" if t.backends.mps.is_available() else ("cuda" if t.cuda.is_available() else "cpu"))

MAIN = __name__ == "__main__"
# %% (loading gpt2_small)
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")


# %% inspect your model
print(gpt2_small.cfg.n_layers)
print(gpt2_small.cfg.n_heads)
print(gpt2_small.cfg.n_ctx)
# %% (running your model)
model_description_text = '''## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)


# %% how many tokens does your model guess correctly?
logits: Float[Tensor, "batch seq d_model"] = gpt2_small(model_description_text, return_type="logits")
predictions = logits.argmax(dim=-1).squeeze()[..., :-1]
actual = gpt2_small.to_tokens(model_description_text).squeeze()[..., 1:]
n_correct = (actual == predictions).sum().item()
accuracy = n_correct / len(actual)
correct_tokens = gpt2_small.to_str_tokens(predictions[predictions==actual])
print(f"Accuracy: {accuracy}")
print(f"Correct tokens: {correct_tokens}")

# %% (caching activations)
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)


# %% (indexing shorthand)
attn_patterns_layer_0 = gpt2_cache["pattern", 0]
attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)


# %%
layer0_pattern_from_cache: Float[Tensor, "n_heads seq seq"]  = gpt2_cache["pattern", 0]
hook_q: Float[Tensor, "seq n_heads d_head"] = gpt2_cache["q", 0]
hook_k: Float[Tensor, "seq n_heads d_head"] = gpt2_cache["k", 0]
seq, _, d_head = hook_q.shape

attn_scores = einops.einsum(hook_q, hook_k, "seq_q n_heads d_head, seq_k n_heads d_head -> n_heads seq_q seq_k")
mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
masked_attn_scores = attn_scores.masked_fill(mask, -1e5)
scaled_attn_scores = (masked_attn_scores / d_head ** 0.5)
layer0_pattern_from_q_and_k = scaled_attn_scores.softmax(dim=-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")


# %% (visualizing attention heads)
print(type(gpt2_cache))
attention_pattern_0 = gpt2_cache["pattern", 0]
print(attention_pattern_0.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=gpt2_str_tokens, 
    attention=attention_pattern_0,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))


# %% (neuron activations)
neuron_activations_for_all_layers = t.stack([
    gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)
], dim=1)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens,
    activations=neuron_activations_for_all_layers
)


# %%(topk_tokens)
neuron_activations_for_all_layers_rearranged = utils.to_numpy(einops.rearrange(neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"))

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens], 
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7, 
    first_dimension_name="Layer", 
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12))
)

# %% === FINDING INDUCTION HEADS ===
# %% (cfg)
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

# %% (weights)
weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

if not weights_dir.exists():
    url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
    output = str(weights_dir)
    gdown.download(url, output)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_dir, map_location=device)
model.load_state_dict(pretrained_weights)

# %% visualize attention patterns
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
tokens = model.to_str_tokens(text)

attention_pattern_0 = cache["pattern", 0]

display(cv.attention.attention_patterns(
    tokens=tokens,
    attention=attention_pattern_0,
    attention_head_names=[f"L0H{i}" for i in range(12)],
))

attention_pattern_1 = cache["pattern", 1]

display(cv.attention.attention_patterns(
    tokens=tokens,
    attention=attention_pattern_1,
    attention_head_names=[f"L1H{i}" for i in range(12)],
))


# %% write your own detectors
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    out = []
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        attention_pattern = cache["pattern", layer][head]
        diag_score = t.diagonal(attention_pattern).mean()
        if diag_score > 0.4:
           out.append(f"{layer}.{head}")
    return out

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    out = []
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        attention_pattern = cache["pattern", layer][head]
        diag_score = t.diagonal(attention_pattern, offset=-1).mean()
        if diag_score > 0.4:
           out.append(f"{layer}.{head}")
    return out

    

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    out = []
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        attention_pattern = cache["pattern", layer][head]
        diag_score = attention_pattern[:, 0].mean()
        if diag_score > 0.4:
           out.append(f"{layer}.{head}")
    return out


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %% plot per-token loss on repeated seqeunce
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix: Int[Tensor, "batch 1"] = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    pattern: Int[Tensor, "batch 2*seq_len"] = t.randint(0, model.cfg.d_vocab, (batch, seq_len)).long().to(device)
    return t.cat([prefix, pattern, pattern], dim=-1)


def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    tokens = generate_repeated_tokens(model, seq_len, batch)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    print(type(cache))
    return (tokens, logits, cache)


seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=rep_str, attention=attention_pattern))
# %% induction_attn_detector
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    induction_heads = []
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        attention_pattern: Float[Tensor, "seq seq"] = cache["pattern", layer][head]
        diag = t.diagonal(attention_pattern, offset=-(seq_len-1))
        if diag.mean() > 0.4:
          induction_heads.append(f"{layer}.{head}")
    return induction_heads

print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))


# %% === TRANSFORMERLENS: HOOKS ===
# %% calculate induction scores with hooks
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    diag = t.diagonal(pattern, dim1=-2, dim2=-1, offset=-(seq_len-1))
    score = einops.reduce(diag, "batch head_index diag_len -> head_index", "mean")
    induction_score_store[hook.layer(), :] = score


pattern_hook_names_filter = lambda name: name.endswith("pattern")

print(rep_tokens_10.shape)
# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)


# %% find induction heads in GPT2-small
def visualize_pattern_hook(tokens: any):
    def out(
      pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
      hook: HookPoint,
    ):
      print("Layer: ", hook.layer())
      display(
        cv.attention.attention_patterns(
            tokens=tokens, 
            attention=pattern.mean(0)
        )
      )
    return out

def generate_repeated_token_n_sequences(
    model: HookedTransformer, seq_len: int=50, batch: int=1, n_repeats: int=4
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+n_repeats*seq_len]
    '''
    prefix: Int[Tensor, "batch 1"] = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    pattern: Int[Tensor, "batch seq_len"] = t.randint(0, model.cfg.d_vocab, (batch, seq_len)).long().to(device)
    return t.cat([prefix, *[pattern for _ in range(n_repeats)]], dim=-1)


rep_tokens_n_sequences = generate_repeated_token_n_sequences(gpt2_small, seq_len=50, batch=10, n_repeats=4)
# print(rep_tokens_n_sequences.shape)
# gpt2_small.run_with_hooks(
#     rep_tokens_n_sequences,
#     return_type=None, # For efficiency, we don't need to calculate the logits
#     fwd_hooks=[(
#         pattern_hook_names_filter,
#         visualize_pattern_hook(tokens=gpt2_small.to_str_tokens(rep_tokens[0]))
#     )]
# )

# %% build logit attribution tool
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence
    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens: Float[Tensor, "d_model seq-1"] = W_U[:, tokens[1:]]
    print(embed[:-1].shape, W_U_correct_tokens.shape)
    direct_path: Float[Tensor, "seq-1 seq-1"] = einops.einsum(embed[:-1], W_U_correct_tokens, "seq d_model, d_model seq -> seq") 
    l1_contribution: Float[Tensor, "seq-1 n_heads seq-1"] = einops.einsum(l1_results[:-1], W_U_correct_tokens, "seq n_heads d_model, d_model seq -> seq n_heads")
    l2_contribution: Float[Tensor, "seq-1 n_heads seq-1"] = einops.einsum(l2_results[:-1], W_U_correct_tokens, "seq n_heads d_model, d_model seq -> seq n_heads")
    return t.cat((direct_path.unsqueeze(-1), l1_contribution, l2_contribution), dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")

embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)


# %% logit attribution for the induction heads
seq_len = 50

embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
first_half_tokens = rep_tokens[0, : 1 + seq_len]
second_half_tokens = rep_tokens[0, seq_len:]

# YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
first_half_logit_attr = logit_attribution(embed[:seq_len+1], l1_results[:seq_len+1], l2_results[:seq_len+1], model.W_U, first_half_tokens)
second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.W_U, second_half_tokens)

assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")

# %% ablation
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    out = t.clone(v)
    out[:, :, head_index_to_ablate, :] = 0
    return out


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)

# %% (plot ablation results)
imshow(
    ablation_scores, 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)


# %% ablate all but induction circuit

def ablate_all_but_induction_head_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
) -> Float[Tensor, "batch seq n_heads d_head"]:
    out = t.zeros_like(v)
    if hook.layer() == 0:
      out[:, :, 7, :] = v[:, :, 7, :]
    else:
      out[:, :, 4, :] = v[:, :, 4, :]
      out[:, :, 10, :] = v[:, :, 10, :]
    return out


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores_for_induction_isolation_hook(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(ablate_all_but_induction_head_hook)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores_for_induction_isolation_hook = get_ablation_scores_for_induction_isolation_hook(model, rep_tokens)

imshow(
    ablation_scores_for_induction_isolation_hook, 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads", 
    text_auto=".2f",
    width=900, height=400
)

# %% === REVERSE-ENGINEERING INDUCTION CIRCUITS
# %% Testing IOI Circuit
gpt2_text = "When Mary and John and Bob went to the store, John gave a drink to"
gpt2_tokens = gpt2_small.to_tokens(gpt2_text).long()
gpt2_small.run_with_hooks(
    gpt2_tokens,
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        lambda x: x.endswith("pattern"),
        visualize_pattern_hook(tokens=gpt2_small.to_str_tokens(gpt2_tokens))
    )]
)
# %%
