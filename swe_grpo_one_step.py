"""One GRPO step on one SWE-bench task -- script version of swe_grpo_one_step.ipynb.

The notebook walks through the task fields and the environment one piece at a
time; this script skips that and goes straight to the loop:

    load the task -> build the agent -> sample a group -> score it -> one update

Install:
    pip install "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" "peft>=0.12" torch
    pip uninstall -y torchao     # Colab ships 0.10, below peft's floor; peft raises on it

Run:
    python swe_grpo_one_step.py
"""

import json
import os
import random
import urllib.request

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from IPython.display import display
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/abgoswam/swe_in_prod_vizuara_01/main/utils.py",
        "utils.py")

from utils import NO_COMMAND, SYSTEM, MockEnv, first_bash_block, generate

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
GROUP_SIZE = 6
MAX_TURNS = 4
TEMPERATURE = 1.0
LR = 1e-5
SEED = 0


def load_task():
    """One SWE-bench task. Small and single-file, to keep the walkthrough readable."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    cands = [i for i, r in enumerate(ds)
             if r["patch"].count("diff --git") == 1 and len(r["patch"]) < 1800]
    inst = ds[cands[0]]
    return inst, json.loads(inst["FAIL_TO_PASS"])


def load_policy(device):
    """The model half of `agent = model + harness`. No adapter yet -- running
    the agent needs no trainable parameters."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa").to(device)

    print(f"{MODEL}\n{sum(p.numel() for p in model.parameters()):,} parameters, none trainable yet")
    return model, tok


def run_agent(model, tok, inst, fail_to_pass, max_turns=MAX_TURNS,
              temperature=TEMPERATURE, sample=generate):
    """The harness: prepare, call, parse, execute, append.

    One call returns one rollout -- the whole context, since in RL the
    trajectory is the training example. `max_turns` is the ONLY stopping
    condition; there is no early exit.
    """
    env = MockEnv(fail_to_pass)
    context = [{"role": "system", "content": SYSTEM},
               {"role": "user",   "content": f"ISSUE:\n{inst['problem_statement'][:1500]}"}]

    for _ in range(max_turns):
        prompt = tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        reply  = sample(model, tok, prompt, temperature=temperature)
        action = first_bash_block(reply)
        obs    = env.run(action) if action else NO_COMMAND
        context += [{"role": "assistant", "content": reply},
                    {"role": "user",      "content": obs[:800]}]

    return dict(messages=context, patch=env.patch(), final=dict(env.fs), calls=env.calls)


def reward_random(patch, rng):
    """Stand-in for a real verifier (unit tests, or a reward model).

    A real verifier applies test_patch plus the candidate patch and runs the
    tests. This one only asks "did the agent change anything?" -- 0 when the
    rollout produced no patch, a random score when it did. Called once per
    rollout, when that episode ends.
    """
    if not patch:
        return 0.0
    return round(float(rng.random()), 3)


def add_lora(model):
    """Only now do we need trainable parameters, so only now do we add LoRA.

    The model was loaded fp16 for fast generation and Adam's second moment
    underflows there; PEFT keeps the adapter in fp32, which avoids it.
    """
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    return model


def build_masked(messages, tokenizer, max_len=3072):
    """Token ids plus labels with everything but the assistant turns masked to -100."""
    ids, labels, prev = [], [], ""
    for i, m in enumerate(messages):
        cur = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False)
        assert cur.startswith(prev), "chat template is not append-only"
        seg = tokenizer(cur[len(prev):], add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if m["role"] == "assistant" else [-100] * len(seg)
        prev = cur
    return ids[:max_len], labels[:max_len]


def seq_logprob(model, tok, messages, device):
    """Mean log-prob over supervised (assistant) tokens, and how many there are."""
    ids, labs = build_masked(messages, tok)
    t = torch.tensor([ids], device=device)
    msk = torch.tensor([[0. if l == -100 else 1. for l in labs]], device=device)[:, 1:]
    logits = model(t).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * msk).sum() / msk.sum().clamp(min=1), msk.sum().item()


def grpo_step(model, tok, group, reward, device, lr=LR):
    """One GRPO update: group-relative advantages, then a REINFORCE step."""
    rewards = torch.tensor(reward, dtype=torch.float)
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)

    print(f"mean reward {rewards.mean():.3f}   std {rewards.std(unbiased=False):.3f}")
    if rewards.std(unbiased=False) < 1e-8:
        print("\n  DEGENERATE GROUP: every rollout scored the same, so every advantage is 0\n"
              "  and this update is a no-op. Not a bug -- it is the usual outcome when a\n"
              f"  0.5B model rarely earns reward. Re-run, or raise GROUP_SIZE (now {GROUP_SIZE}).\n")
    display(pd.DataFrame({
        "rollout":    list(range(len(group))),
        "reward":     rewards.numpy().round(3),
        "advantage":  adv.numpy().round(3),
        "sup_tokens": [int(seq_logprob(model, tok, g["messages"], device)[1]) for g in group],
    }))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    logp_before = [seq_logprob(model, tok, g["messages"], device)[0].item() for g in group]

    model.train()
    opt.zero_grad(set_to_none=True)
    loss_total = 0.0
    for g, a in zip(group, adv):
        lp, _ = seq_logprob(model, tok, g["messages"], device)
        loss = -(a.to(device) * lp) / len(group)      # REINFORCE with a group baseline
        loss.backward()
        loss_total += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()

    logp_after = [seq_logprob(model, tok, g["messages"], device)[0].item() for g in group]

    print(f"loss {loss_total:+.5f}   grad_norm {grad_norm:.4f}")
    display(pd.DataFrame({
        "rollout":     list(range(len(group))),
        "advantage":   [round(a.item(), 3) for a in adv],
        "logp_before": [round(b, 4) for b in logp_before],
        "logp_after":  [round(c, 4) for c in logp_after],
        "delta":       [round(c - b, 5) for b, c in zip(logp_before, logp_after)],
    }))


def main():
    for _opt in ["display.max_colwidth", "display.max_rows",
                 "display.max_columns", "display.width"]:
        pd.set_option(_opt, None)

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1. the task: one SWE-bench issue, and the tests that define "fixed"
    inst, fail_to_pass = load_task()
    print(f"{inst['instance_id']}  |  {len(fail_to_pass)} tests must go red -> green")

    # 2. the policy: the model half of agent = model + harness
    model, tok = load_policy(device)

    # 3. the group: GRPO compares rollouts of the SAME task against each other,
    #    so run the agent GROUP_SIZE times from the same prompt. Each episode
    #    is scored the moment it ends -- that is what a real verifier does.
    rng = np.random.default_rng(SEED)
    group, reward = [], []
    for i in range(GROUP_SIZE):
        rollout = run_agent(model, tok, inst, fail_to_pass)
        score = reward_random(rollout["patch"], rng)
        group.append(rollout)
        reward.append(score)
        print(f"rollout {i}: {len(rollout['calls'])} commands, "
              f"patch {'yes' if rollout['patch'] else 'no'}, reward {score}")

    reward = np.array(reward)

    # 4. trainable parameters, needed only from here on
    model = add_lora(model)

    # 5. the update: rewards -> advantages -> one gradient step
    grpo_step(model, tok, group, reward, device)
    
    print("done")


if __name__ == "__main__":
    main()
