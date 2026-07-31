"""Plumbing for the one-GRPO-step walkthrough.

Nothing in this file is a concept the notebook teaches -- it is the prompt
text, the sampling boilerplate and the fake environment, kept here so the
notebook itself can stay about the harness, the reward and the update.

Every method of MockEnv is a lie. The comments say which kind.
"""

import difflib
import re

import pandas as pd
import torch
from IPython.display import display

TARGET = "astropy/modeling/separable.py"

# Verbatim source at astropy@d16bfe05, trimmed to the one function the issue is
# about. The bug is the asymmetry: the cleft branch assigns `left`, the cright
# branch assigns `1`.
MOCK_FILE = '''def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.

    """
    noutp = _compute_n_outputs(left, right)

    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', noutp)
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], : left.shape[1]] = left
    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = 1

    return np.hstack([cleft, cright])'''

RE_BASH = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.S)
RE_HEREDOC = re.compile(r"cat\s*>\s*(\S+)\s*<<\s*'?EOF'?\n(.*?)\nEOF", re.S)

SYSTEM = f"""You are a software engineering agent. You are in a Python repository.
Fix the bug described in the issue.

Respond with exactly ONE bash command per message, in a fenced block:

```bash
your command here
```

Useful commands:
  ls
  cat {TARGET}
  grep -n "pattern" {TARGET}
  python -m pytest

To rewrite a file:
```bash
cat > {TARGET} <<'EOF'
...full new contents...
EOF
```

One short sentence of reasoning, then exactly one command block."""

NO_COMMAND = "No command found. Reply with exactly one ```bash block."


def first_bash_block(reply):
    """The parse step: pull one action out of free text, or None."""
    m = RE_BASH.search(reply)
    return m.group(1).strip() if m else None


def show_rollout(rollout, preview=160):
    """Trajectory table, commands issued, and the candidate patch.

    `preview` caps the preview column; pass None to show messages in full.
    """
    def author(i, m):
        if m["role"] == "assistant":
            return "model"
        return "harness" if i < 2 else "environment"

    def shorten(text):
        one_line = text.replace("\n", " | ")
        if preview is None or len(one_line) <= preview:
            return one_line
        return one_line[:preview] + " […]"

    msgs = rollout["messages"]
    n_turns = (len(msgs) - 2) // 2
    print(f"{len(msgs)} messages = 2 setup + 2 per turn x {n_turns} turns\n")

    display(pd.DataFrame([
        {"i": i, "turn": "setup" if i < 2 else (i - 2) // 2,
         "role": m["role"], "written by": author(i, m),
         "chars": len(m["content"]),
         "preview": shorten(m["content"])}
        for i, m in enumerate(msgs)
    ]))

    print("\ncommands:")
    for c in rollout["calls"]:
        print("  $", c.splitlines()[0] + (" ..." if "\n" in c else ""))

    print("\ncandidate patch:")
    print(rollout["patch"] or "(no file changed)")


def show_group(group, reward):
    """Summary table for a group of rollouts, then every trace verbatim."""
    display(pd.DataFrame([
        {"rollout":     i,
         "reward":      reward[i],
         "patch_lines": len(r["patch"].split("\n")) if r["patch"] else 0,
         "n_commands":  len(r["calls"])}
        for i, r in enumerate(group)
    ]))

    for i, r in enumerate(group):
        print(f"\n================ rollout {i} ================")
        for cmd in r["calls"]:
            print("  $", cmd)
        print("\ncandidate patch:")
        print(r["patch"] or "(no file changed)")


def scripted_sampler(replies):
    """A stand-in for `generate`: replays canned replies, ignoring the prompt.

    Lets the same harness be driven by a competent 'model' so a demo can show
    what a successful rollout looks like.
    """
    it = iter(replies)

    def sample(model, tok, prompt, **kwargs):
        return next(it, "```bash\nls\n```")

    return sample


def generate(model, tok, prompt, max_new_tokens=200, temperature=1.0, max_length=3072):
    """The call step: sample a continuation, return only the new text."""
    device = next(model.parameters()).device
    ip = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model.generate(**ip, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.95,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ip["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MockEnv:
    """A shell that never runs a shell.

    Stands in for the container a real harness would exec into. Handles ls,
    cat, grep, pytest and a heredoc write; anything else returns a stub.
    """

    def __init__(self, failing_tests=()):
        self.fs = {TARGET: MOCK_FILE}
        self.orig = dict(self.fs)
        self.failing_tests = list(failing_tests)
        self.calls = []

    def run(self, cmd):
        self.calls.append(cmd)

        m = RE_HEREDOC.search(cmd)
        if m:                                   # real effect on the virtual FS
            path, body = m.group(1), m.group(2)
            self.fs[path] = body
            return f"[mock env] wrote {len(body.splitlines())} lines to {path}"

        if cmd.strip().startswith("ls"):        # canned
            return "\n".join(sorted(self.fs)) + "\nsetup.py\nREADME.rst\ntests/"

        if cmd.strip().startswith("cat "):      # real source, one function of it
            path = cmd.split()[-1]
            if path in self.fs:
                return ("[mock env] only the region near the bug is available\n"
                        + self.fs[path])
            return f"cat: {path}: No such file or directory"

        if cmd.strip().startswith("grep"):      # real search over the fragment
            pat = re.findall(r'"([^"]*)"|\'([^\']*)\'', cmd)
            pat = next((a or b for a, b in pat), "")
            src = self.fs.get(cmd.split()[-1], "")
            hits = [f"{i + 1}:{line}" for i, line in enumerate(src.split("\n"))
                    if pat and pat in line]
            return "\n".join(hits) if hits else "(no matches)"

        if "pytest" in cmd:                     # canned failure, real test names
            body = "\n".join(f"FAILED {n}" for n in self.failing_tests)
            return (f"{body}\n=== {len(self.failing_tests)} failed ===\n"
                    "[mock env] test names are real, but nothing was executed")

        name = cmd.split()[0] if cmd.split() else cmd
        return (f"bash: {name}: command not found\n"
                "[mock env] implements only: ls, cat, grep, pytest, cat > FILE <<'EOF'")

    def patch(self):
        """The candidate patch: a real unified diff of the virtual FS.

        A file the agent created did not exist at the start, so it diffs
        against empty -- the same as `git diff` for a new file.
        """
        out = []
        for path, body in self.fs.items():
            before = self.orig.get(path, "")
            if body != before:
                out += list(difflib.unified_diff(
                    before.split("\n"), body.split("\n"),
                    fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""))
        return "\n".join(out)
