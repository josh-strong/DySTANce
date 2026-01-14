# DySTANce `ContextManager` — how it works and how to use it

This note explains the `ContextManager` implemented in `dev_notebooks/imports.py`, and how it is used in `dev_notebooks/03_training_eval.ipynb`.

## What `ContextManager` is (and isn’t)

`ContextManager` is **not** a Python `with ...` context manager (it does **not** implement `__enter__` / `__exit__`). It’s a helper class that **manages “few-shot context sets”** used to *describe each tool by its behaviour* on a small set of examples, while keeping those examples **disjoint** from the examples used to train the router.

Conceptually (matching the paper-style notation in the code docstring), each tool \(E\) is described for each task \(t\) by:

\[
D_E^t = \{(x_b, y_b^t, m_E^t(x_b))\}_{b=1}^{B_t}
\]

where:
- \(x_b\) is an input image
- \(y_b^t\) is the ground-truth label for task \(t\)
- \(m_E^t(x_b)\) is the tool’s prediction for task \(t\) on that image
- \(B_t\) is the number of context examples (few-shot) used to describe the tool for that task

## What it guarantees

The core guarantee is **no leakage**:

- **Context partition**: examples used only to *describe tools*
- **Routing partition**: examples used only to *train the router*

An image used in the context partition is never used for routing loss computation, so “tool descriptors” remain exogenous to the routing objective.

## How it works internally

### 1) It randomly splits the training dataset into context vs routing

When you instantiate `ContextManager(dataset=..., context_fraction=...)`, it:
- shuffles all dataset indices (`torch.randperm`)
- takes the first `context_fraction` as `context_idx`
- assigns the rest to `routing_idx`

### 2) It pre-indexes valid context examples per (tool, task)

From the **context partition only**, it builds a lookup table:

`pool[(tool_idx, task_idx)] -> [dataset indices]`

An index is added to the pool only if that example is **tool- and task-valid**, based on `tool_mask`.

This matters because in `OpenIRoutedDataset` tools often emit a default `0.5` on unsupported labels; those should not be treated as informative behaviour. The mask filters those out so the context set represents *actual* tool behaviour.

### 3) `sample_context(tool_idx, task_idx)` returns the few-shot context triples

`sample_context(tool_idx, task_idx)`:
- looks up valid indices in `pool[(tool_idx, task_idx)]`
- samples up to `examples_per_tool` indices (few-shot, exchangeable)
- returns:
  - `images`: `Tensor[B, C, H, W]`
  - `gt_labels`: `Tensor[B]`
  - `tool_predictions`: `Tensor[B]`

If no valid context exists for that tool-task pair, it returns `None`. This signals: “this tool has no observable behaviour for this task”.

### 4) `routing_dataset()` returns the router-training subset

`routing_dataset()` returns `torch.utils.data.Subset(dataset, routing_idx)`, i.e. only the examples reserved for router training.

## How to use it (the intended pattern)

The intended flow (as in `03_training_eval.ipynb`) is:

1. Build a full training `OpenIRoutedDataset` (must include `tool_preds` and `tool_mask`).
2. Create a `ContextManager` **from the full training dataset**.
3. Train the router only on the **routing split** returned by `routing_dataset()`.
4. Whenever you need tool descriptors for a particular task `t`, call `sample_context(tool_idx, t)` for each tool and pass the returned triples into the model’s tool-description pipeline (e.g. ANP encoder).

Example sketch:

```python
ctx_mgr = imports.ContextManager(
    dataset=train_dataset_full,
    context_fraction=0.1,   # 10% reserved for tool-description context
    examples_per_tool=32,   # B_t
)

# Train the router only on non-context data
train_dataset = ctx_mgr.routing_dataset()

# Build tool contexts for a particular task
task_idx = 7
for tool_idx in range(ctx_mgr.dataset.M):
    ctx = ctx_mgr.sample_context(tool_idx, task_idx)
    if ctx is None:
        # Tool has no valid behaviour for this task
        continue
    imgs, gt, preds = ctx
    # imgs:  [B, 3, H, W]
    # gt:    [B]
    # preds: [B]
```

## Practical gotcha (important for stacking tensors)

`sample_context()` returns **up to** `examples_per_tool` examples:

- If a tool-task pair has fewer than `examples_per_tool` valid examples, `B < examples_per_tool`.
- If downstream code assumes a fixed `C = examples_per_tool` for stacking tensors across tools (e.g. building `[M, C, ...]` tensors), you’ll need to **pad/truncate** each tool’s context to a common length.

In `03_training_eval.ipynb`, `build_context_tensors()` currently sets `C = ctx_mgr.examples_per_tool` and then stacks per-tool tensors. That will only work reliably if each `sample_context()` returns exactly `C` examples (or if you pad/truncate to `C` before stacking).

