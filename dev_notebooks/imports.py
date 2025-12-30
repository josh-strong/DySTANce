import os, csv, re
import numpy as np
import torch
from typing import Dict, List, Optional

def infer_split_from_filename(fname: str) -> Optional[str]:
    """
    Helper to figure out the split from the filename
    """
    if fname.endswith('test.csv'):
        return 'test'
    elif fname.endswith('val.csv'):
        return 'val'
    elif fname.endswith('train.csv'):
        return 'train'
    else:
        import warnings
        warnings.warn(f"Could not infer split from filename: {fname}", UserWarning)
        return None

def scan_prediction_files(pred_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Helper to scan prediction files, infer the split from the filename, and return 
    registry of the form:   
        registry[split][tool_name] = csv_path
    """
    registry = {"train": {}, "val": {}, "test": {}}
    for fn in os.listdir(pred_dir):
        if not fn.endswith(".csv"):
            continue
        split = infer_split_from_filename(fn)
        if split is None:
            continue
        stem = os.path.splitext(fn)[0]
        tool = re.sub(rf"(_)?{split}(_)?$", "", stem, flags=re.IGNORECASE).strip("_")
        registry[split][tool] = os.path.join(pred_dir, fn)
    return registry

import pandas as pd
def read_predictions_csv(
    csv_path: str,
    label_names: List[str],
    id_candidates=("id", "filename"),
) -> Dict[str, np.ndarray]:
    """
    Takes in a csv path and a list of label names, and returns a dict of image_id -> [L] float array
    This is for the multi-label task, where we treat it as l independent binary classifiers. 
    Note: Missing or unsupported labels are filled with 0.5!

    Returns dict: image_id -> [L] float array
    """
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lc = [h.lower().strip() for h in header]

        id_idx = None
        for cand in id_candidates:
            if cand in header_lc:
                id_idx = header_lc.index(cand)
                break
        if id_idx is None:
            return out

        label_idx = []
        for l in label_names:
            label_idx.append(header_lc.index(l.lower()) if l.lower() in header_lc else None)

        for row in reader:
            img_id = row[id_idx].replace(".jpg", "").strip()
            vec = []
            for j in label_idx:
                if j is None:
                    vec.append(0.5)
                else:
                    try:
                        vec.append(float(row[j]))
                    except Exception:
                        vec.append(0.5)
            out[img_id] = np.asarray(vec, dtype=np.float32)
    return out

from torch.utils.data import Dataset
from PIL import Image

def _fallback_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1)

class OpenIRoutedDataset(Dataset):
    """
    Dataset for the OpenI dataset

    Returns per-sample:
      image        : Tensor[C,H,W]
      gt           : Tensor[L]
      tool_preds   : Tensor[M, L]
      tool_mask    : Tensor[M, L]  (1 = tool valid for task, 0 invalide)
      id           : str
    """

    def __init__(
        self,
        label_csv: str,
        images_dir: str,
        predictions_registry: Dict[str, str],
        label_names: List[str],
        transform=None,
        check_files=False,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.label_names = label_names
        self.L = len(label_names)

        # --- Load labels ---
        self.records = []
        with open(label_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            hmap = {h.strip(): i for i, h in enumerate(header)}
            for row in reader:
                img_id = row[0].strip()
                path = os.path.join(images_dir, f"{img_id}.jpg")
                if check_files and not os.path.exists(path):
                    continue
                gt = [float(row[hmap[l]]) for l in label_names]
                self.records.append({
                    "id": img_id,
                    "path": path,
                    "gt": torch.tensor(gt, dtype=torch.float32)
                })

        # --- Load tool predictions ---
        self.tool_names = sorted(predictions_registry.keys())
        self.M = len(self.tool_names)

        self.tool_preds = []
        for tool in self.tool_names:
            self.tool_preds.append(
                read_predictions_csv(predictions_registry[tool], label_names)
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = _fallback_to_tensor(img)


        preds = torch.full((self.M, self.L), 0.5)
        mask  = torch.zeros((self.M, self.L))

        for m, tool_dict in enumerate(self.tool_preds):
            if rec["id"] in tool_dict:
                p = torch.from_numpy(tool_dict[rec["id"]])
                preds[m] = p
                mask[m] = (torch.abs(p - 0.5) > 1e-4).float()

        return {
            "image": img,
            "gt": rec["gt"],
            "tool_preds": preds,
            "tool_mask": mask,
            "id": rec["id"]
        }

from torch.utils.data import Subset
import random

class ContextManager:
    """
    Manages the few-shot context sets used to describe tools in DySTANce.

    Key idea (from the paper):
    ------------------------
    Tools are NOT identified by IDs or learned embeddings.
    Instead, each tool E is represented only through its behaviour
    on a small, task-specific context set:

        D_E^t = {(x_b, y_b^t, m_E^t(x_b))}_{b=1}^{B_t}

    This class is responsible for:
      1) Constructing these context sets in a leakage-free way
      2) Ensuring context examples are task- and tool-valid
      3) Enforcing a strict separation between:
           - data used to DESCRIBE tools (context)
           - data used to TRAIN the router (routing set)

    """

    def __init__(
        self,
        dataset: OpenIRoutedDataset,
        context_fraction: float = 0.1,
        examples_per_tool: int = 32,
    ):
        """
        Parameters
        ----------
        dataset : OpenIRoutedDataset
            Full TRAINING dataset containing images, ground-truth labels,
            tool predictions, and tool validity masks.

        context_fraction : float
            Fraction of the training data reserved EXCLUSIVELY for
            tool description (context). These samples are never used
            for routing loss computation.

        examples_per_tool : int
            Number of context examples B_t to sample per (tool, task)
            when constructing the ANP summary.
        """

        self.dataset = dataset
        self.examples_per_tool = examples_per_tool

        # ------------------------------------------------------------------
        # 1) Split dataset indices into CONTEXT and ROUTING partitions
        # ------------------------------------------------------------------
        # This enforces the core invariance:
        #   "An image used to describe a tool is never used to train the router."
        #
        # This prevents information leakage and ensures the ANP summaries
        # remain exogenous to the routing objective.
        # ------------------------------------------------------------------
        N = len(dataset)
        perm = torch.randperm(N).tolist()  # random i.i.d. partition
        split = int(context_fraction * N)

        self.context_idx = perm[:split]    # used ONLY for tool descriptors
        self.routing_idx = perm[split:]    # used ONLY for router training

        # ------------------------------------------------------------------
        # 2) Pre-index valid context examples
        # ------------------------------------------------------------------
        # We build a lookup table:
        #
        #   (tool_idx, task_idx) -> [dataset indices]
        #
        # Only examples where:
        #   - the tool actually produced a meaningful prediction
        #   - for the specific task (label)
        #
        # are included.
        #
        # This is critical because many tools emit "0.5" for unsupported
        # labels, which must NOT contaminate the context set.
        # ------------------------------------------------------------------
        self.pool = {}  # maps (tool_idx, task_idx) to list of dataset indices

        for i in self.context_idx:
            item = dataset[i]
            mask = item["tool_mask"]  # shape: [num_tools, num_tasks]

            # Iterate over all (tool, task) pairs and record valid contexts
            for t in range(dataset.M):
                for l in range(dataset.L):
                    if mask[t, l] > 0.5:
                        # This example is informative for tool t on task l
                        self.pool.setdefault((t, l), []).append(i)

    def sample_context(self, tool_idx: int, task_idx: int):
        """
        Samples a few-shot context set D_E^t for a specific tool and task.

        Returns
        -------
        (images, gt_labels, tool_predictions) or None

        images           : Tensor[B, C, H, W]
        gt_labels        : Tensor[B]
        tool_predictions : Tensor[B]

        This tuple corresponds exactly to:
            (x_b, y_b^t, m_E^t(x_b))_{b=1}^{B_t}

        If no valid context exists for (tool, task), returns None.
        This signals that the tool has no observable behavior for this task.
        """

        key = (tool_idx, task_idx)
        candidates = self.pool.get(key, [])

        # If the tool has never produced a valid prediction for this task,
        # we cannot construct a meaningful context descriptor.
        if len(candidates) == 0:
            return None

        # Randomly sample up to B_t context examples (few-shot, exchangeable)
        idxs = random.sample(
            candidates,
            k=min(self.examples_per_tool, len(candidates))
        )

        imgs, gt, preds = [], [], []
        for i in idxs:
            item = self.dataset[i]

            # Each context triple corresponds to:
            #   image x_b
            #   ground-truth label y_b^t
            #   tool prediction m_E^t(x_b)
            imgs.append(item["image"])
            gt.append(item["gt"][task_idx])
            preds.append(item["tool_preds"][tool_idx, task_idx])

        return (
            torch.stack(imgs),
            torch.stack(gt),
            torch.stack(preds),
        )

    def routing_dataset(self):
        """
        Returns the subset of the dataset used for training the router.

        This subset is guaranteed to be disjoint from the context set,
        ensuring no leakage between tool description and routing loss.
        """
        return Subset(self.dataset, self.routing_idx)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class InstructionEncoder(nn.Module):
    """
    Encodes the textual instruction q.
    For now: simple embedding + mean pooling.
    In large-scale settings this would be a frozen LLM / CLIP text encoder.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor):
        """
        Args:
            token_ids: [B, T] integer tokens

        Returns:
            Tensor[B, embed_dim]
        """
        emb = self.embedding(token_ids)          # [B, T, D]
        return emb.mean(dim=1)                   # mean pool over tokens


class ANPToolEncoder(nn.Module):
    """
    Attentive Neural Process encoder for task-conditional tool descriptors.

    Builds a representation z_E^t(p) from a small context set D_E^t.
    """

    def __init__(
        self,
        img_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()

        # Encodes individual context elements:
        # [phi_x(x_b) || y_b^t || m_E^t(x_b)] -> hidden
        self.context_proj = nn.Sequential(
            nn.Linear(img_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Self-attention over context elements
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm_ctx = nn.LayerNorm(hidden_dim)

        # Cross-attention projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(img_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim ** -0.5
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query_embed: torch.Tensor,
        ctx_img_feat: torch.Tensor,
        ctx_gt: torch.Tensor,
        ctx_pred: torch.Tensor,
    ):
        """
        Args:
            query_embed : [B, H]              = u(p)
            ctx_img_feat: [M, C, Dx]          = phi_x(x_b)
            ctx_gt      : [M, C]              = y_b^t
            ctx_pred    : [M, C]              = m_E^t(x_b)

        Returns:
            z_E : [B, M, H]  task-conditional tool descriptors
        """
        M, C, Dx = ctx_img_feat.shape
        B = query_embed.shape[0]

        # ------------------------------------------------------------------
        # 1) Encode context elements
        # ------------------------------------------------------------------
        ctx_input = torch.cat(
            [
                ctx_img_feat,
                ctx_gt.unsqueeze(-1),
                ctx_pred.unsqueeze(-1),
            ],
            dim=-1
        )  # [M, C, Dx+2]

        ctx_emb = self.context_proj(ctx_input)  # [M, C, H]

        # ------------------------------------------------------------------
        # 2) Self-attention over context (per tool)
        # ------------------------------------------------------------------
        ctx_emb_sa, _ = self.self_attn(ctx_emb, ctx_emb, ctx_emb)
        ctx_emb = self.norm_ctx(ctx_emb + ctx_emb_sa)  # residual

        # ------------------------------------------------------------------
        # 3) Cross-attention: query u(p) attends to each tool’s context
        # ------------------------------------------------------------------
        Q = self.W_Q(query_embed)                # [B, H]
        Q = Q.unsqueeze(1).expand(-1, M, -1)     # [B, M, H]

        K = self.W_K(ctx_img_feat)               # [M, C, H]
        V = self.W_V(ctx_emb)                    # [M, C, H]

        # Attention scores: [B, M, C]
        attn_logits = torch.einsum(
            "bmh,mch->bmc", Q, K
        ) * self.scale

        attn = F.softmax(attn_logits, dim=-1)

        # Weighted sum of values -> [B, M, H]
        z = torch.einsum("bmc,mch->bmh", attn, V)

        return self.norm_out(z)

class DySTANceRouter(nn.Module):
    """
    Full DySTANce routing model.

    Given a query (image, instruction, task) and a panel of tools,
    outputs a scalar routing score for each tool.
    """

    def __init__(
        self,
        num_tasks: int,
        vocab_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Image encoder phi_x
        # ------------------------------------------------------------
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.img_dim = 512

        # ------------------------------------------------------------
        # Instruction encoder phi_q
        # ------------------------------------------------------------
        self.text_encoder = InstructionEncoder(vocab_size, 64)

        # ------------------------------------------------------------
        # Task embedding
        # ------------------------------------------------------------
        self.task_embed = nn.Embedding(num_tasks, 32)

        # ------------------------------------------------------------
        # Prompt fusion u(p)
        # ------------------------------------------------------------
        self.prompt_fusion = nn.Sequential(
            nn.Linear(self.img_dim + 64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ------------------------------------------------------------
        # ANP module psi_E^t(p)
        # ------------------------------------------------------------
        self.anp = ANPToolEncoder(
            img_dim=self.img_dim,
            hidden_dim=hidden_dim,
        )

        # ------------------------------------------------------------
        # Router head g_theta
        # ------------------------------------------------------------
        # Input: [u(p) || z_E || m_E^t(x)]
        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_img_feat(self, images: torch.Tensor):
        return self.img_encoder(images).flatten(1)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        task_idx: torch.Tensor,
        tool_preds: torch.Tensor,
        ctx_img_feat: torch.Tensor,
        ctx_gt: torch.Tensor,
        ctx_pred: torch.Tensor,
        tool_mask: torch.Tensor,
    ):
        """
        Args:
            images     : [B, 3, H, W]
            text_tokens: [B, T]
            task_idx   : [B]
            tool_preds : [B, M]          m_E^t(x)
            ctx_img_feat: [M, C, Dx]
            ctx_gt     : [M, C]
            ctx_pred   : [M, C]
            tool_mask  : [B, M]          1 if tool supports task

        Returns:
            scores : [B, M]
        """

        # ------------------------------------------------------------
        # Build u(p)
        # ------------------------------------------------------------
        img_feat = self.extract_img_feat(images)
        txt_feat = self.text_encoder(text_tokens)
        task_feat = self.task_embed(task_idx)

        u_p = torch.cat([img_feat, txt_feat, task_feat], dim=-1)
        u_p = self.prompt_fusion(u_p)  # [B, H]

        # ------------------------------------------------------------
        # Tool descriptors z_E^t(p)
        # ------------------------------------------------------------
        z_E = self.anp(u_p, ctx_img_feat, ctx_gt, ctx_pred)  # [B, M, H]

        # ------------------------------------------------------------
        # Router head
        # ------------------------------------------------------------
        u_exp = u_p.unsqueeze(1).expand(-1, z_E.size(1), -1)
        tool_preds = tool_preds.unsqueeze(-1)

        router_in = torch.cat([u_exp, z_E, tool_preds], dim=-1)
        scores = self.router_head(router_in).squeeze(-1)  # [B, M]

        # ------------------------------------------------------------
        # Hard mask invalid tools
        # ------------------------------------------------------------
        scores = scores.masked_fill(tool_mask == 0, -1e9)

        return scores

class DySTANceLoss(nn.Module):
    """
    Population comp-sum surrogate loss for DySTANce routing.

    This loss:
    - supports soft costs in [0,1]
    - allows multiple near-optimal tools
    - handles variable panel sizes
    - is compatible with the theory in the paper
    """

    def __init__(
        self,
        surrogate_type: str = "logistic",
        lambda_entropy: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.surrogate_type = surrogate_type
        self.lambda_entropy = lambda_entropy
        self.eps = eps

    def forward(
        self,
        router_logits: torch.Tensor,  # [B, M]
        tool_costs: torch.Tensor,     # [B, M] in [0,1]
        validity_mask: torch.Tensor,  # [B, M] in {0,1}
    ):
        B, M = router_logits.shape

        # ------------------------------------------------------------
        # 1) Masked softmax over valid tools
        # ------------------------------------------------------------
        masked_logits = router_logits.masked_fill(validity_mask == 0, -1e9)
        pi = F.softmax(masked_logits, dim=1)  # [B, M]

        # ------------------------------------------------------------
        # 2) Effective panel size per sample
        # ------------------------------------------------------------
        m_eff = validity_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # ------------------------------------------------------------
        # 3) Cost centering (KEY FIX)
        # ------------------------------------------------------------
        # This allows multiple tools to be "correct"
        # and stabilizes comp-sum weights for soft costs.
        active_costs = tool_costs * validity_mask
        min_cost, _ = torch.min(
            active_costs + (1 - validity_mask) * 1e9,
            dim=1,
            keepdim=True,
        )
        centered_costs = active_costs - min_cost  # >= 0

        # ------------------------------------------------------------
        # 4) Comp-sum weights
        # w_j = sum_{k!=j} c_k - m + 2
        # ------------------------------------------------------------
        sum_costs = centered_costs.sum(dim=1, keepdim=True)
        w = (sum_costs - centered_costs) - m_eff + 2.0

        # ------------------------------------------------------------
        # 5) Surrogate Psi(pi)
        # ------------------------------------------------------------
        if self.surrogate_type == "logistic":
            psi = -torch.log(pi + self.eps)
        elif self.surrogate_type == "mae":
            psi = 1.0 - pi
        else:
            raise ValueError(f"Unknown surrogate: {self.surrogate_type}")

        # ------------------------------------------------------------
        # 6) Aggregate comp-sum loss
        # ------------------------------------------------------------
        loss_per_sample = (w * psi * validity_mask).sum(dim=1)
        loss_main = loss_per_sample.mean()

        # ------------------------------------------------------------
        # 7) Entropy regularization (panel-normalized)
        # ------------------------------------------------------------
        log_pi = torch.log(pi + self.eps)
        entropy = -(pi * log_pi).sum(dim=1)
        loss_entropy = -self.lambda_entropy * entropy.mean()

        total_loss = loss_main + loss_entropy

        return total_loss, {
            "loss_main": loss_main.item(),
            "loss_entropy": loss_entropy.item(),
            "avg_panel_size": m_eff.mean().item(),
            "avg_min_cost": min_cost.mean().item(),
        }

from typing import Tuple

class DySTANceLoss_v2(nn.Module):
    """
    Population comp-sum surrogate loss for DySTANce routing.

    Improvements vs. earlier version:
      - explicit renormalization of pi over valid tools
      - entropy computed only over valid tools, optionally normalized by log(panel_size)
      - robust handling of degenerate panels
      - preserves original comp-sum formulation but applies per-sample cost-centering
        (helps allow multiple near-optimal tools)
    """

    def __init__(
        self,
        surrogate_type: str = "logistic",   # "logistic" or "mae"
        lambda_entropy: float = 0.05,
        entropy_normalize_by_log: bool = True,  # normalize entropy by log(m_eff)
        eps: float = 1e-8,
    ):
        super().__init__()
        self.surrogate_type = surrogate_type
        self.lambda_entropy = lambda_entropy
        self.eps = eps
        self.entropy_normalize_by_log = entropy_normalize_by_log

    def forward(
        self,
        router_logits: torch.Tensor,  # [B, M]
        tool_costs: torch.Tensor,     # [B, M] in [0,1] (lower better)
        validity_mask: torch.Tensor,  # [B, M] in {0,1}
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar tensor
            info: dict of python floats for logging
        """
        B, M = router_logits.shape
        device = router_logits.device

        # ----------------------------
        # 1) Masked softmax -> pi over valid tools
        # ----------------------------
        very_neg = -1e9
        masked_logits = router_logits.masked_fill(validity_mask == 0, very_neg)
        pi = F.softmax(masked_logits, dim=1)  # [B, M]

        # Explicitly zero-out invalid slots and renormalize to avoid numerical leakage
        pi = pi * validity_mask
        row_sums = pi.sum(dim=1, keepdim=True)
        # If a row has sum==0 (no valid tools), keep a uniform small mass on valid_mask (degenerate).
        # But usually row_sums>0; add eps to avoid div by zero.
        pi = pi / (row_sums + self.eps)

        # ----------------------------
        # 2) Effective panel size per sample
        # ----------------------------
        m_eff = validity_mask.sum(dim=1, keepdim=True)  # [B,1]
        # Prevent degenerate panels from causing NaNs later; but we also log this condition.
        m_eff_clamped = torch.clamp(m_eff, min=1.0)

        # ----------------------------
        # 3) Cost centering (subtract per-sample min among active tools)
        # ----------------------------
        # Zero out invalid costs, then set invalid positions to large +ve so min ignores them.
        big = 1e9
        active_costs = tool_costs * validity_mask  # zeros at invalid positions
        # prepare for min: invalid -> +big so min picks among actual actives
        costs_for_min = active_costs + (1.0 - validity_mask) * big
        min_cost, _ = torch.min(costs_for_min, dim=1, keepdim=True)  # [B,1], min among actives
        # In degenerate rows (no actives) min_cost will be big; clamp to zero for safety
        min_cost = torch.where(min_cost > big / 2.0, torch.zeros_like(min_cost), min_cost)

        centered_costs = active_costs - min_cost  # now >= 0 (for active positions), invalid remain 0

        # Optional: you could also scale by (max-min) to keep ranges bounded, but centering suffices.
        # ----------------------------
        # 4) Comp-sum weights w_j = sum_{k!=j} c_k - m + 2
        # ----------------------------
        sum_centered = centered_costs.sum(dim=1, keepdim=True)  # [B,1]
        w = (sum_centered - centered_costs) - m_eff_clamped + 2.0  # [B,M]
        # w for invalid entries will be masked out downstream

        # ----------------------------
        # 5) Surrogate Psi(pi)
        # ----------------------------
        if self.surrogate_type == "logistic":
            psi = -torch.log(pi + self.eps)
        elif self.surrogate_type == "mae":
            psi = 1.0 - pi
        else:
            raise ValueError(f"Unknown surrogate_type={self.surrogate_type}")

        # ----------------------------
        # 6) Aggregate comp-sum loss (mask invalid tools)
        #    L_i = sum_j w_ij * Psi(pi_ij)  over valid j
        # ----------------------------
        element = w * psi * validity_mask
        loss_per_sample = element.sum(dim=1)  # [B]
        loss_main = loss_per_sample.mean()

        # ----------------------------
        # 7) Entropy regularization (computed over valid tools only)
        #    You can normalize entropy per-sample by either m_eff or log(m_eff).
        # ----------------------------
        # Compute entropy only on valid tools
        log_pi = torch.log(pi + self.eps)
        entropy_per_row = -(pi * log_pi * validity_mask).sum(dim=1, keepdim=True)  # [B,1]

        if self.entropy_normalize_by_log:
            # Normalize by log(m_eff) to get a value approx in [0,1] (if m_eff>=2)
            # When m_eff==1, log(1)=0 -> avoid dividing by 0; we set denom to 1 in that case.
            denom = torch.log(m_eff_clamped + 1e-8)
            denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
            entropy_norm = entropy_per_row / (denom + self.eps)
        else:
            # simple divide by panel size
            entropy_norm = entropy_per_row / (m_eff_clamped + self.eps)

        loss_entropy = -self.lambda_entropy * entropy_norm.mean()

        total_loss = loss_main + loss_entropy

        info = {
            "loss_main": float(loss_main.detach().cpu().item()),
            "loss_entropy": float(loss_entropy.detach().cpu().item()),
            "avg_panel_size": float(m_eff.mean().detach().cpu().item()),
            "avg_min_cost": float(min_cost.mean().detach().cpu().item()),
        }

        return total_loss, info