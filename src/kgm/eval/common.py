import torch


def get_rank(sim: torch.FloatTensor, true: torch.LongTensor) -> torch.FloatTensor:
    """Compute the rank, exploiting that there is only one true hit."""
    batch_size = true.shape[0]
    true_sim = sim[torch.arange(batch_size), true].unsqueeze(1)
    best_rank = torch.sum(sim > true_sim, dim=1, dtype=torch.long).float() + 1
    worst_rank = torch.sum(sim >= true_sim, dim=1, dtype=torch.long).float()
    return 0.5 * (best_rank + worst_rank)
