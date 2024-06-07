import torch

def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Cosine similarity scores.
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    return torch.einsum('ik,jk->ij', a_norm, b_norm)

def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot product between two tensors.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Dot product scores.
    """
    return torch.einsum('ik,jk->ij', a, b)

def late_interaction(q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
    """
    Computes the late interaction score between two tensors.

    Args:
        q_reps (torch.Tensor): Query representations (batch, seq_len, dim).
        p_reps (torch.Tensor): Passage representations (batch, seq_len, dim).

    Returns:
        torch.Tensor: Late interaction scores.
    """
    logits = torch.einsum('imk,jnk->ijmn', q_reps, p_reps)
    return logits.max(-1).values.sum(-1)

score_functions = {
    "cos": cos,
    "dot": dot,
    "late_interaction": late_interaction,
}
