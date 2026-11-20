"""Full-vocabulary, pointwise-clipped forward KL from Search-E1 Eq. (7)."""

import torch
from torch.utils.checkpoint import checkpoint


def clipped_forward_kl(student_logits, teacher_logits, clip=10.0, *, with_entropy=False):
    """Sum vocabulary contributions, then average policy token positions.

    Clip each P_teacher(v) * (log P_teacher(v) - log P_student(v)) ABOVE
    before summing over v. Negative contributions remain negative. Neither a
    top-k approximation nor a clamp on the summed token KL matches Eq. (7).
    FP32 normalization avoids bf16 underflow; teacher targets never get gradients.
    """
    if clip <= 0:
        raise ValueError("Pointwise clip must be positive")
    log_student = student_logits.float().log_softmax(dim=-1)
    log_teacher = teacher_logits.detach().float().log_softmax(dim=-1)
    contributions = log_teacher.exp() * (log_teacher - log_student)
    kl = contributions.clamp(max=clip).sum(dim=-1).mean()
    if not with_entropy:
        return kl
    with torch.no_grad():
        lp = log_student.detach()
        entropy = -(lp.exp() * lp.nan_to_num()).sum(dim=-1).mean()
    return kl, entropy


def hidden_forward_kl(student_hidden, teacher_hidden, lm_head, clip=10.0, chunk_size=32, *, with_entropy=False):
    """Project a few token positions at a time, keeping the entire vocabulary.

    Checkpoint each projection and loss together so autograd does not retain
    [sequence_length, vocabulary_size] logits across chunks. This changes memory
    use, not the objective. Qwen2.5's frozen, tied lm_head is shared by both sides.
    """
    total = student_hidden.new_zeros((), dtype=torch.float32)
    entropy = torch.zeros_like(total)
    count = student_hidden.shape[0]
    if count == 0 or chunk_size <= 0:
        raise ValueError("Need positive token count and chunk size")

    def chunk_loss(student, teacher):
        # Only student hidden states participate in differentiation.
        with torch.no_grad():
            teacher_logits = lm_head(teacher)
        return clipped_forward_kl(lm_head(student), teacher_logits, clip, with_entropy=with_entropy)

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        value = checkpoint(
            chunk_loss, student_hidden[start:end], teacher_hidden[start:end], use_reentrant=False
        )
        weight = (end - start) / count
        if with_entropy:
            value, chunk_entropy = value
            entropy = entropy + chunk_entropy * weight
        total = total + value * weight
    return (total, entropy) if with_entropy else total
