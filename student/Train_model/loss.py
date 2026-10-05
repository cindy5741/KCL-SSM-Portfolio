import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class CLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(CLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cos_sim = nn.CosineSimilarity(dim=-1) # the similarity function
        
    def forward(self, features, labels=None, mask=None, teacher_features=None):
        """
        features: [bsz, n_views, emb_dim] (e.g., [bsz, 2, emb_dim] for student's two augmented views)
        labels: [bsz] (optional, for supervised CL)
        mask: [bsz, bsz] (optional)
        teacher_features: [bsz, emb_dim] (optional, standard question features from teacher)
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 
        #  KDCL Logic: Add Teacher Feature as a new View
        # 
        contrast_count_student = features.shape[1] # usually 2 (f1, f2)
        
        if teacher_features is not None:
            # 1. Prepare all features: [f1, f2, f_teacher] -> [bsz, 3, emb_dim]
            teacher_features_unsqueezed = teacher_features.unsqueeze(1) 
            all_features = torch.cat([features, teacher_features_unsqueezed], dim=1)
            
            contrast_count_total = all_features.shape[1] # 3
            contrast_feature = torch.cat(torch.unbind(all_features, dim=1), dim=0) # [bsz*3, emb_dim]
            
            # 2. Build the new full mask: [bsz*3, bsz*3]
            # All blocks use the label mask
            row1 = torch.cat([mask, mask, mask], dim=1) # [bsz, 3*bsz]
            full_mask = torch.cat([row1, row1, row1], dim=0) # [3*bsz, 3*bsz]
            mask = full_mask
            
        else:
            # Original CL logic
            contrast_count_total = contrast_count_student
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # 
        #  End of KDCL Logic
        # 

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count_total # 2 or 3
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask (already tiled if teacher_features is used)
        if teacher_features is None:
            mask = mask.repeat(anchor_count, contrast_count_total)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class StudentLoss(nn.Module):
    """Student loss that matches Eq.(6)(7)(8) in the paper.

    Given:
      - z_i^s : student embedding of query i (one embedding per sample)
      - z_ci^s: student embedding of the standard question of class c_i
      - z_ci^t: teacher embedding of the standard question of class c_i
      - z_ci  : mixed standard-question embedding per Eq.(7)
      - {z_k^t}: teacher embeddings of ALL standard questions (bank), size K (typically 250)

    Loss per sample i (Eq.6):
      L_student(i) = -log( ( sum_{j in P(i)} exp(sim(z_i^s, z_j^s)/tau)
                             + alpha * exp(sim(z_i^s, z_ci)/tau)
                             + beta  * sum_{k=1..K} exp(sim(z_i^s, z_k^t)/tau)
                           )
                           /
                           ( sum_{z in A(i)} exp(sim(z_i^s, z)/tau) )
                         )

    And final loss (Eq.8):
      L_student = (1/N) * sum_i L_student(i)

    Notes:
      - Here we set A(i) = { z_j^s (j != i, in-batch) } U { z_ci } U { z_k^t (bank) }.
      - P(i) are in-batch positives: same label as i and j != i.
    """

    def __init__(self, temperature: float = 0.1, alpha: float = 1.0, beta: float = 1.0, lam: float = 0.5, eps: float = 1e-12):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)
        self.eps = float(eps)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)

    def mix_z_ci(self, z_ci_s: torch.Tensor, z_ci_t: torch.Tensor) -> torch.Tensor:
        """Eq.(7): z_ci = lam * z_ci^t + (1-lam) * z_ci^s."""
        z = self.lam * z_ci_t + (1.0 - self.lam) * z_ci_s
        return self._normalize(z)

    def forward(
        self,
        z_query_s: torch.Tensor,          # [N, D]
        labels: torch.Tensor,             # [N]
        z_ci: torch.Tensor,               # [N, D] mixed per-sample (Eq.7)
        teacher_bank: torch.Tensor        # [K, D]
    ) -> torch.Tensor:

        device = z_query_s.device
        N = z_query_s.size(0)

        # Normalize
        zq = self._normalize(z_query_s)
        zc = self._normalize(z_ci)
        zb = self._normalize(teacher_bank)

        # In-batch query-query similarities
        sim_q = torch.matmul(zq, zq.T) / self.temperature          # [N, N]
        exp_q = torch.exp(sim_q)

        # Masks
        eye = torch.eye(N, device=device, dtype=torch.bool)
        not_self = ~eye

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & not_self                 # [N, N]

        # Term 1: sum_{j in P(i)} exp(sim(z_i, z_j)/tau)
        num_pos = (exp_q * pos_mask.float()).sum(dim=1)            # [N]

        # Denominator query part: sum_{j != i} exp(sim(z_i, z_j)/tau)
        denom_q = (exp_q * not_self.float()).sum(dim=1)            # [N]

        # Term 2: exp(sim(z_i, z_ci)/tau)
        sim_ci = (zq * zc).sum(dim=1) / self.temperature           # [N]
        exp_ci = torch.exp(sim_ci)                                 # [N]
        zb = teacher_bank
        zb = zb.to(zq.device)
        # Term 3: sum_{k} exp(sim(z_i, z_k^t)/tau)
        sim_bank = torch.matmul(zq, zb.T) / self.temperature       # [N, K]
        exp_bank = torch.exp(sim_bank)
        bank_sum = exp_bank.sum(dim=1)                             # [N]

        numerator = num_pos + self.alpha * exp_ci + self.beta * bank_sum
        denominator = denom_q + exp_ci + bank_sum

        loss_i = -torch.log((numerator + self.eps) / (denominator + self.eps))
        return loss_i.mean()

