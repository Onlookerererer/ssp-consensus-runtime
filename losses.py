import torch
import torch.nn as nn
import torch.nn.functional as F

# Legacy Candidate-Disjoint Relation helpers; unused by SSP + Consensus.
def candidate_disjoint_mask(img_partial_mask, txt_partial_mask):
    """Reliable in-batch negatives from paired candidate intersections only."""
    joint = ((img_partial_mask > 0) & (txt_partial_mask > 0)).float()
    negative_mask = (joint @ joint.T) == 0
    negative_mask.fill_diagonal_(False)
    return negative_mask


def candidate_disjoint_relation_loss(img_feat, txt_feat, img_partial_mask,
                                     txt_partial_mask, margin=0.2):
    """Bidirectional hardest-negative hinge on existing L2-normalized embeddings."""
    negative_mask = candidate_disjoint_mask(img_partial_mask, txt_partial_mask)
    similarity = img_feat @ txt_feat.T
    positive = similarity.diagonal()
    negative_similarity = similarity.masked_fill(~negative_mask, -torch.inf)
    valid_i2t = negative_mask.any(dim=1)
    valid_t2i = negative_mask.any(dim=0)
    # Invalid rows use their positive score before the hinge, avoiding infinities.
    hardest_i2t = torch.where(valid_i2t, negative_similarity.max(dim=1).values, positive)
    hardest_t2i = torch.where(valid_t2i, negative_similarity.max(dim=0).values, positive)
    loss_i2t = (F.relu(margin + hardest_i2t - positive) * valid_i2t).sum()
    loss_t2i = (F.relu(margin + hardest_t2i - positive) * valid_t2i).sum()
    loss = (loss_i2t / valid_i2t.sum().clamp_min(1)
            + loss_t2i / valid_t2i.sum().clamp_min(1))
    return loss, valid_i2t.float().mean(), valid_t2i.float().mean()


class SSPLoss(nn.Module):
    """
    Semantic Survivor Principle (SSP) Loss with Noise Robustness
    Core Philosophy:
    Partial labels tell the model "where to look", but "what to find" is determined by 
    data accumulation over time through Markov chain dynamics.
    Enhanced with noise robustness for high partial label noise scenarios.
    (Removed confidence threshold filtering)
    """
    
    def __init__(self, partial_labels, ema_decay, img_partial_labels=None, txt_partial_labels=None):
        super().__init__()
        
        self.eps = 1e-8                 # Small constant for numerical stability
        self.ema_decay = ema_decay      # EMA decay for state count update
        
        # Process input tensors
        def to_tensor(x):
            if isinstance(x, torch.Tensor): return x.float().cuda()
            return torch.Tensor(x).float().cuda()

        self.global_partial = to_tensor(partial_labels) if partial_labels is not None else None
        self.img_partial = to_tensor(img_partial_labels)
        self.txt_partial = to_tensor(txt_partial_labels)

        self.num_samples, self.num_classes = self.img_partial.shape

        # Initialize state counts (S_0) with EMA support
        if self.global_partial is not None:
            init_img = (self.global_partial + self.img_partial) / 2.0
            init_txt = (self.global_partial + self.txt_partial) / 2.0
        else:
            init_img = self.img_partial.clone()
            init_txt = self.txt_partial.clone()
        
        self.register_buffer("mc_img_state_count", init_img)
        self.register_buffer("mc_txt_state_count", init_txt)
        self.register_buffer("update_strength", torch.ones(1))  # Dynamic update strength

        # Define valid state space (Mask) with noise robustness
        self.mc_img_mask = (self.img_partial > 0)
        self.mc_txt_mask = (self.txt_partial > 0)
        self.mc_joint_mask = self.mc_img_mask & self.mc_txt_mask 
        
        # Precompute normalized prior distribution
        self.img_prior_norm = self.img_partial / self.img_partial.sum(dim=1, keepdim=True).clamp(min=self.eps)
        self.txt_prior_norm = self.txt_partial / self.txt_partial.sum(dim=1, keepdim=True).clamp(min=self.eps)

        # Learnable temperature parameter with regularization
        self.tau_param = nn.Parameter(torch.tensor(0.))
        self.tau_reg = 1e-4  # Regularization for temperature

    def update_mc_state_count(self, pred_img, pred_txt, sample_index, epoch=None):
        """
        Update Markov chain state counts with noise robustness
        - EMA update to reduce noise accumulation
        - Dynamic update strength decay
        (Removed confidence threshold filtering)
        """
        
        batch_img_mask = self.mc_img_mask[sample_index]
        batch_txt_mask = self.mc_txt_mask[sample_index]

        # Get pure transition probabilities (only constrained by Mask)
        mc_img_trans = pred_img.detach() * batch_img_mask.float()
        mc_txt_trans = pred_txt.detach() * batch_txt_mask.float()

        # Neighbor Refining V1: calibrate evidence before the unchanged EMA update.
        neighbor = getattr(self, 'neighbor_refiner', None)
        if neighbor is not None:
            # V1.1: both gates and detached diagnosis start from pre-write history.
            old_img_state = self.mc_img_state_count[sample_index].detach().clone()
            old_txt_state = self.mc_txt_state_count[sample_index].detach().clone()
            original_img_trans, original_txt_trans = mc_img_trans, mc_txt_trans
            mc_img_trans, mc_txt_trans = neighbor.refine(
                mc_img_trans, mc_txt_trans, sample_index, self.mc_joint_mask[sample_index],
                old_img_state, old_txt_state)
            neighbor.stage_next_diagnosis(
                original_img_trans, original_txt_trans, mc_img_trans, mc_txt_trans,
                old_img_state, old_txt_state, batch_img_mask, batch_txt_mask,
                self.ema_decay, self.update_strength.item(), self.eps)

        # EMA update instead of pure accumulation (reduce noise impact)
        update_weight = self.update_strength.item() * (1 - self.ema_decay)
        current_img_state = self.mc_img_state_count[sample_index]
        current_txt_state = self.mc_txt_state_count[sample_index]
        
        # EMA: new_state = ema_decay * old_state + (1-ema_decay) * new_transition
        new_img_state = self.ema_decay * current_img_state + update_weight * mc_img_trans
        new_txt_state = self.ema_decay * current_txt_state + update_weight * mc_txt_trans

        # Normalization with mask preservation
        img_sum = new_img_state.sum(dim=1, keepdim=True).clamp(min=self.eps)
        txt_sum = new_txt_state.sum(dim=1, keepdim=True).clamp(min=self.eps)

        self.mc_img_state_count[sample_index] = torch.where(
            batch_img_mask,
            new_img_state / img_sum,
            current_img_state  # Keep original values outside mask
        )
        self.mc_txt_state_count[sample_index] = torch.where(
            batch_txt_mask,
            new_txt_state / txt_sum,
            current_txt_state
        )

    def get_mc_joint_stationary_dist(self, sample_index):
        """Get joint stationary distribution with noise robustness"""
        s_img = self.mc_img_state_count[sample_index]
        s_txt = self.mc_txt_state_count[sample_index]
        mask = self.mc_joint_mask[sample_index]

        # Pure geometric mean with small epsilon for numerical stability
        joint_dist = (s_img + self.eps) * (s_txt + self.eps)

        # Normalization
        joint_sum = joint_dist.sum(dim=1, keepdim=True).clamp(min=self.eps)
        joint_dist = torch.where(mask, joint_dist / joint_sum, joint_dist)
        
        return joint_dist

    def mc_intra_chain_loss(self, pred_img, pred_txt, sample_index):
        mc_joint_dist = self.get_mc_joint_stationary_dist(sample_index)
        if getattr(self, 'capture_trajectory_joint', False):
            self.trajectory_joint = mc_joint_dist.detach().clone()
        mask = self.mc_joint_mask[sample_index]

        smooth_mask = mask.float() * 0.9 + 0.1 / self.num_classes
        
        pred_img_clamped = torch.clamp(pred_img, min=self.eps, max=1.0)
        ce_img = - torch.pow(pred_img_clamped, 0.5)
        loss_img = mc_joint_dist * ce_img * smooth_mask
        loss_img = torch.sum(loss_img, dim=1)
        
        pred_txt_clamped = torch.clamp(pred_txt, min=self.eps, max=1.0)
        ce_txt = - torch.pow(pred_txt_clamped, 0.5)
        loss_txt = mc_joint_dist * ce_txt * smooth_mask
        loss_txt = torch.sum(loss_txt, dim=1)
        
        valid_samples = (mask.sum(dim=1) > 0).sum()
        if valid_samples == 0: return torch.tensor(0.0).cuda()
            
        return (loss_img.sum() + loss_txt.sum()) / valid_samples

    def mc_inter_chain_loss(self, img_feat, txt_feat, sample_index):
        # Temperature regularization to prevent collapse
        tau = 0.05 + 0.15 * torch.sigmoid(self.tau_param)
        tau_reg_loss = self.tau_reg * (self.tau_param ** 2)
        
        feat_sim = img_feat @ txt_feat.T
        logits = feat_sim / tau

        pred_sim = F.softmax(logits, dim=1)
        pred_sim_T = F.softmax(logits.T, dim=1)

        mc_joint_dist = self.get_mc_joint_stationary_dist(sample_index)
        semantic_sim = mc_joint_dist @ mc_joint_dist.T
        semantic_sim = semantic_sim / semantic_sim.sum(dim=1, keepdim=True).clamp(min=self.eps)
        
        pred_sim_clamped = torch.clamp(pred_sim, min=self.eps, max=1.0)
        ce_i2t = - torch.pow(pred_sim_clamped, 0.5)
        loss_i2t = semantic_sim * ce_i2t
        loss_i2t = torch.sum(loss_i2t, dim=1).mean()
        
        pred_sim_T_clamped = torch.clamp(pred_sim_T, min=self.eps, max=1.0)
        ce_t2i = - torch.pow(pred_sim_T_clamped, 0.5)
        loss_t2i = semantic_sim.T * ce_t2i
        loss_t2i = torch.sum(loss_t2i, dim=1).mean()
        
        return (loss_i2t + loss_t2i) * 0.5 + tau_reg_loss

    def forward(self, pred_img, pred_txt, sample_index, img_feat, txt_feat, configs, epoch=None):
        self.update_mc_state_count(pred_img, pred_txt, sample_index, epoch)
        mc_intra = self.mc_intra_chain_loss(pred_img, pred_txt, sample_index)
        mc_inter = self.mc_inter_chain_loss(img_feat, txt_feat, sample_index)
        lamda = configs.lamda if hasattr(configs, 'lamda') else 0.1
        
        # Dynamic lambda decay for late training (more aggressive without confidence threshold)
        if epoch is not None:
            lamda = lamda * (0.98 ** (epoch))  # Faster decay
        
        return mc_intra + lamda * mc_inter
