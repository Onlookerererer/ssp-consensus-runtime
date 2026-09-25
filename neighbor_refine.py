"""Candidate-constrained Neighbor Refining V1; no labels enter its decisions."""
import json
import logging

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_neighbor_distribution(features, distribution, sample_indices, k, chunk_size=512):
    """Exact cosine Top-K once per epoch, with bounded pairwise-score storage."""
    features = F.normalize(features.detach(), dim=1)
    distribution = distribution.detach()
    n = features.shape[0]
    effective_k = min(k, max(n - 1, 0))
    support = torch.zeros_like(distribution)
    neighbors = torch.empty((n, effective_k), dtype=torch.long, device=features.device)
    if not effective_k:
        return support, neighbors
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        similarities = features[start:stop] @ features.T
        similarities.masked_fill_(sample_indices[start:stop, None] == sample_indices[None, :], -torch.inf)
        positions = similarities.topk(effective_k, dim=1).indices
        neighbors[start:stop] = sample_indices[positions]
        support[start:stop] = distribution[positions].mean(dim=1)
    return support, neighbors


@torch.no_grad()
def reweight_neighbor_distribution(img_features, txt_features, distribution, positions, chunk_size=512):
    """Disagreement-weighted neighborhood evidence on already selected edges only."""
    img_features = F.normalize(img_features.detach(), dim=1)
    txt_features = F.normalize(txt_features.detach(), dim=1)
    distribution = distribution.detach()
    support = torch.zeros_like(distribution)
    if positions.shape[1] == 0:
        return support
    for start in range(0, img_features.shape[0], chunk_size):
        stop = min(start + chunk_size, img_features.shape[0])
        selected = positions[start:stop]
        sim_img = (img_features[start:stop, None] * img_features[selected]).sum(dim=2).clamp(-1, 1)
        sim_txt = (txt_features[start:stop, None] * txt_features[selected]).sum(dim=2).clamp(-1, 1)
        reliability = (1 - 0.5 * (sim_img - sim_txt).abs()).clamp(0, 1)
        mass = reliability.sum(dim=1, keepdim=True)
        weights = reliability / mass.clamp_min(1e-8)
        neighbor_distribution = distribution[selected]
        weighted = (weights[:, :, None] * neighbor_distribution).sum(dim=1)
        # Degenerate rows retain the original KNN's uniform evidence, not zero mass.
        support[start:stop] = torch.where(
            mass > 1e-8, weighted, neighbor_distribution.mean(dim=1))
    return support


@torch.no_grad()
def compute_invariant_neighbor_distribution(img_features, txt_features, img_distribution,
                                            txt_distribution, sample_indices, k, chunk_size=512):
    """Modality-invariant neighborhood relation, with one Top-K per chunk."""
    img_features = F.normalize(img_features.detach(), dim=1)
    txt_features = F.normalize(txt_features.detach(), dim=1)
    img_distribution = img_distribution.detach()
    txt_distribution = txt_distribution.detach()
    n = img_features.shape[0]
    effective_k = min(k, max(n - 1, 0))
    img_support = torch.zeros_like(img_distribution)
    txt_support = torch.zeros_like(txt_distribution)
    neighbors = torch.empty((n, effective_k), dtype=torch.long, device=img_features.device)
    if not effective_k:
        return img_support, txt_support, neighbors
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sim_img = img_features[start:stop] @ img_features.T
        sim_txt = txt_features[start:stop] @ txt_features.T
        sim_inv = torch.minimum(sim_img, sim_txt)
        sim_inv.masked_fill_(sample_indices[start:stop, None] == sample_indices[None, :], -torch.inf)
        positions = sim_inv.topk(effective_k, dim=1).indices
        # Gather both compact distributions with the same positions; export global IDs only.
        img_support[start:stop] = img_distribution[positions].mean(dim=1)
        txt_support[start:stop] = txt_distribution[positions].mean(dim=1)
        neighbors[start:stop] = sample_indices[positions]
    return img_support, txt_support, neighbors


@torch.no_grad()
def compute_mean_neighbor_distribution(img_features, txt_features, img_distribution,
                                       txt_distribution, sample_indices, k, chunk_size=512):
    """Shared similarity mean ablation, with one Top-K per chunk."""
    img_features = F.normalize(img_features.detach(), dim=1)
    txt_features = F.normalize(txt_features.detach(), dim=1)
    img_distribution = img_distribution.detach()
    txt_distribution = txt_distribution.detach()
    n = img_features.shape[0]
    effective_k = min(k, max(n - 1, 0))
    img_support = torch.zeros_like(img_distribution)
    txt_support = torch.zeros_like(txt_distribution)
    neighbors = torch.empty((n, effective_k), dtype=torch.long, device=img_features.device)
    if not effective_k:
        return img_support, txt_support, neighbors
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sim_img = img_features[start:stop] @ img_features.T
        sim_txt = txt_features[start:stop] @ txt_features.T
        sim_mean = 0.5 * (sim_img + sim_txt)
        sim_mean.masked_fill_(sample_indices[start:stop, None] == sample_indices[None, :], -torch.inf)
        positions = sim_mean.topk(effective_k, dim=1).indices
        # Gather both compact distributions with the same positions; export global IDs only.
        img_support[start:stop] = img_distribution[positions].mean(dim=1)
        txt_support[start:stop] = txt_distribution[positions].mean(dim=1)
        neighbors[start:stop] = sample_indices[positions]
    return img_support, txt_support, neighbors


@torch.no_grad()
def compute_candidate_neighbor_distribution(g_img, g_txt, joint, margin_threshold,
                                            support_threshold, available, eps=1e-8):
    masked = (g_img.detach() + g_txt.detach()) * 0.5 * joint
    mass = masked.sum(dim=1)
    q = masked / (mass[:, None] + eps)
    top = q.topk(min(2, q.shape[1]), dim=1).values
    margin = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0)
    img_candidate = g_img.masked_fill(~joint, -torch.inf).argmax(dim=1)
    txt_candidate = g_txt.masked_fill(~joint, -torch.inf).argmax(dim=1)
    in_joint = joint.gather(1, img_candidate[:, None]).squeeze(1)
    agreement = (img_candidate == txt_candidate) & in_joint & available
    reliable = (agreement & (joint.sum(dim=1) > 1) & (mass > eps)
                & (mass >= support_threshold) & (margin >= margin_threshold))
    return q, mass, margin, agreement, reliable


@torch.no_grad()
def refine_semantic_evidence(u, q, reliable, beta):
    # Assign only selected rows: unreliable/singleton samples remain bitwise equal.
    refined = u.detach().clone()
    if beta and reliable.any():
        selected = u.detach()[reliable]
        refined[reliable] = (1 - beta) * selected + beta * selected.sum(dim=1, keepdim=True) * q[reliable]
    return refined


@torch.no_grad()
def permute_candidate_distribution(q_true, joint, shift=1):
    """Roll active values by the fixed +1 (right) or -1 (left); no RNG."""
    if shift not in (1, -1):
        raise ValueError('Only the fixed right +1 and left -1 controls are supported')
    q = q_true.detach()
    joint = joint.detach().bool()
    size = joint.sum(dim=1, keepdim=True)
    classes = q.shape[1]
    positions = torch.arange(classes, device=q.device).expand_as(joint)
    active_positions = positions.masked_fill(~joint, classes).sort(dim=1).values
    rank = joint.long().cumsum(dim=1) - 1
    source_rank = (rank - shift).remainder(size.clamp_min(1))
    source_index = active_positions.gather(1, source_rank).clamp_max(classes - 1)
    q_perm = torch.where(joint, q.gather(1, source_index), torch.zeros_like(q))

    original_sorted = q.sort(dim=1).values
    permuted_sorted = q_perm.sort(dim=1).values
    same_values = (original_sorted == permuted_sorted).all(dim=1)
    full_support = ((q > 0) == joint).all(dim=1) & ((q_perm > 0) == joint).all(dim=1)
    finite = torch.isfinite(q).all(dim=1) & torch.isfinite(q_perm).all(dim=1)
    sums = q.sum(dim=1)
    normalized = torch.isclose(sums, torch.ones_like(sums), atol=1e-7, rtol=1e-6)
    same_sum = torch.isclose(q_perm.sum(dim=1), sums, atol=1e-7, rtol=1e-6)
    unique_top = original_sorted[:, -1] > original_sorted[:, -2] if classes > 1 else torch.zeros_like(sums, dtype=torch.bool)
    changed_top = q_perm.argmax(dim=1) != q.argmax(dim=1)
    valid = ((size.squeeze(1) > 1) & same_values & full_support & finite & normalized
             & same_sum & unique_top & changed_top)
    return q_perm, valid


@torch.no_grad()
def simulate_history_update(old_state, evidence, mask, ema_decay, update_strength, eps=1e-8):
    """Diagnostic-only copy of SSP's exact per-modal state update; no real writes."""
    old = old_state.detach().clone()
    transition = evidence.detach().clone()
    update_weight = update_strength * (1 - ema_decay)
    new = ema_decay * old + update_weight * transition
    normalizer = new.sum(dim=1, keepdim=True).clamp(min=eps)
    return torch.where(mask.detach(), new / normalizer, old)


class NeighborRefiner:
    def __init__(self, num_samples, configs):
        self.num_samples = num_samples
        self.mode = getattr(configs, 'neighbor_mode', None) or 'consensus'
        if self.mode not in ('consensus', 'causal_consensus', 'causal_shared', 'causal_invariant', 'similarity_mean', 'causal_reweight', 'disagree', 'agree', 'uniform_q', 'permuted_q', 'permuted_q_left'):
            raise ValueError('off must not instantiate a refiner')
        self.k = configs.neighbor_k
        self.beta = configs.neighbor_beta
        self.margin_threshold = configs.neighbor_margin
        self.support_threshold = configs.neighbor_support_threshold
        if self.k < 1 or not 0 <= self.beta <= 1:
            raise ValueError('neighbor_k must be positive and neighbor_beta must be in [0, 1]')
        if not 0 <= self.margin_threshold <= 1 or not 0 <= self.support_threshold <= 1:
            raise ValueError('Neighbor thresholds must be in [0, 1]')
        self.bank = None
        self.support = None
        self.available = None
        self.last = None
        self.last_next = None
        self.last_gate = None
        self.last_changed = None
        self.last_joint = None
        self.last_old_top = None

    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.pending = None
        self.last = None
        self.last_next = None
        self.last_gate = None
        self.last_changed = None
        self.last_joint = None
        self.last_old_top = None
        self.stats = dict(epoch=epoch, total_train_samples=self.num_samples,
                          processed_samples=0, intersection_size_0=0,
                          intersection_size_1=0, ambiguous_samples=0,
                          evidence_samples=0, ambiguous_evidence_samples=0,
                          reliable_samples=0, agreement_samples=0,
                          ambiguous_agreement_samples=0, support_mass_sum=0.,
                          margin_sum=0., ambiguous_support_mass_sum=0.,
                          ambiguous_margin_sum=0., refine_count=0)
        self.stats.update(mode=self.mode, reliable_consensus=0,
                          survivor_agree_with_neighbor=0, survivor_disagree_with_neighbor=0,
                          survivor_agree_with_neighbor_image=0, survivor_agree_with_neighbor_text=0,
                          survivor_disagree_with_neighbor_image=0, survivor_disagree_with_neighbor_text=0,
                          image_refine_count=0, text_refine_count=0, both_modalities_refined=0,
                          only_image_refined=0, only_text_refined=0, no_refine_count=0,
                          next_diagnosed_samples=0)
        if self.mode in ('permuted_q', 'permuted_q_left'):
            self.stats.update(permuted_reliable_count=0, argmax_changed_count=0,
                              multiset_preserved_count=0, invalid_permutation_count=0)
        if self.mode == 'permuted_q_left':
            self.stats.update(reliable_attempt_count=0, joint_size_2_count=0, joint_size_ge3_count=0,
                              left_equals_right_mapping_count=0, left_differs_right_mapping_count=0)
        self.correction = {scope: {m: dict(A=0, B=0, C=0, D=0) for m in ('image', 'text')}
                           for scope in ('all_ambiguous', 'refined_ambiguous', 'refined_modality_ambiguous')}
        self.next_correction = {scope: {m: dict(A=0, B=0, C=0, D=0) for m in ('image', 'text')}
                                for scope in ('all_ambiguous', 'refined_ambiguous')}
        # V1.2: membership and GT correctness are diagnostics, never gate inputs.
        self.candidate_membership = {
            m: {scope: dict(old_top1_in_modal_candidate=0, old_top1_out_modal_candidate=0,
                            old_top1_in_joint_intersection=0, old_top1_out_joint_intersection=0)
                for scope in ('all_bank', 'ambiguous', 'reliable', 'disagree')}
            for m in ('image', 'text')}
        self.agree_survivor_correctness = {
            m: dict(agree_refine_survivor_correct=0, agree_refine_survivor_wrong=0)
            for m in ('image', 'text')}

    @torch.no_grad()
    def observe(self, index, img_feat, txt_feat, pred_img, pred_txt, img_mask, txt_mask):
        """Only training forwards call this; rows are addressed by dataset index."""
        values = [F.normalize(img_feat.detach(), dim=1), F.normalize(txt_feat.detach(), dim=1)]
        for pred, mask in ((pred_img, img_mask), (pred_txt, txt_mask)):
            evidence = pred.detach() * mask
            values.append(evidence / evidence.sum(dim=1, keepdim=True).clamp_min(1e-8))
        if self.pending is None:
            self.pending = dict(values=[v.new_zeros((self.num_samples, v.shape[1])) for v in values],
                                seen=torch.zeros(self.num_samples, dtype=torch.bool, device=index.device))
        if (index < 0).any() or (index >= self.num_samples).any():
            raise ValueError('Memory bank index outside training set')
        if index.unique().numel() != index.numel() or self.pending['seen'][index].any():
            raise ValueError('Duplicate training sample index within epoch')
        for dest, value in zip(self.pending['values'], values):
            if not torch.isfinite(value).all():
                raise ValueError('Non-finite training memory tensor')
            dest[index] = value
        self.pending['seen'][index] = True

    @torch.no_grad()
    def refine(self, u_img, u_txt, index, joint, old_img_state=None, old_txt_state=None):
        if self.mode in ('disagree', 'agree') and (old_img_state is None or old_txt_state is None):
            raise ValueError('agree/disagree requires both pre-write SSP history states')
        size = joint.sum(dim=1)
        ambiguous = size > 1
        permutation_valid = torch.zeros_like(ambiguous)
        self.stats['processed_samples'] += index.numel()
        self.stats['intersection_size_0'] += (size == 0).sum().item()
        self.stats['intersection_size_1'] += (size == 1).sum().item()
        self.stats['ambiguous_samples'] += ambiguous.sum().item()
        if self.support is None:
            reliable = torch.zeros_like(ambiguous)
            available = torch.zeros_like(ambiguous)
            select_img = select_txt = reliable
            q = torch.zeros_like(u_img)
            mass = margin = u_img.new_zeros(index.shape[0])
            q_update = joint.to(u_img.dtype) / size[:, None].clamp_min(1) if self.mode == 'uniform_q' else q
            neighbor_top = torch.full_like(index, -1)
            refined_img, refined_txt = u_img, u_txt
        else:
            available = self.available[index]
            q, mass, margin, agreement, reliable = compute_candidate_neighbor_distribution(
                self.support[0][index], self.support[1][index], joint,
                self.margin_threshold, self.support_threshold, available)
            # On reliable rows both modality argmaxes, and q.argmax, coincide.
            neighbor_top = self.support[0][index].masked_fill(~joint, -torch.inf).argmax(dim=1)
            select_img = select_txt = reliable
            if old_img_state is not None and old_txt_state is not None:
                agree_img = reliable & (old_img_state.detach().argmax(dim=1) == neighbor_top)
                agree_txt = reliable & (old_txt_state.detach().argmax(dim=1) == neighbor_top)
                disagree_img, disagree_txt = reliable & ~agree_img, reliable & ~agree_txt
                self.stats['survivor_agree_with_neighbor'] += (agree_img & agree_txt).sum().item()
                self.stats['survivor_disagree_with_neighbor'] += (disagree_img | disagree_txt).sum().item()
                for modality, agree_m, disagree_m in (('image', agree_img, disagree_img),
                                                       ('text', agree_txt, disagree_txt)):
                    self.stats['survivor_agree_with_neighbor_' + modality] += agree_m.sum().item()
                    self.stats['survivor_disagree_with_neighbor_' + modality] += disagree_m.sum().item()
                if self.mode == 'disagree':
                    select_img, select_txt = disagree_img, disagree_txt
                elif self.mode == 'agree':
                    select_img, select_txt = agree_img, agree_txt
            # V1.5: true KNN q above still controls the gate; only the update target changes.
            q_update = joint.to(q.dtype) / size[:, None].clamp_min(1) if self.mode == 'uniform_q' else q
            if self.mode in ('permuted_q', 'permuted_q_left'):
                # V1.6: keep the true gate; only permute probability-to-candidate mapping.
                shift = -1 if self.mode == 'permuted_q_left' else 1
                q_update, permutation_valid = permute_candidate_distribution(q, joint, shift=shift)
                changed_top = q_update.argmax(dim=1) != q.argmax(dim=1)
                same_values = (q_update.sort(dim=1).values == q.sort(dim=1).values).all(dim=1)
                self.stats['permuted_reliable_count'] += reliable.sum().item()
                self.stats['argmax_changed_count'] += (reliable & changed_top).sum().item()
                self.stats['multiset_preserved_count'] += (reliable & same_values).sum().item()
                self.stats['invalid_permutation_count'] += (reliable & ~permutation_valid).sum().item()
                if self.mode == 'permuted_q_left':
                    # V1.7: compare both fixed mappings on this same current q/J, for logging only.
                    q_right, _ = permute_candidate_distribution(q, joint)
                    differs = (q_update != q_right).any(dim=1)
                    self.stats['reliable_attempt_count'] += reliable.sum().item()
                    self.stats['joint_size_2_count'] += (reliable & (size == 2)).sum().item()
                    self.stats['joint_size_ge3_count'] += (reliable & (size >= 3)).sum().item()
                    self.stats['left_equals_right_mapping_count'] += (reliable & ~differs).sum().item()
                    self.stats['left_differs_right_mapping_count'] += (reliable & differs).sum().item()
                select_img = select_txt = reliable & permutation_valid
            refined_img = refine_semantic_evidence(u_img, q_update, select_img, self.beta)
            refined_txt = refine_semantic_evidence(u_txt, q_update, select_txt, self.beta)
            amb_available = ambiguous & available
            self.stats['evidence_samples'] += available.sum().item()
            self.stats['ambiguous_evidence_samples'] += amb_available.sum().item()
            self.stats['agreement_samples'] += agreement.sum().item()
            self.stats['ambiguous_agreement_samples'] += (agreement & ambiguous).sum().item()
            self.stats['support_mass_sum'] += mass[available].sum().item()
            self.stats['margin_sum'] += margin[available].sum().item()
            self.stats['ambiguous_support_mass_sum'] += mass[amb_available].sum().item()
            self.stats['ambiguous_margin_sum'] += margin[amb_available].sum().item()
        changed_img = (refined_img != u_img).any(dim=1)
        changed_txt = (refined_txt != u_txt).any(dim=1)
        changed = changed_img | changed_txt
        self.stats['reliable_samples'] += reliable.sum().item()
        self.stats['reliable_consensus'] += reliable.sum().item()
        self.stats['refine_count'] += changed.sum().item()
        self.stats['image_refine_count'] += changed_img.sum().item()
        self.stats['text_refine_count'] += changed_txt.sum().item()
        self.stats['both_modalities_refined'] += (changed_img & changed_txt).sum().item()
        self.stats['only_image_refined'] += (changed_img & ~changed_txt).sum().item()
        self.stats['only_text_refined'] += (changed_txt & ~changed_img).sum().item()
        self.stats['no_refine_count'] += (~changed).sum().item()
        self.last = (u_img.detach(), u_txt.detach(), refined_img, refined_txt, ambiguous, changed)
        self.last_changed = (changed_img, changed_txt)
        self.last_joint = joint.detach()
        self.last_gate = dict(q=q, reliable=reliable, image_selected=select_img,
                              text_selected=select_txt, neighbor_top1=neighbor_top,
                              available=available, q_true=q, q_update=q_update,
                              support_mass=mass, margin=margin)
        if self.mode in ('permuted_q', 'permuted_q_left'):
            self.last_gate['permutation_valid'] = permutation_valid
        return refined_img, refined_txt

    @torch.no_grad()
    def record_old_survivor_diagnostics(self, old_img_state, old_txt_state, img_mask, txt_mask):
        """Read-only membership of raw old-state argmax; no remasking of history."""
        available = self.last_gate['available']
        reliable = self.last_gate['reliable']
        neighbor_top = self.last_gate['neighbor_top1']
        ambiguous = self.last[4]
        for modality, old, modal_mask in (('image', old_img_state, img_mask),
                                          ('text', old_txt_state, txt_mask)):
            top = old.detach().argmax(dim=1)
            in_modal = modal_mask.gather(1, top[:, None]).squeeze(1).bool()
            in_joint = self.last_joint.gather(1, top[:, None]).squeeze(1).bool()
            scopes = dict(all_bank=available, ambiguous=available & ambiguous,
                          reliable=reliable, disagree=reliable & (top != neighbor_top))
            for scope, select in scopes.items():
                counts = self.candidate_membership[modality][scope]
                counts['old_top1_in_modal_candidate'] += (select & in_modal).sum().item()
                counts['old_top1_out_modal_candidate'] += (select & ~in_modal).sum().item()
                counts['old_top1_in_joint_intersection'] += (select & in_joint).sum().item()
                counts['old_top1_out_joint_intersection'] += (select & ~in_joint).sum().item()

    @torch.no_grad()
    def stage_next_diagnosis(self, u_img, u_txt, refined_img, refined_txt,
                            old_img_state, old_txt_state, img_mask, txt_mask,
                            ema_decay, update_strength, eps=1e-8):
        """Keep top-1 only; all cloned temporary histories are discarded immediately."""
        self.last_old_top = (old_img_state.detach().argmax(dim=1), old_txt_state.detach().argmax(dim=1))
        self.record_old_survivor_diagnostics(old_img_state, old_txt_state, img_mask, txt_mask)
        next_top = []
        for old, u, refined, mask in ((old_img_state, u_img, refined_img, img_mask),
                                      (old_txt_state, u_txt, refined_txt, txt_mask)):
            original_next = simulate_history_update(old, u, mask, ema_decay, update_strength, eps)
            refined_next = simulate_history_update(old, refined, mask, ema_decay, update_strength, eps)
            next_top.append((original_next.argmax(dim=1), refined_next.argmax(dim=1)))
        self.last_next = tuple(next_top)

    @torch.no_grad()
    def diagnose(self, labels):
        """Logging only, after refinement; labels are neither retained nor returned."""
        ui, ut, ri, rt, ambiguous, changed = self.last
        for modality, before, after, changed_m in (('image', ui, ri, self.last_changed[0]),
                                                  ('text', ut, rt, self.last_changed[1])):
            old_ok = labels.gather(1, before.argmax(dim=1, keepdim=True)).squeeze(1) > 0
            new_ok = labels.gather(1, after.argmax(dim=1, keepdim=True)).squeeze(1) > 0
            for scope, select in (('all_ambiguous', ambiguous), ('refined_ambiguous', ambiguous & changed),
                                  ('refined_modality_ambiguous', ambiguous & changed_m)):
                counts = self.correction[scope][modality]
                for key, mask in (('A', old_ok & new_ok), ('B', ~old_ok & new_ok),
                                  ('C', old_ok & ~new_ok), ('D', ~old_ok & ~new_ok)):
                    counts[key] += (mask & select).sum().item()
        if self.last_next is not None:
            self.stats['next_diagnosed_samples'] += labels.shape[0]
            for modality, (before, after), changed_m in zip(('image', 'text'), self.last_next, self.last_changed):
                old_ok = labels.gather(1, before[:, None]).squeeze(1) > 0
                new_ok = labels.gather(1, after[:, None]).squeeze(1) > 0
                for scope, select in (('all_ambiguous', ambiguous), ('refined_ambiguous', ambiguous & changed_m)):
                    counts = self.next_correction[scope][modality]
                    for key, mask in (('A', old_ok & new_ok), ('B', ~old_ok & new_ok),
                                      ('C', old_ok & ~new_ok), ('D', ~old_ok & ~new_ok)):
                        counts[key] += (mask & select).sum().item()
        if self.mode == 'agree' and self.last_old_top is not None:
            for modality, old_top, changed_m in zip(('image', 'text'), self.last_old_top, self.last_changed):
                old_correct = labels.gather(1, old_top[:, None]).squeeze(1) > 0
                counts = self.agree_survivor_correctness[modality]
                counts['agree_refine_survivor_correct'] += (changed_m & old_correct).sum().item()
                counts['agree_refine_survivor_wrong'] += (changed_m & ~old_correct).sum().item()
        self.last = None
        self.last_next = None
        self.last_gate = None
        self.last_changed = None
        self.last_joint = None
        self.last_old_top = None

    @torch.no_grad()
    def end_epoch(self):
        report = dict(self.stats)
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None
        if self.mode == 'permuted_q_left':
            attempted = report['reliable_attempt_count']
            report['joint_size_2_ratio'] = ratio(report['joint_size_2_count'], attempted)
            report['joint_size_ge3_ratio'] = ratio(report['joint_size_ge3_count'], attempted)
            report['left_differs_right_mapping_ratio'] = ratio(
                report['left_differs_right_mapping_count'], report['joint_size_ge3_count'])
            report['left_mapping_differs_from_right_count'] = report['left_differs_right_mapping_count']
            report['invalid_count'] = report['invalid_permutation_count']
        report['reliable_ratio'] = ratio(report['reliable_samples'], report['processed_samples'])
        report['ambiguous_reliable_ratio'] = ratio(report['reliable_samples'], report['ambiguous_samples'])
        report['image_refine_ratio'] = ratio(report['image_refine_count'], report['ambiguous_samples'])
        report['text_refine_ratio'] = ratio(report['text_refine_count'], report['ambiguous_samples'])
        report['refine_ratio'] = ratio(report['refine_count'], report['ambiguous_samples'])
        for modality in ('image', 'text'):
            report[modality + '_agree_count'] = report['survivor_agree_with_neighbor_' + modality]
            report[modality + '_disagree_count'] = report['survivor_disagree_with_neighbor_' + modality]
        for prefix, denom in (('', report['evidence_samples']),
                              ('ambiguous_', report['ambiguous_evidence_samples'])):
            report[prefix + 'average_candidate_support_mass'] = ratio(report[prefix + 'support_mass_sum'], denom)
            report[prefix + 'average_neighbor_margin'] = ratio(report[prefix + 'margin_sum'], denom)
            report[prefix + 'agreement_ratio'] = ratio(report[prefix + 'agreement_samples'], denom)
        for scope in self.correction.values():
            for counts in scope.values():
                counts['B_minus_C'] = counts['B'] - counts['C']
        report['correction'] = self.correction
        for scope in self.next_correction.values():
            for counts in scope.values():
                counts['B_minus_C'] = counts['B'] - counts['C']
                counts['B_over_C'] = ratio(counts['B'], counts['C'])
        report['next_survivor_correction'] = self.next_correction
        for modality in ('image', 'text'):
            for counts in self.candidate_membership[modality].values():
                total = counts['old_top1_in_modal_candidate'] + counts['old_top1_out_modal_candidate']
                counts['out_modal_candidate_ratio'] = ratio(counts['old_top1_out_modal_candidate'], total)
                counts['out_joint_intersection_ratio'] = ratio(counts['old_top1_out_joint_intersection'], total)
            counts = self.agree_survivor_correctness[modality]
            total = counts['agree_refine_survivor_correct'] + counts['agree_refine_survivor_wrong']
            counts['agree_refine_survivor_correct_ratio'] = ratio(counts['agree_refine_survivor_correct'], total)
        report['candidate_membership'] = self.candidate_membership
        report['agree_survivor_correctness'] = self.agree_survivor_correctness
        report['memory_epoch_used'] = self.epoch - 1 if self.bank is not None else None
        logging.info('[Neighbor Refining V1] %s', json.dumps(report, allow_nan=False))
        self.last_report = report
        if self.pending is None:
            self.bank = self.support = self.available = None
            return
        ids = self.pending['seen'].nonzero(as_tuple=True)[0]
        values = [v[ids].detach() for v in self.pending['values']]
        self.bank = dict(sample_index=ids, image_embedding=values[0], text_embedding=values[1],
                         image_distribution=values[2], text_distribution=values[3])
        if self.mode == 'causal_reweight':
            row_of_id = torch.full((self.num_samples,), -1, dtype=torch.long, device=ids.device)
            row_of_id[ids] = torch.arange(ids.numel(), device=ids.device)
            self.support = []
            for modality, features, distribution in (('image', values[0], values[2]), ('text', values[1], values[3])):
                _, neighbors = compute_neighbor_distribution(features, distribution, ids, self.k)
                positions = row_of_id[neighbors]
                g = reweight_neighbor_distribution(values[0], values[1], distribution, positions)
                full = distribution.new_zeros((self.num_samples, distribution.shape[1]))
                full[ids] = g
                self.support.append(full)
                self.bank[modality + '_neighbor_indices'] = neighbors
            self.available = self.pending['seen'].clone() & (ids.numel() > 1)
            self.pending = None
            return
        if self.mode == 'similarity_mean':
            img_support, txt_support, neighbors = compute_mean_neighbor_distribution(
                values[0], values[1], values[2], values[3], ids, self.k)
            self.bank['mean_neighbor_indices'] = neighbors
            self.support = []
            for g in (img_support, txt_support):
                full = g.new_zeros((self.num_samples, g.shape[1]))
                full[ids] = g
                self.support.append(full)
            self.available = self.pending['seen'].clone() & (ids.numel() > 1)
            self.pending = None
            return
        if self.mode == 'causal_invariant':
            img_support, txt_support, neighbors = compute_invariant_neighbor_distribution(
                values[0], values[1], values[2], values[3], ids, self.k)
            self.bank['invariant_neighbor_indices'] = neighbors
            self.support = []
            for g in (img_support, txt_support):
                full = g.new_zeros((self.num_samples, g.shape[1]))
                full[ids] = g
                self.support.append(full)
            self.available = self.pending['seen'].clone() & (ids.numel() > 1)
            self.pending = None
            return
        if self.mode == 'causal_shared':
            # Intervention-inspired deconfounding of neighborhood selection.
            shared_feat = F.normalize(0.5 * (values[0] + values[1]), p=2, dim=1)
            # Concatenate columns only: one KNN gathers both modalities with the same
            # compact positions, then averages each column without mixing modalities.
            paired_distributions = torch.cat((values[2], values[3]), dim=1)
            paired_support, neighbors = compute_neighbor_distribution(
                shared_feat, paired_distributions, ids, self.k)
            self.bank['shared_embedding'] = shared_feat
            self.bank['shared_neighbor_indices'] = neighbors  # Global sample IDs: ids[positions].
            self.support = []
            for g in paired_support.split(values[2].shape[1], dim=1):
                full = g.new_zeros((self.num_samples, g.shape[1]))
                full[ids] = g
                self.support.append(full)
            self.available = self.pending['seen'].clone() & (ids.numel() > 1)
            self.pending = None
            return
        self.support = []
        for modality, features, distribution in (('image', values[0], values[2]), ('text', values[1], values[3])):
            g, neighbors = compute_neighbor_distribution(features, distribution, ids, self.k)
            if self.mode == 'causal_consensus' and neighbors.shape[1] > 0:
                # Paired-modality intervention on neighborhood evidence: keep neighbor identity.
                # neighbors holds global sample IDs, as used by the full pending distribution bank.
                paired_distribution = self.pending['values'][3 if modality == 'image' else 2]
                paired_support = paired_distribution[neighbors].mean(dim=1)
                g = torch.sqrt(g * paired_support)
                # Zero-mass rows stay zero; the clamp prevents NaN/Inf without adding evidence.
                g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-8)
            full = distribution.new_zeros((self.num_samples, distribution.shape[1]))
            full[ids] = g
            self.support.append(full)
            self.bank[modality + '_neighbor_indices'] = neighbors
        self.available = self.pending['seen'].clone() & (ids.numel() > 1)
        self.pending = None
