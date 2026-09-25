import torch
import time
import copy
import logging
import math
from pathlib import Path
import numpy as np
import random as rn
import torch.optim as optim
import torch.nn.functional as F
from model import CMNN_Compat, Embedding
from losses import SSPLoss
from load_data import get_loader
from evaluate import fx_calc_map_multilabel
from utils import get_training_args

def to_seed(seed=0):
    """Fix random seeds to ensure experimental reproducibility"""
    np.random.seed(seed)
    rn.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_logging(args):
    """Initialize logging system"""
    log_dir = Path(args.log_dir) if args.log_dir else (
        Path('results') / f'{args.neighbor_mode}_seed_{args.seed}'
        if args.neighbor_mode in ('causal_consensus', 'causal_shared', 'causal_invariant', 'similarity_mean', 'causal_reweight', 'disagree', 'agree', 'uniform_q', 'permuted_q', 'permuted_q_left') else
        Path('results') / f'seed_{args.seed}'
        if args.neighbor_refine else Path('results'))
    if getattr(args, 'independent_train_seed', False) and not args.log_dir:
        method = 'ssp' if args.neighbor_mode == 'off' else args.neighbor_mode
        log_dir = Path('results') / 'independent_seed' / f'{method}_seed_{args.seed}'
    if getattr(args, 'relation_loss', False) and not args.log_dir:
        log_dir = log_dir / 'candidate_disjoint_relation'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / (
        f"{args.dataset}_"
        f"PartialLength_{args.partial_length}_Lamda_{args.lamda}lamda_log.txt"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_path

def evaluate_on_test_set(model, emb, input_data_par, device, epoch):
    """Evaluate mAP on test set"""
    model.eval()
    emb.eval()
    
    with torch.no_grad():
        img_test = torch.tensor(input_data_par['img_test']).to(device)
        text_test = torch.tensor(input_data_par['text_test']).to(device)
        
        view1_feature, view2_feature = model(img_test, text_test)
        
        label = input_data_par['label_test']
        view1_feature = view1_feature.detach().cpu().numpy()
        view2_feature = view2_feature.detach().cpu().numpy()

        img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='cosine')
        txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='cosine')
        avg_map = (img_to_txt + txt_to_img) / 2.0 if (img_to_txt + txt_to_img) > 0 else 0.0
        
        logging.info(f"\n[Test Set Evaluation - Epoch {epoch}]")
        logging.info(f"    - Image to Text MAP = {img_to_txt:.6f}")
        logging.info(f"    - Text to Image MAP = {txt_to_img:.6f}")
        logging.info(f"    - Average MAP = {avg_map:.6f}")
    
    model.train()
    emb.train()
    return avg_map

def train_model(model, emb, data_loaders, input_data_par, optimizer, configs, device):
    time_start = time.time()
    env_enabled = getattr(configs, 'env_weight', 0.0) > 0
    trajectory = None
    equivalence = None
    if getattr(configs, 'trajectory_log', False) or getattr(configs, 'equivalence_capture', None):
        # Legacy diagnostics; disabled in the SSP + Consensus command.
        from laji.diagnostics.trajectory_logging_v5 import TrajectoryLogger, EquivalenceCapture, cpu_copy
        if configs.trajectory_log:
            trajectory = TrajectoryLogger(configs.trajectory_dir, configs.MAX_EPOCH,
                                          input_data_par, configs.anchor_supervision)
        if configs.equivalence_capture:
            equivalence = EquivalenceCapture(configs.equivalence_capture)
    anchor_enabled = getattr(configs, 'anchor_supervision', False)
    # Legacy Candidate-Disjoint Relation: disabled unless explicitly requested.
    relation_enabled = getattr(configs, 'relation_loss', False)
    relation_logged = False
    reference = None
    if getattr(configs, 'reference_mode', 'off') != 'off':
        from crossfit_guidance import ReferenceGuidance
        reference = ReferenceGuidance(
            configs.reference_cache, configs.reference_mode, configs.dataset, configs.partial_length,
            input_data_par['img_partial_label'], input_data_par['txt_partial_label'], device)
        logging.info('[CFSG] mode=%s weight=%.1f ambiguous_samples=%d cache=%s',
                     configs.reference_mode, configs.reference_weight,
                     int(reference.ambiguous.sum()), configs.reference_cache)
    anchor_epoch_stats = []
    if anchor_enabled:
        from laji.old_methods.anchor_supervision import anchor_cross_entropy, select_anchors
        anchor_mask, _ = select_anchors(
            torch.as_tensor(input_data_par['img_partial_label']),
            torch.as_tensor(input_data_par['txt_partial_label']))
        anchor_total = int(anchor_mask.sum().item())
    
    # Initialize loss function
    criterion = SSPLoss(
        partial_labels=input_data_par['label_train'],  
        ema_decay=configs.ema_decay,
        img_partial_labels=input_data_par.get('img_partial_label'),  
        txt_partial_labels=input_data_par.get('txt_partial_label')   
    ).cuda()

    # Neighbor Refining V1: only the training phase can collect or refine evidence.
    neighbor = None
    neighbor_mode = getattr(configs, 'neighbor_mode', None) or (
        'consensus' if getattr(configs, 'neighbor_refine', False) else 'off')
    if neighbor_mode != 'off':
        from neighbor_refine import NeighborRefiner
        neighbor = NeighborRefiner(criterion.num_samples, configs)
    
    # Training history records
    mAP_history = []
    epoch_loss_history = []
    best_avg_map = 0.0
    best_epoch = None
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(configs.MAX_EPOCH):
        if neighbor is not None:
            neighbor.begin_epoch(epoch)
        logging.info('\nEpoch {}/{}'.format(epoch, configs.MAX_EPOCH))
        logging.info('-' * 25)
        
        for phase in ['train', 'valid']:
            if neighbor is not None:
                criterion.neighbor_refiner = neighbor if phase == 'train' else None
            if trajectory is not None:
                criterion.capture_trajectory_joint = phase == 'train'
            running_loss = 0.0
            if env_enabled and phase == 'train':
                env_seen = env_base_sum = env_penalty_sum = env_weighted_sum = 0
                env_zero_reasons = {}
            if reference is not None and phase == 'train':
                guide_seen, guide_loss_sum = 0, 0.0
            if anchor_enabled and phase == 'train':
                anchor_seen = anchor_loss_sum = anchor_image_correct = anchor_text_correct = 0
            model.train() if phase == 'train' else model.eval()
            if trajectory is not None and phase == 'valid':
                trajectory.validation.begin(criterion, neighbor)
            
            for batch_data in data_loaders[phase]:
                imgs, txts, img_labels, txt_labels, ori_labels, index = batch_data
                if env_enabled and (not torch.isfinite(imgs).all() or not torch.isfinite(txts).all()):
                    raise FloatingPointError(f'Environment run: nonfinite input at epoch {epoch} {phase}')
                if reference is not None and (not torch.isfinite(imgs).all() or not torch.isfinite(txts).all()):
                    raise FloatingPointError(f'CFSG: nonfinite input at epoch {epoch} {phase}')
                
                if torch.isnan(imgs).any() or torch.isnan(txts).any():
                    logging.warning(f"Epoch {epoch} {phase}: Skipping batch with NaN values")
                    continue
                
                batch_size = imgs.size(0)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        img_labels = img_labels.cuda()
                        txt_labels = txt_labels.cuda()
                        ori_labels = ori_labels.cuda()
                        index = index.cuda()
                    
                    W = emb(torch.eye(configs.data_class).cuda())
                    view1_feature, view2_feature = model(imgs, txts)
                    
                    view1_flat = view1_feature.view(view1_feature.shape[0], -1)
                    view2_flat = view2_feature.view(view2_feature.shape[0], -1)
                    if anchor_enabled and phase == 'train':
                        image_logits = view1_flat.mm(W.T)
                        text_logits = view2_flat.mm(W.T)
                        view1_predict = F.softmax(image_logits, dim=1)
                        view2_predict = F.softmax(text_logits, dim=1)
                    else:
                        image_logits = view1_flat.mm(W.T)
                        text_logits = view2_flat.mm(W.T)
                        view1_predict = F.softmax(image_logits, dim=1)
                        view2_predict = F.softmax(text_logits, dim=1)

                    if neighbor is not None and phase == 'train':
                        neighbor.observe(index, view1_flat, view2_flat,
                                         view1_predict, view2_predict,
                                         criterion.mc_img_mask[index], criterion.mc_txt_mask[index])
                    if trajectory is not None and phase == 'train':
                        pre_i = cpu_copy(criterion.mc_img_state_count[index])
                        pre_t = cpu_copy(criterion.mc_txt_state_count[index])
                    
                    loss_base = criterion(
                        pred_img=view1_predict,      
                        pred_txt=view2_predict,      
                        sample_index=index,              
                        img_feat=view1_feature,      
                        txt_feat=view2_feature,      
                        configs=configs,
                        epoch=epoch        
                    )
                    loss = loss_base
                    if phase == 'train' and getattr(configs, 'env_weight', 0.0) > 0:
                        from environment_invariance import compute_environment_regularizer
                        joint = criterion.get_mc_joint_stationary_dist(index).detach().clone()
                        mask = criterion.mc_joint_mask[index].detach().clone()
                        penalty_env, env_info = compute_environment_regularizer(
                            image_logits, text_logits, view1_predict, view2_predict, joint, mask)
                        weighted_env = configs.env_weight * penalty_env
                        loss = loss_base + weighted_env
                        base_scalar = loss_base.detach().item()
                        penalty_scalar = penalty_env.detach().item()
                        weighted_scalar = weighted_env.detach().item()
                        if not all(math.isfinite(value) for value in
                                   (base_scalar, penalty_scalar, weighted_scalar)):
                            raise FloatingPointError(
                                f'Environment run: nonfinite loss at epoch {epoch}: '
                                f'base={base_scalar}, penalty={penalty_scalar}, weighted={weighted_scalar}')
                        env_n = env_info['n']
                        env_seen += env_n
                        env_base_sum += env_n * base_scalar
                        env_penalty_sum += env_n * penalty_scalar
                        env_weighted_sum += env_n * weighted_scalar
                        if env_info['reason'] != 'active':
                            reason = env_info['reason']
                            env_zero_reasons[reason] = env_zero_reasons.get(reason, 0) + 1
                            logging.info('[environment] n=%d rank=%d nullspace=%d reason=%s',
                                         env_info['n'], env_info['rank_A'],
                                         env_info['nullspace_dim'], env_info['reason'])
                    if reference is not None:
                        if not torch.isfinite(loss) or not torch.isfinite(view1_predict).all() or not torch.isfinite(view2_predict).all():
                            raise FloatingPointError(f'CFSG: nonfinite loss/prediction at epoch {epoch} {phase}')
                        if phase == 'train':
                            guide_loss, guide_count = reference.loss(view1_predict, view2_predict, index)
                            loss = loss + configs.reference_weight * guide_loss
                            if not torch.isfinite(loss):
                                raise FloatingPointError(f'CFSG: nonfinite total loss at epoch {epoch}')
                            guide_seen += guide_count
                            guide_loss_sum += guide_loss.detach().item() * guide_count
                    if relation_enabled and phase == 'train':
                        from losses import candidate_disjoint_relation_loss
                        relation_loss, valid_i2t, valid_t2i = candidate_disjoint_relation_loss(
                            view1_flat, view2_flat, img_labels, txt_labels,
                            margin=configs.relation_margin)
                        loss = loss + configs.relation_weight * relation_loss
                        if not relation_logged:
                            logging.info('    - [relation] loss=%.6f  valid_i2t_ratio=%.6f  valid_t2i_ratio=%.6f',
                                         relation_loss.detach().item(), valid_i2t.item(), valid_t2i.item())
                            relation_logged = True
                    if anchor_enabled and phase == 'train':
                        anchor_loss, count, correct_image, correct_text = anchor_cross_entropy(
                            image_logits, text_logits, img_labels, txt_labels)
                        loss = loss + configs.anchor_lambda * anchor_loss
                        anchor_seen += count
                        anchor_loss_sum += float(anchor_loss.detach()) * count
                        anchor_image_correct += correct_image
                        anchor_text_correct += correct_text
                    
                    if env_enabled and not torch.isfinite(loss.detach()).all():
                        raise FloatingPointError(f'Environment run: nonfinite total loss at epoch {epoch} {phase}')
                    if phase == 'train':
                        if trajectory is not None:
                            trajectory.record(epoch, index, view1_predict, view2_predict,
                                              neighbor, pre_i, pre_t,
                                              criterion.mc_img_state_count[index],
                                              criterion.mc_txt_state_count[index],
                                              criterion.trajectory_joint)
                        if equivalence is not None:
                            smoke_q = cpu_copy(neighbor.last_gate['q'])
                            smoke_reliable = cpu_copy(neighbor.last_gate['reliable'])
                        if neighbor is not None and reference is None:
                            neighbor.diagnose(ori_labels)
                        loss.backward()
                        optimizer.step()
                        if equivalence is not None:
                            equivalence.batch(loss, criterion, smoke_q, smoke_reliable)
                
                running_loss += loss.item() * batch_size
            
            dataset_size = len(data_loaders[phase].dataset)
            epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
            if env_enabled and phase == 'train':
                logging.info(
                    '[environment epoch] epoch=%d valid_samples=%d loss_base_mean=%.12g '
                    'P_env_mean=%.12g weighted_penalty_mean=%.12g zero_batches=%d reasons=%s',
                    epoch, env_seen, env_base_sum / env_seen if env_seen else 0.0,
                    env_penalty_sum / env_seen if env_seen else 0.0,
                    env_weighted_sum / env_seen if env_seen else 0.0,
                    sum(env_zero_reasons.values()), env_zero_reasons)
            if reference is not None and phase == 'train':
                logging.info('    - [CFSG] mode=%s seen=%d loss_guide_mean=%.8f weight=%.1f',
                             configs.reference_mode, guide_seen,
                             guide_loss_sum / guide_seen if guide_seen else 0.0, configs.reference_weight)
            if anchor_enabled and phase == 'train':
                if anchor_seen != anchor_total:
                    raise AssertionError(f'anchor_seen={anchor_seen}, expected={anchor_total}')
                anchor_mean_loss = anchor_loss_sum / anchor_seen if anchor_seen else 0.0
                anchor_stats = dict(epoch=epoch, anchor_seen_count=anchor_seen,
                                    anchor_loss_mean=anchor_mean_loss,
                                    image_raw_accuracy=anchor_image_correct / anchor_seen if anchor_seen else 0.0,
                                    text_raw_accuracy=anchor_text_correct / anchor_seen if anchor_seen else 0.0)
                anchor_epoch_stats.append(anchor_stats)
                logging.info('    - [anchor] seen=%d  CE=%.6f  raw_acc_I=%.6f  raw_acc_T=%.6f',
                             anchor_seen, anchor_mean_loss,
                             anchor_stats['image_raw_accuracy'], anchor_stats['text_raw_accuracy'])
            
            if phase == 'train':
                if neighbor is not None:
                    # Publish before validation, using only training-forward tensors.
                    neighbor.end_epoch()
                if equivalence is not None:
                    equivalence.epoch_bank(neighbor)
                    equivalence.epochs[-1]['train_epoch_loss'] = epoch_loss
                start_time = time.time()
                model.eval()
                t_imgs, t_txts, t_labels = [], [], []
                with torch.no_grad():
                    for batch_data in data_loaders['valid']:
                        imgs, txts, _, _, ori_labels, _ = batch_data
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                        
                        t_view1_feature, t_view2_feature = model(imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(ori_labels.cpu().numpy())
                
                t_imgs = np.concatenate(t_imgs) if t_imgs else np.array([])
                t_txts = np.concatenate(t_txts) if t_txts else np.array([])
                t_labels = np.concatenate(t_labels) if t_labels else np.array([])
                
                img2txt = fx_calc_map_multilabel(t_imgs, t_txts, t_labels, metric='cosine') if len(t_imgs) > 0 else 0.0
                txt2img = fx_calc_map_multilabel(t_txts, t_imgs, t_labels, metric='cosine') if len(t_txts) > 0 else 0.0
                avg_map = (img2txt + txt2img) / 2. if (img2txt + txt2img) > 0 else 0.0
                mAP_history.append(avg_map)
                if equivalence is not None:
                    equivalence.validation(avg_map)
                model.train()
                end_time = time.time()
                print("time per batch: ", end_time - start_time)
            
            lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            if phase == 'train':
                logging.info(f"    - [{phase:<5}] Loss: {epoch_loss:>6.4f}  Img2Txt: {img2txt:>6.4f}  Txt2Img: {txt2img:>6.4f}  Lr: {lr:>6g}")
            else:
                logging.info(f"    - [{phase:<5}] Loss: {epoch_loss:>6.4f}  Avg mAP: {avg_map:>6.4f}")
                epoch_loss_history.append(epoch_loss)
                if equivalence is not None:
                    equivalence.epochs[-1]['valid_epoch_loss'] = epoch_loss
                if trajectory is not None:
                    trajectory.validation.end(epoch, criterion, neighbor)
            
            # Save the best model weights based on validation average mAP
            if phase == 'valid' and avg_map > best_avg_map:
                best_avg_map = avg_map
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                if getattr(configs, 'best_checkpoint', None):
                    if reference is not None:
                        # Same selection point, plain save only: no checkpoint probe/audit for CFSG.
                        destination = Path(configs.best_checkpoint)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open('wb') as stream:
                            torch.save(dict(
                                model_state_dict=model.state_dict(), emb_state_dict=emb.state_dict(),
                                epoch_zero_based=epoch, best_validation_map=float(avg_map),
                                training_config=dict(vars(configs)),
                                model_config=dict(img_input_dim=input_data_par['img_dim'],
                                                  text_input_dim=input_data_par['text_dim'],
                                                  output_dim=configs.output_dim, num_class=configs.data_class)), stream)
                    else:
                        from checkpoint_io import save_best_checkpoint
                        save_best_checkpoint(configs.best_checkpoint, model, emb,
                                             input_data_par, configs, epoch, avg_map)
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_on_test_set(model, emb, input_data_par, device, epoch + 1)

    time_end = time.time()
    time_used = time_end - time_start
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    logging.info(f'Best validation Average mAP: {best_avg_map:.6f}')
    logging.info(f'Best validation epoch (zero-based): {best_epoch}')
    if trajectory is not None:
        trajectory.save(best_epoch, best_avg_map)
    if equivalence is not None:
        equivalence.save(model, emb, criterion)
    if anchor_enabled and configs.log_dir:
        import json
        Path(configs.log_dir, 'anchor_training_stats.json').write_text(
            json.dumps(dict(anchor_total_count=anchor_total, epochs=anchor_epoch_stats), indent=2),
            encoding='utf-8')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    return model, mAP_history

def main():
    """Main function: initialize parameters, data, model and start training"""
    args = get_training_args()
    if args.best_checkpoint and Path(args.best_checkpoint).exists():
        raise FileExistsError(args.best_checkpoint)
    setup_logging(args)
    logging.info(args)
    
    # Device initialization
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")
    to_seed(args.seed)
    
    logging.info("\n[SSP]: Data loading starts...")
    dataset = args.dataset
    data_loader, input_data_par = get_loader(dataset, args.batch_size, args.partial_length)
    # V1.4: keep fixed data/candidates, then seed initialization, shuffle and dropout.
    if args.independent_train_seed:
        to_seed(args.seed)
    args.data_class = input_data_par['num_class']
    logging.info('    - Train Numbers: {train:>4}  Valid Numbers: {valid:>4}  Test Numbers: {test:>4}  Classes Numbers: {classes:>4}'.format(
                 train=input_data_par["img_train"].shape[0], 
                 valid=input_data_par["img_valid"].shape[0], 
                 test=input_data_par["img_test"].shape[0], 
                 classes=input_data_par["label_train"].shape[1]))
    
    # Model initialization
    model_ft = CMNN_Compat(
        img_input_dim=input_data_par['img_dim'], 
        text_input_dim=input_data_par['text_dim'], 
        output_dim=args.output_dim, 
        num_class=input_data_par['num_class']
    ).to(device)

    # Embedding layer and optimizer initialization
    emb = Embedding(args.data_class, args.output_dim).cuda()
    optimizer = optim.Adam([
        {'params': emb.parameters(), 'lr': args.lr},
        {'params': model_ft.parameters(), 'lr': args.lr}
    ])
    
    logging.info("\n[SSP]: Training starts...")
    model_ft, _ = train_model(model_ft, emb, data_loader, input_data_par, optimizer, args, device)

    logging.info("\n[Final Test Set Evaluation]")
    view1_feature, view2_feature = model_ft(
        torch.tensor(input_data_par['img_test']).to(device), 
        torch.tensor(input_data_par['text_test']).to(device)
    )
    
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='cosine')
    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='cosine')
    avg_map = (img_to_txt + txt_to_img) / 2.0
    
    logging.info("\n[SSP FINAL RESULT]:")
    logging.info(f"    - Image to Text MAP = {img_to_txt:.6f}")
    logging.info(f"    - Text to Image MAP = {txt_to_img:.6f}")
    logging.info(f"    - Average MAP = {avg_map:.6f}")
    
if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).resolve().parent)
    try:
        main()
    except Exception:
        logging.exception('Execution stopped; the current epoch is the last Epoch entry above.')
        raise
