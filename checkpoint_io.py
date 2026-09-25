"""Opt-in inference checkpoint; never changes training/history state."""
import copy
from pathlib import Path
import time

import torch


@torch.no_grad()
def fixed_outputs(model, emb, image, text, class_input):
    image_feature, text_feature = model(image, text)
    w = emb(class_input)
    return {
        'image_feature': image_feature,
        'text_feature': text_feature,
        'W': w,
        'image_probability': torch.softmax(image_feature @ w.T, dim=1),
        'text_probability': torch.softmax(text_feature @ w.T, dim=1),
    }


def save_best_checkpoint(path, model, emb, data, configs, epoch, validation_map):
    """Save both modules at the original best-selection point, with an eval probe.

    Probe only copies of the modules: original modes, buffers, gradients and
    parameters are untouched. fork_rng protects CPU and CUDA RNG streams.
    This has no optimizer/history/bank and is not a resume checkpoint.
    """
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        probe_model = copy.deepcopy(model).eval()
        probe_emb = copy.deepcopy(emb).eval()
        image = torch.as_tensor(data['img_train'][:8], dtype=torch.float32, device=device)
        text = torch.as_tensor(data['text_train'][:8], dtype=torch.float32, device=device)
        class_input = torch.eye(configs.data_class, device=device)
        outputs = fixed_outputs(probe_model, probe_emb, image, text, class_input)
        payload = {
            'format_version': 1,
            'epoch_zero_based': int(epoch),
            'best_validation_map': float(validation_map),
            'selection_point': 'after valid loss, at original best-model selection',
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'emb_state_dict': {k: v.detach().cpu().clone() for k, v in emb.state_dict().items()},
            'model_config': {
                'img_input_dim': int(data['img_dim']),
                'text_input_dim': int(data['text_dim']),
                'output_dim': int(configs.output_dim),
                'num_class': int(configs.data_class),
            },
            'training_config': dict(vars(configs)),
            'probe': {
                'mode': 'eval', 'source': 'training set rows 0:8',
                'sample_indices': torch.arange(len(image)),
                'image': image.cpu().clone(), 'text': text.cpu().clone(),
                'class_input': class_input.cpu().clone(),
                'outputs': {k: v.cpu().clone() for k, v in outputs.items()},
            },
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        # Explicit Python file I/O exposes Windows errors as OSError.
        for attempt in range(20):
            try:
                with temporary.open('wb') as stream:
                    torch.save(payload, stream)
                break
            except OSError:
                if attempt == 19:
                    raise
                time.sleep(0.1)
        # Retry only completed-file replacement, never a training step.
        for attempt in range(20):
            try:
                temporary.replace(destination)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)


def verify_checkpoint(path, device='cuda:0'):
    """Reload persisted states and compare to outputs measured at save time."""
    from model import CMNN_Compat, Embedding

    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']
    model = CMNN_Compat(**config).to(device).eval()
    emb = Embedding(config['num_class'], config['output_dim']).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    emb.load_state_dict(checkpoint['emb_state_dict'], strict=True)
    probe = checkpoint['probe']
    actual = fixed_outputs(model, emb, *(probe[k].to(device) for k in ('image', 'text', 'class_input')))
    checks = {}
    for name, output in actual.items():
        expected = probe['outputs'][name]
        output = output.cpu()
        checks[name] = {
            'exact_equal': torch.equal(output, expected),
            'max_abs_error': float((output - expected).abs().max()),
        }
        if not checks[name]['exact_equal']:
            raise AssertionError(f'Checkpoint reload probe differs: {name}: {checks[name]}')
    return {
        'status': 'PASS', 'checkpoint': str(Path(path).resolve()),
        'epoch_zero_based': checkpoint['epoch_zero_based'],
        'best_validation_map': checkpoint['best_validation_map'],
        'probe_mode': probe['mode'], 'probe_source': probe['source'],
        'outputs': checks,
    }