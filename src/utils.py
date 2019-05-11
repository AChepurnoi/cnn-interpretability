import os
import torch

CHECKPOINT_DIR = ""


def load_checkpoint(model, checkpoint, optimizer=None):
    exists = os.path.isfile(CHECKPOINT_DIR + checkpoint)
    if exists:
        state = torch.load(CHECKPOINT_DIR + checkpoint, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        optimizer_state = state.get('optimizer')
        if optimizer and optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        print("Checkpoint loaded: %s " % state['extra'])
        return state['extra']
    else:
        print("Checkpoint not found")
    return {'epoch': 0, 'lb_acc': 0}

def save_checkpoint(model, extra, checkpoint, optimizer=None):
    state = {'state_dict': model.state_dict(),
             'extra': extra}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, CHECKPOINT_DIR + checkpoint)
    print('model saved to %s' % (CHECKPOINT_DIR + checkpoint))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
