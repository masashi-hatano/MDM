import torch


def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k or k.startswith('embed_start') or k.startswith('embed_goal') or k.startswith('embed_ca') for k in missing_keys])

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model