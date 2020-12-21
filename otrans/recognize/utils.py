import torch
from otrans.data import BLK


def select_tensor_based_index(tensor, index):
    # tensor: [b, c, t, v]
    # index: [b]
    # return [b, t, v]
    assert tensor.dim() >= 2
    assert index.dim() == 1

    batch_size = tensor.size(0)
    tensor_len = tensor.size(1)

    base_index = torch.arange(batch_size, device=tensor.device) * tensor_len
    indices = base_index + index

    if tensor.dim() == 2:
        select_tensor = torch.index_select(tensor.reshape(batch_size * tensor_len), 0, indices.long())
    else:
        assert tensor.dim() == 3
        select_tensor = torch.index_select(tensor.reshape(batch_size * tensor_len, tensor.size(-1)), 0, indices.long())

    return select_tensor


def select_chunk_states_and_mask_based_index(tensor, tensor_mask, index):
    # tensor: [b, c, t, v]
    # index: [b]
    # return [b, t, v]
    assert tensor.dim() == 4
    assert tensor_mask.dim() == 3
    assert index.dim() == 1

    b, c, t, v = tensor.size()

    base_index = torch.arange(b, device=tensor.device) * c
    indices = base_index + index

    select_tensor = torch.index_select(tensor.reshape(b * c, t, v), 0, indices.long())
    select_tensor_mask = torch.index_select(tensor_mask.reshape(b * c, 1, t), 0, indices.long())

    return select_tensor, select_tensor_mask


def fill_tensor_based_index(tensor, index, value, blank=BLK):

    assert tensor.dim() == 2
    assert value.dim() == 1
    assert value.size(0) == tensor.size(0)
    assert index.size(0) == value.size(0)
    assert tensor.size(1) >= int(torch.max(index))

    for b in range(index.size(0)):
        pos = int(index[b])
        if int(value[b]) == blank:
            continue
        else:
            tensor[b, pos] = value[b]

    return tensor


def select_hidden(tensor, indices, beam_width):
    n_layers, _, hidden_size = tensor.size()
    tensor = tensor.transpose(0, 1)
    tensor = tensor.unsqueeze(1).repeat([1, beam_width, 1, 1]).reshape(-1, n_layers, hidden_size)
    new_tensor = torch.index_select(tensor, dim=0, index=indices)
    new_tensor = new_tensor.transpose(0, 1).contiguous()
    return new_tensor