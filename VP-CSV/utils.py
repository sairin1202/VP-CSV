import torch
import numpy as np
IMG_SEQ_NUM = 5



def get_semantic_idx(label, vtokens, num_text_tokens, text_seq_len):
    # print(len(label), len(vtokens))
    assert len(label) == len(vtokens)
    batch_in_index = []
    batch_out_index = []
    for bs in range(len(label)):
        in_index = []
        l = label[bs]
        vts = vtokens[bs]
        assert len(vts) == 5
        assert len(l) == text_seq_len + IMG_SEQ_NUM*8*8*2
        for _ in range(5):
            vt = vts[_]
            # print(vt)
            start = text_seq_len + _*8*8
            end = text_seq_len + (_+1)*8*8
            for idx in range(start, end):
                # print(l[idx].item()-num_text_tokens, vt)
                if l[idx].item()-num_text_tokens in vt:
                    in_index.append(idx)
            start = text_seq_len + (_+5)*8*8
            end = text_seq_len + (_+5+1)*8*8
            for idx in range(start, end):
                # print(l[idx].item()-num_text_tokens, vt)
                if l[idx].item()-num_text_tokens in vt:
                    in_index.append(idx)
        out_index = []
        for idx in range(text_seq_len, len(l)):
            if idx not in in_index:
                out_index.append(idx)
        batch_out_index.append(out_index)
        batch_in_index.append(in_index)
    return batch_in_index, batch_out_index

def compute_semantic_loss(labels, in_indexs, out_indexs, vtokens, logits, text_seq_len, num_text_tokens, reduce=True):
    probs = torch.softmax(logits, dim=-1)

    all_losses = []
    for label, in_index, out_index, vtoken, prob in zip(labels, in_indexs, out_indexs, vtokens, probs):
        all_losses.append(semantic_loss(prob, label, in_index, out_index, vtoken, text_seq_len, num_text_tokens))
    return sum(all_losses) / len(all_losses) if all_losses else None



def semantic_loss(prob, label, in_index, out_index, vtoken, text_seq_len, num_text_tokens):
    eps = 1e-5
    # add eps to log
    assert len(prob) == len(label)
    if len(in_index) == 0:
        return 0
    vts = []
    for vt in vtoken:
        vts.extend(list(vt))
        vts = list(vts)
    vtoken = list(set(vts))
    tar_prob = torch.sum(prob[text_seq_len:, np.array(vtoken)+num_text_tokens], dim=-1)
    out_index = np.array(out_index) - text_seq_len
    loss = sum([-torch.log(prob[idx, label[idx]]+eps) for idx in in_index])
    loss = loss / len(in_index)
    return loss