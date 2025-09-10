import torch
import numpy as np

def ctc_greedy_decode(output, blank=38): 
    chr_map = "0123456789KGABCDEFGHIJKLMNOPQRSTUVWXYZ" 
    result = []

    output = output.detach().cpu().numpy()
    for b in range(output.shape[0]):
        seq = output[b] 
        pred = np.argmax(seq, axis=1)  
        s = ""
        last = -1
        for p in pred:
            if p != last and p != blank and p < len(chr_map): 
                s += chr_map[p]
            last = p
        result.append(s)
    return result
