import torch
from falconn import LSHIndex, LSHConstructionParameters, get_default_parameters

class KVManager:
    def __init__(self, batch_size=2, threshold=0.9, max_length=512, eviction_mode='FIFO'):
        self.key_cache = []  # [batch_size, num_heads, seq_len, head_dim]
        self.value_cache = []  # [batch_size, num_heads, seq_len, head_dim]
        self.inputs_embeds = []  # # [batch_size, seq_len, embedding_dim]
        self.coordinates = []
        self.batch_size = batch_size
        self.threshold = threshold
        self.indices_to_delete = None



        self.eviction_idx = []
        self.eviction_mode = eviction_mode
        if self.eviction_mode == 'FIFO':
            for i in range(batch_size):
                self.eviction_idx.append(i)
        else:
            # LRU
            for i in range(max_length):
                self.eviction_idx.append(0)

        self.max_length = max_length
        # (batch_size, seq_length * embedding_dim)
        # self.params = get_default_parameters()

    def reset_cache(self):
        self.key_cache = []
        self.value_cache = []
        self.inputs_embeds = []


    def to_dict(self):
        return{
            'key_cache': self.key_cache,
            'value_cache': self.value_cache,
            'inputs_embeds': self.inputs_embeds,
            'coordinates': self.coordinates,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'indices_to_delete': self.indices_to_delete,
            'eviction_idx': self.eviction_idx,
            'eviction_mode': self.eviction_mode,
            'max_length': self.max_length
        }




