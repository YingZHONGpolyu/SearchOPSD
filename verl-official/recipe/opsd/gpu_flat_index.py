"""GPU-resident float32 inner-product search for the existing flat FAISS index."""

import threading
import numpy as np
import torch


class GPUFlatIPIndex:
    """Retain float32 vectors and exact inner products; scan in bounded chunks."""

    def __init__(self, index, chunk_size=1_000_000, reserve_gib=4):
        import faiss

        if not isinstance(index, faiss.IndexFlat) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError('GPU retrieval requires an uncompressed IndexFlatIP')
        self.ntotal, self.d = index.ntotal, index.d
        self.chunk_size = chunk_size
        required = self.ntotal * self.d * 4
        free, _ = torch.cuda.mem_get_info()
        if required + reserve_gib * 2**30 > free:
            raise MemoryError(f'Index needs {required / 2**30:.2f} GiB plus {reserve_gib} GiB reserve; '
                              f'only {free / 2**30:.2f} GiB is free')
        # Share the CPU FAISS storage during transfer instead of creating another full copy.
        vectors = faiss.rev_swig_ptr(index.get_xb(), self.ntotal * self.d).reshape(self.ntotal, self.d)
        self.vectors = torch.empty((self.ntotal, self.d), dtype=torch.float32, device='cuda')
        for start in range(0, self.ntotal, chunk_size):
            self.vectors[start:start + chunk_size].copy_(torch.from_numpy(vectors[start:start + chunk_size]))
        # Disable TF32 so the accelerator does not change the index precision.
        torch.set_float32_matmul_precision('highest')
        self.lock = threading.Lock()

    @torch.no_grad()
    def search(self, queries, k):
        """Match FAISS's score/index arrays while bounding temporary score memory."""
        if not 0 < k <= self.ntotal:
            raise ValueError('Requested k exceeds the index size')
        with self.lock:
            query = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device='cuda')
            best_scores = torch.empty((len(query), 0), dtype=torch.float32, device='cuda')
            best_ids = torch.empty((len(query), 0), dtype=torch.int64, device='cuda')
            for start in range(0, self.ntotal, self.chunk_size):
                block = self.vectors[start:start + self.chunk_size]
                scores = query @ block.T
                values, ids = scores.topk(min(k, len(block)), dim=1)
                ids += start
                merged_scores = torch.cat((best_scores, values), dim=1)
                merged_ids = torch.cat((best_ids, ids), dim=1)
                best_scores, positions = merged_scores.topk(min(k, merged_scores.shape[1]), dim=1)
                best_ids = merged_ids.gather(1, positions)
            return best_scores.cpu().numpy(), best_ids.cpu().numpy()
