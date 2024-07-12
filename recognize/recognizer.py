from .vab import EOS, BOS
import torch


class Recognizer():
    def __init__(self, model, beam_width=5, nbest=1, max_len=50, 
                 idx2unit=None, ngpu=1,
                 lm=None, lm_weight=0.0):

        self.ngpu = ngpu
        self.model = model
        self.model.eval()
        if self.ngpu > 0 : self.model.cuda()

        self.idx2unit = idx2unit

        self.beam_width = beam_width
        self.max_len = max_len
        self.nbest = nbest

        self.lm = lm
        self.lm_weight = lm_weight
        if self.lm is not None:
            self.lm.eval()
        
    def encode(self, inputs, inputs_mask):
        inputs, inputs_mask = self.model.frontend(inputs, inputs_mask)
        
        memory, memory_mask, _ = self.model.encoder(inputs, inputs_mask)

        return memory, memory_mask

    def decode(self, preds, memory, memory_mask):
        log_probs,_ = self.model.decoder.inference(preds, memory, memory_mask)
        return log_probs

    def recognize(self, inputs, inputs_mask):
        memory, memory_mask = self.encode(inputs, inputs_mask)
        
        b, t, v = memory.size()

        beam_memory = memory.unsqueeze(1).repeat([1, self.beam_width, 1, 1]).view(b * self.beam_width, t, v)
        beam_memory_mask = memory_mask.unsqueeze(1).repeat([1, self.beam_width, 1]).view(b * self.beam_width, t)

        preds = torch.ones([b * self.beam_width, 1], dtype=torch.long, device=memory.device) * BOS

        scores = torch.FloatTensor([0.0] + [-float('inf')] * (self.beam_width - 1))
        scores = scores.to(memory.device).repeat([b]).unsqueeze(1)
        ending_flag = torch.zeros_like(scores, dtype=torch.bool)

        with torch.no_grad():
            for _ in range(1, self.max_len+1):
                preds, scores, ending_flag = self.decode_step(
                    preds, beam_memory, beam_memory_mask, scores, ending_flag)

                # whether stop or not
                if ending_flag.sum() == b * self.beam_width:
                    break

            scores = scores.view(b, self.beam_width)
            preds = preds.view(b, self.beam_width, -1)

            sorted_scores, offset_indices = torch.sort(scores, dim=-1, descending=True)

            base_indices = torch.arange(b, dtype=torch.long, device=offset_indices.device) * self.beam_width
            base_indices = base_indices.unsqueeze(1).repeat([1, self.beam_width]).view(-1)
            preds = preds.view(b * self.beam_width, -1)
            indices = offset_indices.view(-1) + base_indices

            # remove BOS
            sorted_preds = preds[indices].view(b, self.beam_width, -1)
            nbest_preds = sorted_preds[:, :min(self.beam_width, self.nbest), 1:]
            nbest_scores = sorted_scores[:, :min(self.beam_width, self.nbest)]

        return self.nbest_translate(nbest_preds), nbest_scores

    def decode_step(self, preds, memory, memory_mask, scores, flag):
        batch_size = int(scores.size(0) / self.beam_width)

        batch_log_probs = self.decode(preds, memory, memory_mask)
        
        if self.lm is not None:
            batch_lm_log_probs, lm_hidden = self.lm_decode(preds)
            batch_lm_log_probs = batch_lm_log_probs.squeeze(1)
            batch_log_probs = batch_log_probs + self.lm_weight * batch_lm_log_probs
            # print(batch_log_probs)

        if batch_log_probs.dim() == 3:
            batch_log_probs = batch_log_probs.squeeze(1)

        last_k_scores, last_k_preds = batch_log_probs.topk(self.beam_width)

        last_k_scores = mask_finished_scores(last_k_scores, flag)
        last_k_preds = mask_finished_preds(last_k_preds, flag)

        # update scores
        scores = scores + last_k_scores
        scores = scores.view(batch_size, self.beam_width * self.beam_width)

        # 裁剪，找到这几个beam中的最好的结果
        scores, offset_k_indices = torch.topk(scores, k=self.beam_width)
        scores = scores.view(-1, 1)

        device = scores.device
        base_k_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, self.beam_width])
        base_k_indices *= self.beam_width ** 2
        best_k_indices = base_k_indices.view(-1) + offset_k_indices.view(-1)

        # update predictions
        best_k_preds = torch.index_select(
            last_k_preds.view(-1), dim=-1, index=best_k_indices)
        
        preds_index = best_k_indices.floor_divide(self.beam_width)

        preds_symbol = torch.index_select(
            preds, dim=0, index=preds_index)
        preds_symbol = torch.cat(
            (preds_symbol, best_k_preds.view(-1, 1)), dim=1)

        end_flag = torch.eq(preds_symbol[:, -1], EOS).view(-1, 1)

        return preds_symbol, scores, end_flag

    def nbest_translate(self, nbest_preds):
        assert nbest_preds.dim() == 3
        batch_size, nbest, lens = nbest_preds.size()
        results = []
        for b in range(batch_size):
            nbest_list = []
            for n in range(nbest):
                pred = []
                for i in range(lens):
                    token = int(nbest_preds[b, n, i])
                    if token == EOS:
                        break
                    pred.append(self.idx2unit[token])
                nbest_list.append(' '.join(pred))
            results.append(nbest_list)
        return results
    
    def lm_decode(self, preds, hidden=None):
        log_probs = self.lm.predict(preds, last_frame=True)

        return log_probs, hidden
    
def mask_finished_scores(score, flag):
    beam_width = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_width > 1:
        unfinished = torch.cat(
            (zero_mask, flag.repeat([1, beam_width - 1])), dim=1)
        finished = torch.cat(
            (flag.bool(), zero_mask.repeat([1, beam_width - 1])), dim=1)
    else:
        unfinished = zero_mask
        finished = flag.bool()
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred, flag):
    beam_width = pred.size(-1)
    finished = flag.repeat([1, beam_width])
    return pred.masked_fill_(finished.bool(), EOS)
