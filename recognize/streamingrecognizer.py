import torch
import logging
from .vab import PAD, BLK, EOS

class StreamingRecognizer:
    def __init__(self, model, beam_width=5, nbest=1, max_len=50, 
                 idx2unit=None, ngpu=1, lm=None, lm_weight=0.0, 
                  ngram_lm=None, mode = "gready", alpha=0.1, beta=0.0):
        self.model = model
        self.ngpu = ngpu
        self.model.eval()
        if self.ngpu > 0 : self.model.cuda()

        self.lm = lm
        self.lm_weight = lm_weight

        self.idx2unit = idx2unit
        self.beam_width = beam_width

        self.mode = mode

        if self.mode == 'beam':
            # import ctcdecode_edited as ctcdecode
            # import ctcdecode
            ctcdecode = None
            vocab_list = [self.idx2unit[i] for i in range(len(idx2unit))]
            self.ctcdecoder = ctcdecode.CTCBeamDecoder(
                vocab_list, beam_width=self.beam_width,
                blank_id=BLK, model_path=ngram_lm, alpha=alpha, beta=beta,
                log_probs_input=True, num_processes=10)
            
    def recognize(self, inputs, inputs_length):

        if self.mode == 'greedy':
            results = self.recognize_greedy(inputs, inputs_length)
        else:
            results = self.recognize_greedy(inputs, inputs_length)
            # results = self.recognize_beam(inputs, inputs_length)

        return self.translate(results)
    
    def recognize_greedy(self, inputs, inputs_length):

        log_probs, length = self.model.inference(inputs, inputs_length)

        _, preds = log_probs.topk(self.beam_width, dim=-1)

        results = []
        for b in range(log_probs.size(0)):
            pred = []
            last_k = PAD
            for i in range(int(length[b])):
                k = int(preds[b][i][0])
                if k == last_k or k == PAD:
                    last_k = k
                    continue
                else:
                    last_k = k
                    pred.append(k)

            results.append(pred)
        return results
    
    def recognize_beam(self, inputs, inputs_length):

        log_probs, length = self.model.inference(inputs, inputs_length)

        beam_results, beam_scores, _, out_seq_len = self.ctcdecoder.decode(log_probs.cpu(), seq_lens=length.cpu())

        best_results = beam_results[:, 0]
        batch_length = out_seq_len[:, 0]
        # print(beam_scores)

        # print(best_results)

        results = []
        for b in range(log_probs.size(0)):
            length = int(batch_length[b])
            tokens = [int(i) for i in best_results[b, :length]]
            results.append(tokens)

        return results

    def translate(self, seqs):
        results = []
        for seq in seqs:
            pred = []
            for i in seq:
                if int(i) == EOS:
                    break
                if int(i) == PAD:
                    continue
                pred.append(self.idx2unit[int(i)])
            results.append(' '.join(pred))
        return results