#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common functions for ASR."""

from itertools import groupby
import numpy as np


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.

    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


class ErrorCalculator(object):
    """Calculate CER and WER for E2E_ASR and CTC models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    def __init__(
        self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False
    ):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.char_list = char_list
        self.space = sym_space
        self.blank = sym_blank
        self.idx_blank = self.char_list.index(self.blank)
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad, is_ctc=False):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :param bool is_ctc: calculate CER score for CTC
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = None, None
        if is_ctc:
            return self.calculate_cer_ctc(ys_hat, ys_pad)
        elif not self.report_cer and not self.report_wer:
            return cer, wer

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)
        return cer, wer

    def calculate_cer_ctc(self, ys_hat, ys_pad):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        cers, char_ref_lens = [], []
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_hat.append(self.char_list[int(idx)])

            for idx in y_true:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_true.append(self.char_list[int(idx)])

            hyp_chars = "".join(seq_hat)
            ref_chars = "".join(seq_true)
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace(self.space, " ")
            seq_hat_text = seq_hat_text.replace(self.blank, "")
            seq_true_text = "".join(seq_true).replace(self.space, " ")
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """
        import editdistance

        char_eds, char_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """
        import editdistance

        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)
