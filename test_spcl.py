import unittest

import torch
import torch.nn.functional as F

from module import build_match_sen_shift_label
from spcl import (SPCLLogger, SPCLScheduler, build_pairwise_mask,
                  conversation_scores, difficulty, utterance_scores)


class SPCLTest(unittest.TestCase):
    def test_logit_decomposition_and_scores(self):
        torch.manual_seed(1)
        logits = {key: torch.randn(5, 3) for key in ('t', 'v', 'a')}
        joint = sum(logits.values())
        labels = torch.tensor([0, 1, 2, 0, 1])
        losses = utterance_scores(joint, labels)
        conv, streams = conversation_scores(logits, labels, [2, 3])
        self.assertEqual(tuple(conv.shape), (5,))
        self.assertEqual(tuple(streams.shape), (3, 2))
        self.assertTrue(torch.isfinite(difficulty(losses, conv)).all())

    def test_null_curriculum_matches_baseline(self):
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
        labels = torch.tensor([0, 1])
        scheduler = SPCLScheduler(float('inf'), 1.0)
        per_item = utterance_scores(logits, labels)
        mask = scheduler.mask(per_item)
        masked = (mask * per_item).sum() / mask.sum()
        baseline = F.nll_loss(F.log_softmax(logits, -1), labels)
        self.assertTrue(torch.allclose(masked, baseline))

        weights = torch.tensor([1.0, 3.0])
        per_item = utterance_scores(logits, labels, weights)
        masked = per_item.sum() / weights[labels].sum()
        baseline = F.nll_loss(F.log_softmax(logits, -1), labels,
                              weight=weights)
        self.assertTrue(torch.allclose(masked, baseline))

    def test_pairwise_mask_matches_shift_label_layout(self):
        mask = torch.tensor([1., 0., 1., 1., 0.])
        labels = torch.tensor([0, 1, 1, 2, 0])
        for shift_win in (-1, 2):
            pair = build_pairwise_mask(mask, [3, 2], shift_win)
            target = build_match_sen_shift_label(shift_win, [3, 2], labels)
            self.assertEqual(pair.shape, target.shape)

    def test_logger_summary(self):
        logger = SPCLLogger(2)
        logger.accumulate(torch.ones(2), torch.tensor([1., 0.]),
                          torch.ones(3, 1), torch.tensor([0, 1]),
                          torch.tensor([0.2, 0.3]), torch.tensor([0.1, 0.2]))
        summary = logger.summary()
        self.assertEqual(summary['expanding_rate'], 0.5)
        self.assertEqual(summary['admitted_per_class'], [1.0, 0.0])


if __name__ == '__main__':
    unittest.main()
