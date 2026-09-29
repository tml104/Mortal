import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


MORTAL_DIR = Path(__file__).resolve().parents[3]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.quarantine_logs.log_validation import (
    InvalidLogData,
    validate_protocol_text,
    validate_tile_and_score_rules,
)
from precompute_pipeline.quarantine_logs.tests.test_quarantine_logs import (
    minimal_events,
    render,
)


def parsed(events):
    return validate_protocol_text(render(events))


class TileInventoryTests(unittest.TestCase):
    def test_rejects_fifth_copy_across_initial_hands(self):
        events = minimal_events()
        events[1]['tehais'][0][:4] = ['1m'] * 4
        events[1]['tehais'][1][0] = '1m'

        with self.assertRaisesRegex(InvalidLogData, '1m.*4'):
            validate_tile_and_score_rules(parsed(events))

    def test_rejects_two_red_fives_of_same_suit(self):
        events = minimal_events()
        events[1]['tehais'][0][0] = '5mr'
        events[1]['tehais'][1][0] = '5mr'

        with self.assertRaisesRegex(InvalidLogData, '赤五万'):
            validate_tile_and_score_rules(parsed(events))

    def test_does_not_count_discard_and_call_as_new_tiles(self):
        events = minimal_events()
        events[1]['dora_marker'] = '9s'
        events[1]['tehais'][0][:3] = ['1m'] * 3
        events[2:2] = [
            {'type': 'tsumo', 'actor': 0, 'pai': '1m'},
            {'type': 'dahai', 'actor': 0, 'pai': '1m', 'tsumogiri': True},
            {
                'type': 'pon',
                'actor': 1,
                'target': 0,
                'pai': '1m',
                'consumed': ['1m', '1m'],
            },
        ]

        validate_tile_and_score_rules(parsed(events))

    def test_counts_duplicate_multi_hora_ura_marker_only_once(self):
        events = minimal_events()
        events[1]['tehais'][0][:3] = ['2p'] * 3
        events[2:3] = [
            {
                'type': 'hora',
                'actor': 1,
                'target': 0,
                'deltas': [-8000, 8000, 0, 0],
                'ura_markers': ['2p'],
            },
            {
                'type': 'hora',
                'actor': 2,
                'target': 0,
                'deltas': [-8000, 0, 8000, 0],
                'ura_markers': ['2p'],
            },
        ]

        validate_tile_and_score_rules(parsed(events))


class ScoreContinuityTests(unittest.TestCase):
    def test_accepts_riichi_payment_and_kyotaku_payout(self):
        events = minimal_events()
        events[2:3] = [
            {'type': 'reach_accepted', 'actor': 0},
            {'type': 'hora', 'actor': 1, 'target': 0, 'deltas': [-8000, 9000, 0, 0]},
        ]
        next_kyoku = deepcopy(events[1])
        next_kyoku.update({
            'kyoku': 2,
            'oya': 1,
            'scores': [16000, 34000, 25000, 25000],
        })
        events[-1:-1] = [next_kyoku, {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]}, {'type': 'end_kyoku'}]

        validate_tile_and_score_rules(parsed(events))

    def test_accepts_carried_kyotaku_after_draw(self):
        events = minimal_events()
        events[1]['scores'] = [24000, 25000, 25000, 25000]
        events[1]['kyotaku'] = 1
        next_kyoku = deepcopy(events[1])
        next_kyoku.update({'kyoku': 2, 'oya': 1})
        events[-1:-1] = [next_kyoku, {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]}, {'type': 'end_kyoku'}]

        validate_tile_and_score_rules(parsed(events))

    def test_rejects_next_kyoku_score_discontinuity(self):
        events = minimal_events()
        next_kyoku = deepcopy(events[1])
        next_kyoku.update({'kyoku': 2, 'oya': 1, 'scores': [24900, 25100, 25000, 25000]})
        events[-1:-1] = [next_kyoku, {'type': 'ryukyoku', 'deltas': [0, 0, 0, 0]}, {'type': 'end_kyoku'}]

        with self.assertRaisesRegex(InvalidLogData, '局间分数'):
            validate_tile_and_score_rules(parsed(events))

    def test_rejects_result_that_breaks_score_and_kyotaku_conservation(self):
        events = minimal_events()
        events[2] = {'type': 'hora', 'actor': 1, 'target': 0, 'deltas': [-8000, 8500, 0, 0]}

        with self.assertRaisesRegex(InvalidLogData, '供托'):
            validate_tile_and_score_rules(parsed(events))


if __name__ == '__main__':
    unittest.main()
