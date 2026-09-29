"""筛选入口使用的 mjai 协议结构检查。"""

import json
from collections import Counter


TILES = {
    *(f'{number}{suit}' for suit in 'mps' for number in range(1, 10)),
    'E', 'S', 'W', 'N', 'P', 'F', 'C',
    '5mr', '5pr', '5sr', '?',
}

ACTION_FIELDS = {
    'tsumo': ('actor', 'pai'),
    'dahai': ('actor', 'pai', 'tsumogiri'),
    'chi': ('actor', 'target', 'pai', 'consumed'),
    'pon': ('actor', 'target', 'pai', 'consumed'),
    'daiminkan': ('actor', 'target', 'pai', 'consumed'),
    'kakan': ('actor', 'pai', 'consumed'),
    'ankan': ('actor', 'consumed'),
    'dora': ('dora_marker',),
    'reach': ('actor',),
    'reach_accepted': ('actor',),
    'hora': ('actor', 'target', 'deltas'),
    'ryukyoku': ('deltas',),
}
KNOWN_TYPES = {
    'start_game', 'start_kyoku', 'end_kyoku', 'end_game', *ACTION_FIELDS,
}


class InvalidLogData(ValueError):
    """表示可以明确归因于牌谱内容的输入错误。"""


def _fail(line, message):
    prefix = f'第 {line} 行: ' if line is not None else ''
    raise InvalidLogData(prefix + message)


def _require_fields(event, event_type, fields, line):
    for field in fields:
        if field not in event:
            _fail(line, f'{event_type} 缺少字段 {field}')


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_actor(event, field, line):
    value = event[field]
    if not _is_int(value) or not 0 <= value <= 3:
        _fail(line, f'{field} 必须是 0..3 的整数')


def _validate_tile(value, field, line):
    if not isinstance(value, str) or value not in TILES:
        _fail(line, f'{field} 含无效牌 {value!r}')


def _validate_int4(value, field, line):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(_is_int(item) for item in value)
    ):
        _fail(line, f'{field} 必须是四个整数')


def _validate_event(event, line):
    if not isinstance(event, dict):
        _fail(line, '事件必须是 JSON object')
    event_type = event.get('type')
    if not isinstance(event_type, str) or event_type not in KNOWN_TYPES:
        _fail(line, f'未知事件类型 {event_type!r}')

    if event_type == 'start_game':
        names = event.get('names')
        if names is not None and (
            not isinstance(names, list)
            or len(names) != 4
            or not all(isinstance(name, str) for name in names)
        ):
            _fail(line, 'start_game.names 必须是四个字符串')
        return

    if event_type == 'start_kyoku':
        fields = (
            'bakaze', 'dora_marker', 'kyoku', 'honba', 'kyotaku',
            'oya', 'scores', 'tehais',
        )
        _require_fields(event, event_type, fields, line)
        _validate_tile(event['bakaze'], 'bakaze', line)
        _validate_tile(event['dora_marker'], 'dora_marker', line)
        if event['bakaze'] not in {'E', 'S', 'W'}:
            _fail(line, 'bakaze 必须是 E、S 或 W')
        if not _is_int(event['kyoku']) or not 1 <= event['kyoku'] <= 4:
            _fail(line, 'kyoku 必须是 1..4 的整数')
        for field in ('honba', 'kyotaku'):
            if not _is_int(event[field]) or not 0 <= event[field] <= 255:
                _fail(line, f'{field} 必须是 0..255 的整数')
        _validate_actor(event, 'oya', line)
        _validate_int4(event['scores'], 'scores', line)
        tehais = event['tehais']
        if not isinstance(tehais, list) or len(tehais) != 4:
            _fail(line, 'tehais 必须包含四家手牌')
        for player_id, hand in enumerate(tehais):
            if not isinstance(hand, list) or len(hand) != 13:
                _fail(line, f'tehais[{player_id}] 必须包含 13 张牌')
            for tile in hand:
                _validate_tile(tile, f'tehais[{player_id}]', line)
        return

    fields = ACTION_FIELDS.get(event_type)
    if fields is None:
        return
    _require_fields(event, event_type, fields, line)
    for field in ('actor', 'target'):
        if field in event:
            _validate_actor(event, field, line)
    for field in ('pai', 'dora_marker'):
        if field in event:
            _validate_tile(event[field], field, line)
    if 'tsumogiri' in event and not isinstance(event['tsumogiri'], bool):
        _fail(line, 'tsumogiri 必须是布尔值')
    if 'consumed' in event:
        expected = {'chi': 2, 'pon': 2, 'daiminkan': 3, 'kakan': 3, 'ankan': 4}[event_type]
        consumed = event['consumed']
        if not isinstance(consumed, list) or len(consumed) != expected:
            _fail(line, f'{event_type}.consumed 必须包含 {expected} 张牌')
        for tile in consumed:
            _validate_tile(tile, f'{event_type}.consumed', line)
    if 'deltas' in event:
        _validate_int4(event['deltas'], f'{event_type}.deltas', line)
    if event_type == 'hora' and 'ura_markers' in event:
        markers = event['ura_markers']
        if not isinstance(markers, list):
            _fail(line, 'hora.ura_markers 必须是牌数组')
        for tile in markers:
            _validate_tile(tile, 'hora.ura_markers', line)


def validate_protocol_events(events):
    """检查一场牌谱的事件结构，并返回原事件列表。"""
    if not events:
        _fail(None, '牌谱为空')
    for line, event in enumerate(events, 1):
        _validate_event(event, line)
    if events[0]['type'] != 'start_game':
        _fail(1, '首事件必须是 start_game')
    if events[-1]['type'] != 'end_game':
        _fail(len(events), '末事件必须是 end_game')

    in_kyoku = False
    result_type = None
    kyoku_count = 0
    for line, event in enumerate(events[1:], 2):
        event_type = event['type']
        if event_type == 'start_game':
            _fail(line, '一份牌谱只能包含一个 start_game')
        if event_type == 'start_kyoku':
            if in_kyoku:
                _fail(line, '上一局尚未 end_kyoku，不能开始下一局')
            in_kyoku = True
            result_type = None
            kyoku_count += 1
            continue
        if event_type == 'end_game':
            if line != len(events):
                _fail(line, 'end_game 后不能再有事件')
            if in_kyoku:
                _fail(line, '当前局尚未 end_kyoku')
            continue
        if not in_kyoku:
            _fail(line, f'{event_type} 必须位于一局之内')
        if event_type == 'end_kyoku':
            if result_type is None:
                _fail(line, '每局必须先有 hora 或 ryukyoku 结算')
            in_kyoku = False
            continue
        if event_type == 'hora':
            if result_type == 'ryukyoku':
                _fail(line, 'ryukyoku 后不能再出现 hora')
            result_type = 'hora'
            continue
        if event_type == 'ryukyoku':
            if result_type is not None:
                _fail(line, '一局只能有一次 ryukyoku，且不能与 hora 混用')
            result_type = 'ryukyoku'
            continue
        if result_type is not None:
            _fail(line, '结算后只能继续 hora 或结束本局')

    if kyoku_count == 0:
        _fail(None, '牌谱至少需要一局')
    return events


def validate_protocol_text(raw_log):
    """解析逐行 JSON，并检查完整 mjai 对局结构。"""
    if not isinstance(raw_log, str):
        raise TypeError('raw_log 必须是字符串')
    events = []
    for line, raw_line in enumerate(raw_log.splitlines(), 1):
        if not raw_line.strip():
            _fail(line, '不允许空行')
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _fail(line, f'JSON 无法解析: {exc.msg}')
        events.append(event)
    return validate_protocol_events(events)


def _base_tile(tile):
    return {'5mr': '5m', '5pr': '5p', '5sr': '5s'}.get(tile, tile)


def _witness_physical_tile(tile, counts, red_counts, line, source):
    if tile == '?':
        return
    base = _base_tile(tile)
    counts[base] += 1
    if counts[base] > 4:
        _fail(line, f'{source} 使 {base} 的物理数量超过 4 张')
    if tile.endswith('r'):
        suit_name = {'m': '万', 'p': '筒', 's': '索'}[tile[1]]
        red_counts[tile[1]] += 1
        if red_counts[tile[1]] > 1:
            _fail(line, f'{source} 含超过一张赤五{suit_name}')


def validate_tile_and_score_rules(events):
    """检查每局物理牌来源，以及包含供托的局间点数守恒。"""
    counts = Counter()
    red_counts = Counter()
    ura_by_index = []
    scores = None
    kyotaku = None
    accepted_reaches = set()
    result_type = None
    result_deltas = [0, 0, 0, 0]
    expected_scores = None
    expected_kyotaku = None

    for line, event in enumerate(events, 1):
        event_type = event['type']
        if event_type == 'start_kyoku':
            if expected_scores is not None:
                if event['scores'] != expected_scores:
                    _fail(
                        line,
                        f'局间分数不连续: expected={expected_scores}, actual={event["scores"]}',
                    )
                if event['kyotaku'] != expected_kyotaku:
                    _fail(
                        line,
                        '局间供托不连续: '
                        f'expected={expected_kyotaku}, actual={event["kyotaku"]}',
                    )
            scores = list(event['scores'])
            kyotaku = event['kyotaku']
            if sum(scores) + kyotaku * 1000 != 100_000:
                _fail(line, '局初四家点数与供托合计必须为 100000')
            counts.clear()
            red_counts.clear()
            ura_by_index.clear()
            accepted_reaches.clear()
            result_type = None
            result_deltas = [0, 0, 0, 0]
            for hand in event['tehais']:
                for tile in hand:
                    _witness_physical_tile(tile, counts, red_counts, line, '起手牌')
            _witness_physical_tile(
                event['dora_marker'], counts, red_counts, line, '宝牌指示牌',
            )
            continue

        if scores is None:
            continue
        if event_type == 'tsumo':
            _witness_physical_tile(event['pai'], counts, red_counts, line, '摸牌')
        elif event_type == 'dora':
            _witness_physical_tile(
                event['dora_marker'], counts, red_counts, line, '追加宝牌指示牌',
            )
        elif event_type == 'reach_accepted':
            actor = event['actor']
            if actor in accepted_reaches:
                _fail(line, f'玩家 {actor} 在同一局重复 reach_accepted')
            accepted_reaches.add(actor)
            scores[actor] -= 1000
            kyotaku += 1
        elif event_type == 'hora':
            result_type = 'hora'
            for player_id, delta in enumerate(event['deltas']):
                result_deltas[player_id] += delta
            markers = event.get('ura_markers') or []
            for index, tile in enumerate(markers):
                if index < len(ura_by_index):
                    if ura_by_index[index] != tile:
                        _fail(line, f'多家和牌的里宝牌指示位置 {index} 不一致')
                    continue
                ura_by_index.append(tile)
                _witness_physical_tile(
                    tile, counts, red_counts, line, '里宝牌指示牌',
                )
        elif event_type == 'ryukyoku':
            result_type = 'ryukyoku'
            for player_id, delta in enumerate(event['deltas']):
                result_deltas[player_id] += delta
        elif event_type == 'end_kyoku':
            delta_sum = sum(result_deltas)
            if result_type == 'hora':
                expected_delta_sum = kyotaku * 1000
                if delta_sum != expected_delta_sum:
                    _fail(
                        line,
                        '和牌结算后的点数与供托不守恒: '
                        f'expected_delta_sum={expected_delta_sum}, actual={delta_sum}',
                    )
                remaining_kyotaku = 0
            else:
                if delta_sum != 0:
                    _fail(line, f'流局 deltas 合计必须为 0，实际为 {delta_sum}')
                remaining_kyotaku = kyotaku
            for player_id, delta in enumerate(result_deltas):
                scores[player_id] += delta
            if sum(scores) + remaining_kyotaku * 1000 != 100_000:
                _fail(line, '局末四家点数与供托合计必须为 100000')
            expected_scores = list(scores)
            expected_kyotaku = remaining_kyotaku
            scores = None
            kyotaku = None
    return events
