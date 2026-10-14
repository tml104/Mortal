"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const api = require("../replay-state.js");

function opening(overrides = {}) {
    return {type: "start_kyoku", bakaze: "E", kyoku: 1, honba: 0, kyotaku: 0, oya: 0,
        scores: [25000, 25000, 25000, 25000], dora_marker: "1p",
        tehais: [["1m", "2m", "3m"], ["2m", "3m", "4m"], ["E", "E", "W"], ["5sr", "5s", "5s"]], ...overrides};
}

test("立直扣分、供托、和牌结算可前后回放，初始供托及分数以牌谱为准", () => {
    const events = [opening({kyotaku: 2}), {type: "reach", actor: 0}, {type: "dahai", actor: 0, pai: "1m"},
        {type: "reach_accepted", actor: 0}, {type: "hora", actor: 1, target: 0, deltas: [-8000, 11000, 0, 0]}];
    const original = JSON.stringify(events), frames = api.buildReplay(events).rounds[0].frames;
    assert.equal(frames[0].board.deposits, 2);
    assert.equal(frames[3].board.deposits, 3);
    assert.equal(frames[3].board.players[0].score, 24000);
    assert.equal(frames[2].board.players[0].river[0].reach, true);
    assert.equal(frames[4].board.deposits, 0);
    assert.deepEqual(frames[4].board.players.map(p => p.score), [16000, 36000, 25000, 25000]);
    assert.equal(frames[0].board.players[0].score, 25000);
    assert.equal(JSON.stringify(events), original);
});

test("流局供托保留，次局重置座次及分数，显式 scores 不重复累计 deltas", () => {
    const replay = api.buildReplay([opening(), {type: "reach_accepted", actor: 2},
        {type: "ryukyoku", deltas: [1500, 1500, -1500, -1500]},
        opening({oya: 1, kyoku: 2, kyotaku: 1, honba: 1, scores: [26500, 26500, 22500, 23500]}),
        {type: "hora", actor: 0, target: 0, deltas: [4000, -1000, -1000, -1000], scores: [30500, 25500, 21500, 22500]}]);
    assert.equal(replay.rounds[0].frames.at(-1).board.deposits, 1);
    assert.equal(api.seat(0, replay.rounds[1].frames[0].board.dealer), "北");
    assert.deepEqual(replay.rounds[1].frames.at(-1).board.players.map(p => p.score), [30500, 25500, 21500, 22500]);
});

test("46 位掩码按动作编号顺序解释，包括赤牌、吃、立直与 pass", () => {
    const ids = [0, 34, 37, 38, 41, 42, 43, 45];
    const meta = {mask_bits: Number(ids.reduce((mask, id) => mask | (1n << BigInt(id)), 0n)), q_values: ids.map(id => id / 10)};
    assert.deepEqual(api.decodeWeights(meta).map(row => row.id), ids);
    const frame = {event: {type: "chi", actor: 1, pai: "2m", consumed: ["3m", "4m"]}, before: {lastDiscard: {tile: "2m"}, players: [{hand: []}, {hand: ["3m", "4m"]}]}};
    assert.equal(api.selectedAction(frame), 38);
    assert.equal(api.actionLabel(34, frame), "打赤五万");
    assert.equal(api.actionLabel(38, frame), "吃 二万 · 三万 · 四万");
    assert.equal(api.actionLabel(37, frame), "立直");
    assert.match(api.actionLabel(45, frame), /跳过/);
    assert.throws(() => api.decodeWeights({...meta, q_values: [0]}), /不一致/);
});

test("加杠保留原碰牌方向，主动作与第二阶段杠选择分别标记", () => {
    const events = [opening(), {type: "dahai", actor: 2, pai: "E"},
        {type: "pon", actor: 0, target: 2, pai: "E", consumed: ["E", "E"]},
        {type: "tsumo", actor: 0, pai: "E"}, {type: "kakan", actor: 0, pai: "E", consumed: ["E", "E", "E"]}];
    events[0].tehais[0] = ["E", "E", "1m"];
    const frames = api.buildReplay(events).rounds[0].frames, last = frames.at(-1);
    assert.equal(last.board.players[0].melds[0].type, "kakan");
    assert.equal(last.board.players[0].melds[0].target, 2);
    assert.equal(frames[1].board.players[2].river[0].called, false);
    assert.equal(last.board.players[2].river[0].called, true);
    assert.equal(api.selectedAction(last), 42);
    assert.equal(api.selectedAction(last, true), 27);
    assert.equal(api.selectedAction({event: {type: "dahai", pai: "1m"}}, true), -1);
});

test("荣和双响逐次累计结算，赤牌吃法沿用实际手牌", () => {
    const frames = api.buildReplay([opening({kyotaku: 1}),
        {type: "hora", actor: 1, target: 0, deltas: [-8000, 9000, 0, 0]},
        {type: "hora", actor: 2, target: 0, deltas: [-3900, 0, 3900, 0]}]).rounds[0].frames;
    assert.deepEqual(frames.at(-1).board.players.map(p => p.score), [13100, 34000, 28900, 25000]);
    assert.equal(frames.at(-1).board.deposits, 0);
    const frame = {event: {type: "chi", actor: 0}, before: {lastDiscard: {tile: "4p"}, players: [{hand: ["5pr", "6p"]}]}};
    assert.equal(api.actionLabel(38, frame), "吃 四筒 · 赤五筒 · 六筒");
});
