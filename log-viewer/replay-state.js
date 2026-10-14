/* mjai 回放状态及动作解释；不依赖 DOM，也不重新调用模型。 */
(function (root) {
    "use strict";
    const tiles = [
        ...Array.from({length: 9}, (_, i) => `${i + 1}m`),
        ...Array.from({length: 9}, (_, i) => `${i + 1}p`),
        ...Array.from({length: 9}, (_, i) => `${i + 1}s`),
        "E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr"
    ];
    const winds = ["东", "南", "西", "北"];
    const honors = {E: "东", S: "南", W: "西", N: "北", P: "白", F: "发", C: "中", "?": "暗牌"};
    const plain = tile => tile.replace(/r$/, "");
    const tileName = tile => honors[tile] || `${tile.endsWith("r") ? "赤" : ""}${"一二三四五六七八九"[Number(tile[0]) - 1]}${{m: "万", p: "筒", s: "索"}[tile[1]]}`;
    const seat = (player, dealer) => winds[(player - dealer + 4) % 4];

    // 每个事件保留独立快照，前进、后退均不会改变原始牌谱或之前的分数。
    function buildReplay(events) {
        const rounds = [];
        let names = [], board = null, round = null;
        function remove(player, tile) {
            const index = player.hand.indexOf(tile);
            if (index < 0) throw new Error(`牌谱手牌中找不到 ${tile}`);
            player.hand.splice(index, 1);
        }
        for (const event of events) {
            if (event.type === "start_game") { names = event.names.slice(); continue; }
            if (event.type === "start_kyoku") {
                board = {
                    dealer: event.oya, wind: event.bakaze, number: event.kyoku,
                    honba: event.honba, deposits: event.kyotaku,
                    dora: [event.dora_marker], lastDiscard: null,
                    players: event.tehais.map((hand, i) => ({
                        hand: hand.slice(), drawn: null, river: [], melds: [],
                        score: event.scores[i], reached: false, reachIndex: null
                    }))
                };
                round = {title: `${honors[event.bakaze]}${event.kyoku}局 ${event.honba}本场`, frames: []};
                rounds.push(round);
            } else if (!board) {
                continue;
            } else {
                board = structuredClone(board);
            }
            const before = round.frames.length ? round.frames.at(-1).board : board;
            const actor = board.players[event.actor];
            switch (event.type) {
                case "tsumo": actor.hand.push(event.pai); actor.drawn = event.pai; break;
                case "dahai":
                    remove(actor, event.pai); actor.drawn = null;
                    actor.river.push({tile: event.pai, reach: actor.reachIndex === actor.river.length, called: false});
                    board.lastDiscard = {tile: event.pai, actor: event.actor};
                    break;
                case "reach": actor.reachIndex = actor.river.length; break;
                case "reach_accepted":
                    actor.reached = true;
                    actor.score -= 1000; board.deposits += 1;
                    break;
                case "chi": case "pon": case "daiminkan":
                    board.players[event.target].river.at(-1).called = true;
                    event.consumed.forEach(tile => remove(actor, tile));
                    actor.drawn = null;
                    actor.melds.push({type: event.type, taken: event.pai, consumed: event.consumed.slice(), target: event.target});
                    break;
                case "ankan":
                    event.consumed.forEach(tile => remove(actor, tile)); actor.drawn = null;
                    actor.melds.push({type: "ankan", consumed: event.consumed.slice()});
                    break;
                case "kakan": {
                    remove(actor, event.pai); actor.drawn = null;
                    const meld = actor.melds.find(m => m.type === "pon" && plain(m.taken) === plain(event.pai));
                    if (!meld) throw new Error("加杠缺少对应碰牌");
                    meld.type = "kakan"; meld.added = event.pai;
                    break;
                }
                case "dora": board.dora.push(event.dora_marker); break;
                case "hora": board.deposits = 0; break;
            }
            // deltas 已包含和牌获得的供托，不另行给赢家加点；scores 若存在则是权威值。
            if (event.deltas) event.deltas.forEach((delta, i) => { board.players[i].score += delta; });
            if (event.scores) event.scores.forEach((score, i) => { board.players[i].score = score; });
            round.frames.push({event, board, before});
        }
        return {names, rounds};
    }

    // mask_bits 为 46 位掩码，不能使用 JavaScript 的 32 位位运算。
    function decodeWeights(meta) {
        if (!meta || !Array.isArray(meta.q_values) || meta.mask_bits == null) return [];
        const mask = BigInt(meta.mask_bits), result = [];
        for (let id = 0; id < 46; id++) {
            if (mask & (1n << BigInt(id))) result.push({id, q: meta.q_values[result.length]});
        }
        if (result.length !== meta.q_values.length || result.some(row => !Number.isFinite(row.q))) {
            throw new Error("合法动作掩码与 Q 值不一致");
        }
        return result;
    }

    // 吃牌方向按被吃牌位于顺子的低、中、高位命名；优先显示实际消耗的赤牌。
    function chiTiles(id, frame) {
        const discard = frame.before.lastDiscard;
        if (!discard) return [];
        const tile = discard.tile, number = Number(tile[0]), suit = tile[1];
        const low = number - (id - 38);
        const hand = frame.before.players[frame.event.actor]?.hand || [];
        return [low, low + 1, low + 2].map(n => {
            if (n === number) return tile;
            const base = `${n}${suit}`;
            return n === 5 && hand.includes(`${base}r`) ? `${base}r` : base;
        });
    }

    function actionLabel(id, frame, kanSelect = false) {
        if (id <= 36) return `${kanSelect ? "杠" : "打"}${tileName(tiles[id])}`;
        if (id === 37) return "立直";
        if (id >= 38 && id <= 40) return `吃 ${chiTiles(id, frame).map(tileName).join(" · ")}`;
        if (id === 41) return `碰${frame.before.lastDiscard ? tileName(frame.before.lastDiscard.tile) : "牌"}`;
        if (id === 42) {
            if (["ankan", "kakan", "daiminkan"].includes(frame.event.type)) return describeEvent(frame.event);
            return "杠（具体牌见杠牌选择）";
        }
        if (id === 43) return frame.event.type === "hora" ? (frame.event.actor === frame.event.target ? "自摸和牌" : "荣和") : "和牌";
        if (id === 44) return "九种九牌流局";
        return "跳过（不吃／碰／杠／荣和）";
    }

    function selectedAction(frame, kanSelect = false) {
        const event = frame.event;
        if (kanSelect) return ["ankan", "kakan", "daiminkan"].includes(event.type) ? tiles.indexOf(plain(event.pai || event.consumed[0])) : -1;
        switch (event.type) {
            case "dahai": return tiles.indexOf(event.pai);
            case "reach": return 37;
            case "chi": return 38 + Number(event.pai[0]) - Math.min(Number(event.pai[0]), ...event.consumed.map(t => Number(t[0])));
            case "pon": return 41;
            case "ankan": case "kakan": case "daiminkan": return 42;
            case "hora": return 43;
            case "ryukyoku": return 44;
            case "none": return 45;
            default: return -1;
        }
    }

    function describeEvent(event) {
        const name = event.pai ? tileName(event.pai) : "";
        const labels = {
            start_kyoku: "小局开始", tsumo: `摸${name}`, dahai: `打${name}${event.tsumogiri ? "（摸切）" : "（手切）"}`,
            reach: "宣告立直", reach_accepted: "立直成立（扣 1,000 点，供托 +1）",
            chi: `吃 ${[event.pai, ...(event.consumed || [])].filter(Boolean).sort().map(tileName).join(" · ")}`,
            pon: `碰${name}`, daiminkan: `大明杠${name}`,
            ankan: `暗杠${event.consumed ? tileName(plain(event.consumed[0])) : ""}`, kakan: `加杠${name}`,
            hora: event.actor === event.target ? "自摸和牌" : "荣和", ryukyoku: "流局", dora: `翻宝牌指示牌${event.dora_marker ? tileName(event.dora_marker) : ""}`,
            end_kyoku: "小局结束", end_game: "半庄结束", none: "跳过"
        };
        return labels[event.type] || event.type;
    }

    const api = {tiles, winds, tileName, seat, buildReplay, decodeWeights, actionLabel, selectedAction, describeEvent};
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    else root.MjaiReplay = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
