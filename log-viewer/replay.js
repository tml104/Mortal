/* 评测回放界面：只消费牌谱中的事件与 Q 值。 */
(function () {
    "use strict";
    const api = MjaiReplay;
    const replay = api.buildReplay(JSON.parse(document.getElementById("replay-data").textContent));
    const $ = id => document.getElementById(id);
    let roundIndex = 0, actionIndex = 0, viewpoint = 0, timer = null;
    const names = replay.names.map(name => ["Mortal-baseline-epoch3.46", "Mortal-baseline-epoch4-progress46.3397pct"].includes(name) ? "baseline · 第4轮46.3397%" : name === "Mortal-epoch8-complete" ? "epoch8 · 完整模型" : name);
    const playerViews = [];

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    // 使用原查看器的本地牌面资源；横置牌与加杠均保留对应形态。
    function tileImage(tile, className = "", sideways = false) {
        const honor = {E: "ji_e", S: "ji_s", W: "ji_w", N: "ji_n", P: "no", F: "ji_h", C: "ji_c", "?": "bk"};
        const name = honor[tile] || `${tile[1]}s${tile[0]}${tile.endsWith("r") ? "r" : ""}`;
        const image = element("img", `tile ${className}`);
        image.src = `files/images/p_${name}_${sideways ? 3 : 1}.${tile.endsWith("r") ? "png" : "gif"}`;
        image.alt = api.tileName(tile); image.title = image.alt;
        return image;
    }

    for (let side = 0; side < 4; side++) {
        const player = element("div", `player side-${side}`);
        const river = element("div", "river"), hand = element("div", "hand");
        const concealed = element("div", "concealed"), melds = element("div", "melds"), name = element("div", "player-name");
        hand.append(concealed, melds); player.append(river, hand, name); $("board").append(player);
        playerViews.push({river, concealed, melds, name});
    }
    for (const [index, round] of replay.rounds.entries()) {
        const option = element("option", "", `${index + 1}. ${round.title}`);
        option.value = index; $("round").append(option);
    }

    function renderPlayer(frame, id) {
        const player = frame.board.players[id], side = (id - viewpoint + 4) % 4, view = playerViews[side];
        view.name.textContent = `P${id} · ${names[id]}${player.reached ? " · 已立直" : ""}`;
        view.river.classList.toggle("long-river", player.river.length > 18);
        view.concealed.replaceChildren(); view.melds.replaceChildren(); view.river.replaceChildren();
        const hand = player.hand.slice();
        if (player.drawn) hand.splice(hand.lastIndexOf(player.drawn), 1);
        hand.sort((a, b) => api.tiles.indexOf(a.replace(/r$/, "")) - api.tiles.indexOf(b.replace(/r$/, "")) || a.localeCompare(b));
        hand.forEach(tile => view.concealed.append(tileImage(tile)));
        if (player.drawn) view.concealed.append(tileImage(player.drawn, "drawn"));
        for (const entry of player.river) {
            // 被副露取走的牌留下空格，保留巡目与立直宣言的位置。
            if (entry.called) { view.river.append(element("span")); continue; }
            const latest = frame.event.type === "dahai" && frame.event.actor === id && entry === player.river.at(-1);
            view.river.append(tileImage(entry.tile, `${entry.reach ? "reach-tile" : ""} ${latest ? "latest" : ""}`, entry.reach));
        }
        for (const meld of player.melds) {
            const node = element("div", "meld");
            if (meld.type === "ankan") {
                ["?", ...meld.consumed.slice(1, 3), "?"].forEach(tile => node.append(tileImage(tile)));
            } else {
                const tiles = meld.consumed.slice();
                const direction = (meld.target - id + 4) % 4;
                const position = Math.min(tiles.length, [0, 2, 1, 0][direction]);
                tiles.splice(position, 0, meld.taken);
                tiles.forEach((tile, i) => node.append(tileImage(tile, i === position ? "called" : "", i === position)));
                if (meld.added) {
                    const added = tileImage(meld.added, "added", true);
                    added.style.left = `${position * 31}px`; node.append(added);
                }
            }
            view.melds.append(node);
        }
        const badge = $(["seat-bottom", "seat-right", "seat-top", "seat-left"][side]);
        badge.classList.toggle("dealer", id === frame.board.dealer);
        badge.classList.toggle("reached", player.reached);
        badge.replaceChildren(element("div", "", `${api.seat(id, frame.board.dealer)} · P${id}${player.reached ? " 立直" : ""}`), element("strong", "", player.score.toLocaleString("en-US")));
    }

    // 以真实 Q 值排序；条形仅展示距本次最高 Q 的相对差距，不转换成概率。
    function weightTable(meta, frame, kanSelect = false) {
        const rows = api.decodeWeights(meta).sort((a, b) => b.q - a.q || a.id - b.id);
        if (!rows.length) return null;
        const table = element("table", "weights-table"), thead = element("thead"), head = element("tr");
        ["合法动作", "Q 值", "相对大小", "执行"].forEach(text => head.append(element("th", "", text)));
        thead.append(head); table.append(thead);
        const body = element("tbody"), selected = api.selectedAction(frame, kanSelect);
        const high = rows[0].q, low = rows.at(-1).q;
        for (const row of rows) {
            const tr = element("tr", row.id === selected ? "selected" : "");
            tr.dataset.actionId = row.id;
            tr.append(element("td", "", api.actionLabel(row.id, frame, kanSelect)), element("td", "q", row.q.toFixed(5)));
            const cell = element("td", "bar-cell"), bar = element("div", "q-bar"), fill = element("span");
            fill.style.width = `${high === low ? 100 : 5 + 95 * (row.q - low) / (high - low)}%`;
            bar.title = `最高 Q − 当前 Q = ${(high - row.q).toFixed(5)}`;
            bar.append(fill); cell.append(bar); tr.append(cell, element("td", "status", row.id === selected ? "● 实际" : ""));
            body.append(tr);
        }
        table.append(body); return table;
    }

    function render() {
        const round = replay.rounds[roundIndex], frame = round.frames[actionIndex], event = frame.event;
        $("round").value = roundIndex; $("timeline").max = round.frames.length - 1; $("timeline").value = actionIndex;
        $("position").textContent = `${actionIndex + 1}/${round.frames.length}`;
        $("prev").disabled = roundIndex === 0 && actionIndex === 0;
        $("next").disabled = roundIndex === replay.rounds.length - 1 && actionIndex === round.frames.length - 1;
        $("round-title").textContent = round.title;
        $("round-sticks").replaceChildren(element("div", "", `本场 ${frame.board.honba}`), element("div", "", `供托 ${frame.board.deposits}棒`));
        $("dora-tiles").replaceChildren(...frame.board.dora.map(tile => tileImage(tile)));
        for (let id = 0; id < 4; id++) renderPlayer(frame, id);
        const who = event.actor == null ? "" : `${api.seat(event.actor, frame.board.dealer)}家 P${event.actor} · `;
        $("event-title").textContent = `${who}${api.describeEvent(event)}`;
        const details = [];
        if (event.actor != null) details.push(names[event.actor]);
        if (event.deltas) details.push(`结算：${event.deltas.map((delta, i) => `P${i} ${delta >= 0 ? "+" : ""}${delta}`).join(" / ")}`);
        if (event.meta?.eval_time_ns != null) details.push(`推理 ${(event.meta.eval_time_ns / 1e6).toFixed(1)} ms`);
        $("event-details").textContent = details.join(" · ") || "使用上方控制或键盘 ← / → 回放";
        $("json").textContent = JSON.stringify(event, null, 2);
        $("weights").replaceChildren();
        const table = weightTable(event.meta, frame);
        if (table) {
            $("weights").append(element("h3", "weights-title", `本次决策 · ${event.meta.is_greedy === false ? "采样" : "贪心"} · ${api.decodeWeights(event.meta).length} 个合法动作`), table);
            const kanTable = weightTable(event.meta.kan_select, frame, true);
            if (kanTable) $("weights").append(element("h3", "weights-title", "第二阶段：选择杠哪种牌"), kanTable);
        } else {
            const decision = ["dahai", "chi", "pon", "ankan", "kakan", "daiminkan", "reach"].includes(event.type);
            $("weights").append(element("p", "empty-weights", decision ? "此步骤未记录模型 Q 值，可能使用了唯一合法动作的快速决策。无法还原未记录的候选权重。" : "此事件没有模型决策权重。摸牌、立直成立和结算等事件由牌局推进产生。"));
        }
        // 新事件从列表顶部开始；不要让前一事件的滚动位置隐藏实际候选。
        $("weights").scrollTop = 0; $("json").scrollTop = 0;
    }

    function stop() { clearInterval(timer); timer = null; $("play").textContent = "播放"; }
    function step(direction) {
        const round = replay.rounds[roundIndex];
        if (direction > 0 && actionIndex === round.frames.length - 1) {
            if (roundIndex === replay.rounds.length - 1) { stop(); return; }
            roundIndex++; actionIndex = 0;
        } else if (direction < 0 && actionIndex === 0) {
            if (roundIndex === 0) return;
            roundIndex--; actionIndex = replay.rounds[roundIndex].frames.length - 1;
        } else actionIndex += direction;
        render();
    }
    function play() { timer = setInterval(() => step(1), Number($("speed").value)); $("play").textContent = "暂停"; }
    $("prev").onclick = () => { stop(); step(-1); };
    $("next").onclick = () => { stop(); step(1); };
    $("play").onclick = () => timer ? stop() : play();
    $("speed").onchange = () => { if (timer) { stop(); play(); } };
    $("round").onchange = () => { stop(); roundIndex = Number($("round").value); actionIndex = 0; render(); };
    $("timeline").oninput = () => { stop(); actionIndex = Number($("timeline").value); render(); };
    $("viewpoint").onclick = () => { viewpoint = (viewpoint + 1) % 4; render(); };
    document.addEventListener("keydown", event => {
        if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
        if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); stop(); step(event.key === "ArrowRight" ? 1 : -1); }
        if (event.code === "Space") { event.preventDefault(); timer ? stop() : play(); }
    });
    // 牌桌按可用宽高一起缩放，右侧列表和 JSON 保持浏览器的正常字号。
    function resize() {
        const bounds = $("stage").getBoundingClientRect();
        $("board").style.setProperty("--board-scale", Math.max(.1, Math.min(bounds.width - 10, bounds.height - 10) / 640));
    }
    new ResizeObserver(resize).observe($("stage"));
    render(); resize();
})();
