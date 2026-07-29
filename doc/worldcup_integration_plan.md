# 世界杯 2026 国家队模型 — 前后端接入执行计划

> 基于对四个仓库的实质代码调研（2026-05-26）。给各仓库实施 agent 用。
> 全程中文，代码标识符保持英文。所有关键结论附 `file:line` 证据。

涉及仓库：
- `tikaml`（主仓库，预测服务，已含 `/national/*` 端点）— `/Users/yafet/Documents/github/tikaml`
- `tikaml-data-service`（数据/编排层）— `/Users/yafet/Documents/github/tikaml-data-service`
- `tikaml-ai-service`（LLM 分析）— `/Users/yafet/Documents/github/tikaml-ai-service`
- `web-sportalpha`（前端）— `/Users/yafet/Documents/github/liveai/web-sportalpha`

---

## 0. 总览与四条贯穿全局的关键决策

### 数据流（端到端）
```
[Opta 逆向]──season_sync──> data-service `matches` 表(已含 WOC 赛程)
                                  │
                  ┌───────────────┴───────────────────────────┐
            (赛前/滚球)                                   (赛果后/每日)
                  │                                             │
   data-service 国家队 trigger(新, 绕过特征工程)        国家队 sim 任务(新)
   POST tika /national/predict ─┐                  POST tika /national/simulate
                                │                  (played/played_ko/bracket)
                                ▼                                │
              match_predictions(source="national")    national_champion_odds / _bracket / _standings(新表)
                                │                                │
        publish Redis tika:{opta_id}:prediction          /api/v2/national/champion-odds, /bracket
        (含 tournament_context) + WS 推送                       │
                  │                                             ▼
            ai-service 生成分析 ─> ai_results            前端 排行榜/对阵树(新组件)
                  │
            /api/v2/matches/{id}/predictions?source=national + WS tika_prediction
                  ▼
   前端 复用 AIPredictionSection/ScoreHeatmap/滚球(零新建)
```

### 决策 1 — 国家队管道必须**独立于俱乐部特征工程链路**（最重要）
俱乐部预测链路被两道硬绑死：
- `data-service/app/predictions/trigger.py:71-83` 硬过滤 `if comp_id not in FIVE_LEAGUE_IDS: return False`；
- `trigger.py:101-111` 调 `compute_match_features` 拼 ~98 维 `feature_vector`，国家队返回 `predictable=False` 直接 `return False`。

后果：**国家队比赛永远过不了俱乐部 trigger，既不会被预测，也永远不会 publish `tika:*:prediction`** → 这正是 ai-service 收不到任何国家队触发的根因。
→ **不要复用 `trigger_prediction`。** 新建独立 `app/national/{trigger,scheduler}.py`，绕过 `compute_match_features`，直接调主仓库 `/national/*`（国家队模型自包含、无需特征工程，`tikaml/src/national_api.py` 直接收队名）。

### 决策 2 — 预测输出已镜像俱乐部 goals 块 → 前端预测组件整套复用
`/national/predict` 的 `predictions.goals` 形状与俱乐部完全一致（`tikaml/src/national_api.py:143-199`）：`home_win/draw/away_win/expected_home/expected_away/predicted_total/over_under/score_matrix/recommended_score`，滚球额外 `lambda_remaining_*/next_goal`。前端 `Prediction` 类型（`web-sportalpha/src/lib/types/api.ts:222-251`）**逐字段匹配** → `AIPredictionSection`/`ScoreHeatmap`/`toPredictionSnapshots`/滚球 WS **可零改动复用**。

### 决策 3 — 队名归一化是隐藏地雷（务必先处理）
`/national/predict` 入参是**队名字符串**，模型用 `attack.get(team, 0.0)` 查评分（`tikaml/src/national_poisson.py:148-149`）。模型只认 `ratings.json` 里的 299 个队名（如 `"United States"`/`"South Korea"`/`"China PR"`），**Opta 的队名体系大概率不同** → 不匹配就静默回退 0 评分、给出错误预测。
→ data-service 在调 `/national/predict` 前，**必须把 Opta 队名映射到模型已知队名**。模型已知队名清单可由 `GET /national/ratings?top=300` 拉全量。建一张映射表/归一化层（参考记忆 `reference_international_data_sources.md`：结果集队名 vs FIFA 仅 195/333 能对上，已知是脏的）。**这是上线正确性的前置条件，不是可选项。**

### 决策 4 — 单一数据源（Opta/data-service），**不引 API-Football**（除非 Opta 逆向对国际赛实时不可靠）
详见 §6。结论：前端所有元数据（logo/赛程/比分/积分/名单/滚球）已由 data-service 提供，模型自包含——API-Football 对当前架构**冗余**。仅作 Opta 国际赛实时数据不可靠时的**后备**，且只补赛程/积分/对阵/live，**不取赔率**（结构性拿不到），并注意 key **2026-06-07 到期早于开赛 06-11**。

---

## 1. 部署（主仓库 tikaml）— 测试可获取预测推理

`/national/*` 已在 `main`（commit `1143145`/`e827e73`），server 启动时加载模型（`src/server.py:74 lifespan` → `:114 national_api.load_national()`），路由挂载于 `:133`（`Depends(verify_api_key)`，X-API-Key）。健康检查 `/` 已上报 national 加载状态（`server.py:475-477`）。

容器配置（`docker-compose.yml`）：service `predict`，容器 `tikaml-predict`，端口 `127.0.0.1:8001`，外部网络 `infra_default`，env `TIKA_API_KEY`。

### 部署步骤（服务器 `sport`，5.223.47.115，路径 `/root/tikaml`）
```bash
ssh sport 'cd /root/tikaml && git pull && docker compose up -d --build'
```

### 部署后冒烟测试（在服务器本机或经反代 https://api.sportalpha.io）
```bash
# 健康检查：national.loaded 应为 true，teams≈299
curl -s http://127.0.0.1:8001/ | jq '.national'

# 单场赛前预测（X-API-Key 必填）
curl -s -X POST http://127.0.0.1:8001/national/predict \
  -H "X-API-Key: $TIKA_API_KEY" -H "Content-Type: application/json" \
  -d '{"home_team":"Spain","away_team":"Morocco","neutral":true}' | jq '.predictions.goals'

# 滚球加时（90' 1-1，进加时传 total_minutes=120）
curl -s -X POST http://127.0.0.1:8001/national/predict \
  -H "X-API-Key: $TIKA_API_KEY" -H "Content-Type: application/json" \
  -d '{"home_team":"Spain","away_team":"Morocco","prediction_type":"live","minute":95,"home_goals":1,"away_goals":1,"total_minutes":120}' | jq '.predictions.goals'

# 赛事模拟（赛前夺冠/晋级概率）
curl -s -X POST http://127.0.0.1:8001/national/simulate \
  -H "X-API-Key: $TIKA_API_KEY" -H "Content-Type: application/json" \
  -d '{"n_sims":10000}' | jq '.teams[:5]'

# 实力榜
curl -s "http://127.0.0.1:8001/national/ratings?top=20" -H "X-API-Key: $TIKA_API_KEY" | jq '.ratings[:5]'
```
预期：`/national/predict` 返回 goals 块；`/national/simulate` 返回各队 `P_champion` 等；`/national/ratings` TOP 为 Argentina/Spain/England/France 等。
**注意**：`total_minutes` 仅接受 90 或 120（`national_api.py` field_validator），其它值 422。

---

## 2. data-service 执行计划

> 现状：WOC 竞赛已配置（`app/opta/constants.py` comp_id=`70excpe1synn9kadnbppahdn7`），`season_sync` 已把 WOC 赛程写进 `matches` 表（淘汰赛 TBD 槽位跳过，`app/collector/season_sync.py:109-112`）。**国际赛业务代码未开工**。无任何 API-Football key/客户端。

### 2.0 前置：核对竞赛 ID + 填 key
- **核对 `70excpe1synn9kadnbppahdn7` 是"国家队世界杯"而非"俱乐部世界杯"**（Opta 两者都有；ai-service `context/match_context.py:22` 也映射了同 ID 为 "FIFA World Cup"，需确认语义）。
- 填 `.env` 的 `TIKA_API_KEY`（接 `/national/*` 必需，与 `/predict` 同一 key，`tikaml/src/server.py:133`）；`TIKA_PREDICTION_URL` 已是 `https://api.sportalpha.io`（`app/config.py:66-70`）。

### 2.1 扩展预测客户端 `app/predictions/client.py`
`TikaPredictionClient`（`client.py:21-162`，单例 `tika_client`）已配好 `base_url`+`X-API-Key`（`:27-35`）。**复用同一个 `self._client`**，新增三方法（无 `feature_vector`，body 对齐 `tikaml/src/national_api.py:59-92`）：
- `national_predict(home_team, away_team, neutral=True, prediction_type="prematch", minute=0, home_goals=0, away_goals=0, home_red_cards=0, away_red_cards=0, total_minutes=90)` → POST `/national/predict`。返回 `{"predictions":{"goals":{...}}, "model_metadata":{...}}`。
- `national_simulate(n_sims=10000, seed=0, played=None, played_ko=None, bracket=None)` → POST `/national/simulate`。返回 `{"n_sims", "teams":[{team, P_advance, P_win_group, P_R16, P_QF, P_SF, P_F, P_champion}]}`（列名见 `tikaml/src/national_simulate.py:287-293`）。
- `national_ratings(top=30)` → GET `/national/ratings`。
错误处理照现有 try/except 返回 None 模式（`client.py:94-106`）。

### 2.2 队名归一化层（决策 3）
新建 `app/national/team_map.py`：维护 Opta 队名 → 模型队名 的映射 + 归一化函数。启动时（或定时）拉 `tika_client.national_ratings(top=300)` 得到模型已知队名全集，对 `matches` 表里 WOC 比赛的队名做匹配，**未命中要告警**（参考俱乐部 `app/odds/team_map.py` 的映射风格）。

### 2.3 独立国家队管道 `app/national/{scheduler,trigger}.py`（决策 1）
- **赛前**：定时查 `matches` 表 `competition_id == WOC && status=="PreMatch"` 的比赛 → 经队名归一化 → `tika_client.national_predict(...)` → 存 `match_predictions`（`source="national"`, `prediction_type="goals"`, `feature_snapshot_id` 留空）。`neutral`：东道主(US/CA/MX)本国主场比赛 `neutral=False`，其余 `True`（记忆 `reference_international_data_sources.md`：Opta 对东道主主场已标 `neutral=False`，可直接读）。
- **publish + WS**（给 ai-service 与前端）：仿 `trigger.py:218-268`，向 Redis `tika:{opta_match_id}:prediction` 发消息（结构对齐 `trigger.py:218-228`），并向 `match:{opta_match_id}:tika_prediction` 发 `PredictionSchema` 兼容数组。**消息里增补 `tournament_context`**（`stage`/小组或淘汰赛、`p_champion`/`p_advance` 等），供 ai-service 渲染（见 §3）。
- **滚球**：`MatchCollector` 对所有 active 比赛起 collector（`app/collector/manager.py:75-112` 无竞赛过滤，WC 可被扫到）。在 `app/collector/match_collector.py:585 _evaluate_tika_triggers` 按 `competition_id` 分流：WOC → 走国家队 trigger（传 `minute/score/red_cards/total_minutes`，**不拼 feature_vector**）。加时把 `total_minutes=120`（依据比赛 `period`/status 判断）。
- **赛事模拟**：定时任务（每场赛果后或每日）→ 组装 `played`（小组赛已踢结果）/`played_ko`（淘汰赛胜者）/`bracket` → `tika_client.national_simulate(...)` → 写 `national_champion_odds`（+ `national_bracket`）。
- 启动挂载：`app/collector/__main__.py:51-59` 仿 `run_prediction_scheduler`，`asyncio.create_task(run_national_scheduler(stop_event))`。

### 2.4 数据库新表 + Alembic
- **单场预测复用现有 `match_predictions`**（`app/models/predictions.py:7-52`，形状完全兼容；唯一约束 `:46-50`）— 不建新表。
- 新建 `app/models/national_*.py`（继承 `Base`）：
  - `national_standings`：`competition_id, season_id, group_label(A-L), team_id, team_name, played, won, drawn, lost, gf, ga, gd, points, rank`。
  - `national_bracket`：`season_id, round(R32/R16/QF/SF/F), match_no, team1, team2, winner, source_match1_no, source_match2_no`（matches 表 TBD 槽位被跳过，必须独立存）。
  - `national_champion_odds`：`season_id, computed_at, n_sims, team_id, team_name, p_advance, p_win_group, p_r16, p_qf, p_sf, p_final, p_champion`（镜像 `/national/simulate` 的 teams 行；带时间戳可留历史）。
- **迁移**：① 在 `migrations/env.py:11-24` import 新模型（**否则 autogenerate 漏检**；注意 `live_streams` 当前就漏在 import 外，别学）；② `uv run alembic revision --autogenerate -m "national tables"`（产出 `012_*`，`down_revision="011_live_streams"`）；③ 人工核对 JSONB/约束；④ `uv run alembic upgrade head`。参考最近的新表迁移 `migrations/versions/011_live_streams.py`。

### 2.5 对外 REST API `app/api/v2/national.py`（挂进 `__init__.py:25-39` 的 `_fe`，`verify_frontend_key`）
- 单场预测：**直接复用现有 `GET /api/v2/matches/{match_id}/predictions?source=national`**（`app/api/v2/predictions.py:12-23`，返回 `PredictionSchema`，`app/schemas/matches.py:216-243`）— 前端零改动解析。
- 赛程：前端用现有 `GET /api/v2/matches?competition=WOC` 即可（WOC 在 `COMPETITIONS`，默认列表已含，`matches.py:75,86-88`）。
- 新增：`GET /v2/national/standings`（12 组积分，形状复用 `standings.py` 的 `group_name` 分组）、`GET /v2/national/bracket`（对阵树）、`GET /v2/national/champion-odds`（夺冠/晋级概率榜，镜像 simulate teams 行）。
- CORS 已允许 `X-API-Key` + `GET/POST`（`app/main.py:78-79`），无需改。

### 2.6 实时推送（WS）— 复用
WS `/ws/match/{opta_match_id}`（`app/ws/routes.py:32`），listener psubscribe `match:*:*`（`app/ws/manager.py:81-108`），俱乐部滚球预测发 `match:{opta_id}:tika_prediction`（`trigger.py:238`）。**国家队滚球只要发同频道同形状即可，前端 WS 客户端零改动收到**。夺冠概率更新走轮询 `/v2/national/champion-odds`（WS manager 只匹配 `match:*:*`，全局类推送不必上 WS）。

---

## 3. ai-service 执行计划

> 现状：5 种分析（`prediction_interpretation`/`summary`/`discussion`/`pre_match_preview`/`post_match_report`），OpenRouter（`app/llm/client.py`，非 Anthropic 直连）。订阅 Redis `tika:*:prediction` + `match:*:{event,status_changed,lineups_ready,summary_timer}`（`app/worker/manager.py:64-70`）。写 `ai_results`、只读 `matches`。无国家队代码（仅 `context/match_context.py:22` 映射了 WC 竞赛名）。

### 3.0 唯一硬阻塞在 data-service（非本仓库）
ai-service 收不到国家队触发的根因是 data-service 永不 publish（§0 决策 1 / §2.3）。**前置条件 = data-service 发 `tika:{opta_match_id}:prediction`（结构对齐 `trigger.py:218-228`）**。ai-service 本身改动很小。

### 3.1 渲染 tournament_context — `app/context/tika_context.py`
扩展 `format_tika_prediction`（`tika_context.py:4`），识别消息里新增的 `tournament_context`（赛事阶段、夺冠/晋级概率），拼进给 LLM 的上下文。现 `tika_context.py` 只格式化 goals/corners/yellows，无赛事级字段。

### 3.2 背景层降级 — `app/context/match_context.py`
为国家队 `competition_id` 加分支（或在 `app/context/assembler.py:126 _build_match_background` 早返回简版）：**跳过** `team_feature_cache`/`referee_feature_cache`/联赛积分榜/按联赛取近5场（这些对国家队全空，`match_context.py:70-181/184/264`），改用"赛事+赛程阶段+小组积分/夺冠概率"简版背景。当前失败会被 try/except 吞成 None（`assembler.py:155-157`），不崩但背景缺失、质量降。

### 3.3 prompt 微调 — `app/prompts/`
`prediction_interpretation.py:7`、`pre_match.py:2-7`、`post_match.py` 加条件指引："若为国际赛事，引用夺冠/出线概率与赛事阶段，**不要假设联赛积分榜/裁判数据**"。LLM 不自造数字的硬约束保持不变。

### 3.4 类型与配置
- 复用现有 5 种 handler 的代码路径与 LLM 调用即可；如前端要区分，可新增 `ai_results.type` 值（如 `tournament_outlook`）— `ai_results.type` 是 `String(32)` 无 DB enum，可直接写（`app/models/ai_results.py:20`）。
- **接入无需新增环境变量**（除非要为国家队设独立开关/预算）。
- 验证：`redis-cli publish tika:{test_opta_id}:prediction '{...goals + tournament_context...}'` → 应触发 `prediction_interpretation` 并写 `ai_results`（`manager.py:206-277`）。前置：该比赛在 `matches` 表有 `opta_match_id` 且 Redis 有 `match:{opta_id}:state`，否则 `manager.py:548-559` 查不到 match_id、背景全空。

---

## 4. 前端 web-sportalpha 执行计划

> 技术栈：Next.js 16 App Router + React 19 + SWR + Tailwind v4/shadcn + 自研 i18n(en/zh)。API base `NEXT_PUBLIC_API_URL=https://api.sportalpha.io`，走 `/api/v2/*` + `/static/*`（`src/lib/api.ts`）。**未用任何第三方/API-Football**。
> 现状：`/world-cup` 专题页（`app/world-cup/page.tsx` → `WorldCupContent`）有 Groups/Schedule/Teams 三 Tab，数据混合（分组结构硬编码于 `lib/world-cup.ts`，积分/赛程/名单来自 API）。**无对阵树、无夺冠概率榜、WC 赛程卡不可点击**（`schedule-view.tsx:46` `WcMatchCard` 无 onClick）。已有 hooks `useWcMatches/useWcTeams/useWcSquads`。

### 4.1 让 WC 单场预测可展示（核心，零新建预测组件 — 推荐路径 A）
国家队预测形状已镜像俱乐部（决策 2），故：
1. 把 `WOC` 加入联赛列表 `src/lib/mock-data/fixtures.ts:1-6`（现仅 EPL/LL/SEA/BUN/LI1）。
2. WC 赛程卡 `WcMatchCard`（`schedule-view.tsx:46-106`）加 `onClick` → 跳现有单场视图 `/dashboard?match=<opta_match_id>`（选中机制 `context/match-context.tsx:34-44`）。
3. 单场视图 `MatchShell`→`AIPredictionSection`（`components/match/ai-prediction.tsx:34`）**整套复用**：1x2 `ProbabilityBar`、推荐比分、大小球 popover、`ScoreHeatmap`(7×7)、滚球多快照 `SnapshotTabs`，数据走 `usePredictions(matchId)`（`source` 可不传或传 national）。
4. 滚球 WS 复用 `hooks/use-live-match.ts`（已处理 `tika_prediction`，`:245-252`），WC 比赛有 `opta_match_id` 即可。
> 路径 B（在 `/world-cup` 内嵌单场视图）需补挂 `MatchProvider`，工作量更大，不推荐。

### 4.2 世界杯特有、需新建
- **淘汰赛对阵树 bracket**：前端完全不存在。新建组件，数据来自新 hook → `GET /api/v2/national/bracket`（或淘汰赛阶段 `useWcMatches`）。可参考 `lib/world-cup.ts:84-91 WOC_STAGES` 的阶段定义。
- **夺冠/晋级概率榜**：前端无对应 hook/组件。`hooks/use-api.ts` 加 `useNationalChampionOdds()` → `GET /api/v2/national/champion-odds`，新建排行榜组件（各队 `p_champion`/`p_advance` 等）。
- **分组积分榜增强**：现 `GroupCard`（`group-card.tsx`）仅显 logo+队名+积分，升级为 W/D/L/GD/晋级位高亮，或复用完整表组件 `StandingsPanel`（`standings-panel.tsx`，有 `group_name` 时可用）。
- **东道主主场标记**：`MatchBrief`/`StandingsEntry` 无东道主字段 → 后端补字段或前端按 `WOC_TEAM_CODES`(USA/MEX/CAN) 硬编码标记。

### 4.3 展示一致性结论（回答用户问题）
**真实比赛开打后，WC 单场比赛页与五大联赛完全一致**（同 `AIPredictionSection`/`ScoreHeatmap`/滚球，因预测形状镜像）。WC **专题层**（分组积分、对阵树、夺冠榜）是五大联赛没有的赛事级视图，需新建，但单场点进去后体验与联赛统一。logo/中文名/时间/比分数据源与组件均已就绪（`/static/logos/{team_id}.png` 已有 WC 队徽）；`venue_name` 字段在类型里但 UI 未渲染（`match-header.tsx`），WC 若要显示球场需补渲染 + 后端填充该字段。i18n 已有 `worldCup.*` 键（`messages/en.json:200-231`/`zh.json`），新组件沿用 `t()`。

---

## 5. 实施顺序（跨仓库依赖）

1. **[主仓库]** 部署 `/national/*` 到服务器并冒烟测试（§1）。← 无依赖，先做。
2. **[data-service]** 队名归一化（§2.2）+ 核对竞赛 ID（§2.0）。← 正确性前置。
3. **[data-service]** 扩展 client（§2.1）+ 独立赛前/模拟管道 + 新表/迁移 + `/v2/national` 端点（§2.3–2.5）。
4. **[data-service]** publish `tika:{opta_id}:prediction`（含 `tournament_context`）+ WS（§2.3）。← 解锁 ai-service 与前端滚球。
5. **[ai-service]** tournament_context 渲染 + 背景降级 + prompt 微调（§3）。← 依赖第 4 步。
6. **[前端]** 路径 A 接单场预测（§4.1）← 依赖第 3 步的 `/matches?competition=WOC` + predictions。
7. **[前端]** 对阵树 + 夺冠榜 + 积分榜增强（§4.2）← 依赖第 3 步的 `/v2/national/*`。
8. **[data-service]** 滚球分流（§2.3）+ 赛事模拟定时（§2.3）。← 开赛前到位。

第 1 步可立刻做；2–4 是 data-service 的主体工作（也是 ai-service/前端的共同前置）；5/6/7 可在 3/4 完成后并行。

---

## 6. API-Football 评估结论（回答用户问题 4）

**结论：当前架构不需要 API-Football，作 Opta 国际赛实时不可靠时的后备。**

依据（均经代码/实测）：
- 前端所有比赛元数据（logo/赛程/比分/积分/名单/滚球）**已由 data-service(Opta) 提供**，WC 队徽 `/static/logos/` 已就绪（前端 grep `api-sports/api-football` **零命中**）。
- 模型**自包含**，预测不需要外部特征。
- API-Football **赔率结构性拿不到**（WC 全赛季 odds=False，连 2022 终态也 False；记忆 `reference_international_data_sources.md`），所以它补不了我们唯一缺的赔率。
- 当前 key 套餐 **2026-06-07 到期，早于开赛 06-11**，若用需续费。
- 引入第二数据源会破坏"与五大联赛统一 UI/单一数据源"的架构，且增加归一化负担。

**唯一值得用 API-Football 的情形**：Opta 逆向对**国际赛**的赛程/赛果/淘汰赛对阵/live events/lineups/standings 不可靠（俱乐部逆向成熟，但国家队大赛逆向未经实战检验）。届时把 API-Football（`league=1, season=2026`，文档见本地 md）作为**这些展示数据的后备源**，**藏在 data-service 现有 `/api/v2/*` 形状之后**（前端不感知、不改），**不取赔率**，注意 key 到期。**不要在前端嵌 API-Football widget**（破坏统一 UI）。

可选小用法：API-Football 自带 `/predictions` 端点，可作我们模型预测的**对照基准**（记忆已记），非必需。

---

## 附：各仓库证据索引（关键 file:line）
- 主仓库：`src/server.py:74,114,133,475`、`src/national_api.py:59-92,143-199`、`src/national_simulate.py:287-293`、`docker-compose.yml`
- data-service：`app/predictions/trigger.py:71-83,101-111,218-268`、`app/predictions/client.py:21-162`、`app/opta/constants.py`、`app/collector/{season_sync.py:109-112,manager.py:75-112,match_collector.py:585}`、`app/models/predictions.py:7-52`、`app/api/v2/{predictions.py:12-23,__init__.py:25-39}`、`app/schemas/matches.py:216-243`、`app/ws/{routes.py:32,manager.py:81-108}`、`migrations/env.py:11-24`、`app/config.py:66-70`
- ai-service：`app/worker/manager.py:64-70,206-277,548-559`、`app/context/{tika_context.py:4,match_context.py:15-23,70-181,assembler.py:126-157}`、`app/models/ai_results.py:12-45`、`app/llm/client.py`、`app/redis_client.py:19`
- 前端：`src/lib/{api.ts,types/api.ts:222-251,world-cup.ts,mock-data/fixtures.ts}`、`src/hooks/{use-api.ts,use-live-match.ts:245-252,use-fixtures.ts}`、`src/components/match/{ai-prediction.tsx:34,score-heatmap.tsx}`、`src/components/world-cup/{world-cup-content.tsx,schedule-view.tsx:46,groups-overview.tsx,group-card.tsx}`、`src/app/world-cup/page.tsx`
