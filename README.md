# Stanford CS329A · 自我改进智能体 HW3

打开 `homework3.ipynb`，可以从一个直接回答问题的模型开始，逐步加入工具路由、并行拆题、候选融合和迭代补查，再生成带来源的研究报告。源代码中的中文注释解释了工具结果如何成为下一步决策的依据。[原始作业](https://github.com/stanford-cs329a/cs329a-homework3-fall2025-public)

## 运行 Notebook

在仓库根目录使用 Python 3.10 或更高版本。下面是 Windows PowerShell 命令。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
if (!(Test-Path .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
.\.venv\Scripts\python.exe -m notebook homework3.ipynb
```

已有 `.env` 时跳过复制模板的命令。在本地填写凭据后按顺序运行单元。语言模型使用 `TOGETHER_API_KEY`。美股工具接受 `MASSIVE_API_KEY` 或原来的 `POLYGON_API_KEY`，数学工具使用 `WOLFRAM_APP_ID`。网页搜索优先使用已有的 `GOOGLE_API_KEY` 和 `GOOGLE_CX_ID`，随后尝试免密钥的 DDGS。两者都没有结果时查询英文维基百科，结果中的 `source` 字段会标明检索范围。

天气工具可以独立运行。Nominatim 将城市名转换为经纬度，地理数据来自 OpenStreetMap 贡献者。公共服务按每秒至多一次请求限速，同一城市的地理编码可复用。Open-Meteo 返回对应当地时间的气象数据。[Nominatim 使用政策](https://operations.osmfoundation.org/policies/nominatim/)

**使用 llmtoken 等兼容接口。** 在本地配置服务商的 HTTPS 地址 `CS329_BASE_URL` 和该站点专用的 `CS329_API_KEY`，然后设置 `CS329_MODEL` 和三个不同的 `CS329_CANDIDATE_MODELS`。模板中提供了 llmtoken 的完整配置示例。llmtoken 公布的 OpenAI 接口地址是 `https://llmtoken.io/v1`。候选模型各自阅读同一批工具证据，融合模型负责比较它们的回答。[llmtoken 官方接入说明](https://llmtoken.io/)

原题指定的多个模型端点已经下架。本实现通过 `cs329a_hw3/utils.py` 设置替代模型。通过 `CS329_MODEL` 可以设置通用模型，也可用 `CS329_GENERATION_MODEL`、`CS329_ROUTER_MODEL`、`CS329_DECOMPOSITION_MODEL`、`CS329_REFINEMENT_MODEL`、`CS329_FUSION_MODEL` 和 `CS329_JUDGE_MODEL` 单独指定角色。Together 的路由默认使用 JSON Schema；其他服务默认通过提示要求 JSON，再由 Pydantic 验证。若端点支持，可以把 `CS329_STRUCTURED_OUTPUT` 设为 `json_object` 或 `json_schema`。三个候选模型名用英文逗号分隔。修改配置后重启内核。配置单元会打印本次请求的模型标识和接口，便于记录实验条件。[Together 模型目录](https://docs.together.ai/docs/serverless/models) · [下架记录](https://docs.together.ai/docs/deprecations)

## 结合讲义读代码

| 作业 | 实现入口 | 对应知识 |
| --- | --- | --- |
| Part 0 | `MultiLMAgent.generate`、`evaluation.evaluate_qa` | 第 3、22–24 页。直接作答建立基线，裁判只在回答产生后看到参考答案。 |
| Part 1 | `APIManager.route_query`、四种工具、`single_LM_with_single_API_call` | 第 30–33 页。先决定采取什么行动，再用外部返回的数据作答。 |
| Part 2a–c | `decompose_query`、`_get_synthesis_prompt`、`decompose_and_fuse` | 第 40–44、18 页。独立问题可以并行查证；多个模型阅读同一批证据，再由融合模型核对分歧。 |
| Part 2d、3 | `iterative_refine`、`run_pipeline` | 第 30–31、63–66 页。后一步依赖前一步结果时，先获得观察，再生成下一个请求。 |
| Part 4 | `DeepResearchAgent.research` | 第 64–66、74–75 页。从子问题检索来源，保留时间信息，把报告中的主张与可核查的出处连接起来。 |

## 评估

原始 Notebook 的 Part 0 基线和路由示例使用 Llama 3.1 8B Instruct Turbo，Part 1f 单工具示例和 Part 3 全量基线使用 Llama 3.3 70B Instruct Turbo Free。Part 3 指定 70B 做拆题和迭代，8B 做最终融合。Part 2c 指定 Gemma 3n E4B、Llama 3.1 8B、GPT-OSS 20B 分别生成候选；简答评测使用 Gemma 3n E4B。Part 4 的单模型对照使用 Llama 3.1 8B，研究报告本身可选模型。[上游原题](https://github.com/stanford-cs329a/cs329a-homework3-fall2025-public/blob/main/homework3.ipynb)

Part 0–2 的四种方法共用 `COMPARISON_DEBUG` 开关。默认每类取两题，共 8 题。切换为 `False` 并重新运行比较单元，可以得到同一批 15 题上的四个分数和柱状图。Part 3 总是在全部 15 题上评估基线和完整流程，至少答对 11 题才超过原题的 70% 门槛。评审模型未返回有效判定时，Notebook 会暂停，便于检查服务配置后重新测量。

Part 3 的规划器会按地点、交易日和当地日期拆分独立查询。比较全天风速时，每座城市的记录含当天逐小时数据的平均值、最大值和有效样本数。行情首查失败时，Part 3 还会查询 Yahoo Finance 的公开日线，核对股票代码、币种和纽约交易日，保留价格口径、来源链接及首查错误。Wolfram 不可用时，能由已知条件推出的数值可以由模型独立计算，输出会标明验证状态。Part 1 的股票和数学示例分别调用原题的行情与 Wolfram 接口，可以单独检查这两个服务是否接通。

运行下面的命令可单独检查路由、证据传递、轮数上限、报告引用和故障处理。测试使用固定响应，因此在没有 API 凭据的环境也能运行；问答准确率由在线 Notebook 实验给出。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 2026-11-26 的 llmtoken 实测

[打开带逐题答案、评审结果、对比图和三份研究报告的 Notebook](results/llmtoken.ipynb)。公开副本省略了工具演示中重复打印的第三方网页摘录，保留标题和来源链接。

| Part 0–2 同一组 8 题 | 答对 | 准确率 |
| --- | ---: | ---: |
| 零样本 | 8/8 | 100% |
| 单工具 + 单模型 | 2/8 | 25% |
| 拆题 + 多模型融合 | 5/8 | 62.5% |
| 迭代补查，最多 4 轮 | 5/8 | 62.5% |

全量 15 题上，零样本模型答对 **15/15**。第一版完整流程答对 **6/15**。改进按日期拆题、天气日统计、历史行情补查和工具失败后的处理后，使用同一裁判对 15 题全部重新生成、评审，答对 **13/15（86.7%）**。原题给出的全量满分门槛是超过 70%。这是按公开题集参考答案计算的模型裁判分数。

全量零样本基线沿用了调试集中 8 道相同题目的原回答及原判定，并新跑另外 7 题。Part 3 的两次完整流程各自独立测完 15 题。改进后仍失分的两题是全天风速比较和电影片名。模型按取得的全天风速均值判断芝加哥稍高，但参考答案为波士顿；电影检索返回同名作品，因此模型未能确认目标片名。

| 实验角色 | 上游作业中的模型 ID | 本次发送给 llmtoken 的模型 ID |
| --- | --- | --- |
| Part 0 零样本、Part 1e 路由与参数提取、最终融合、Part 4 单模型对照 | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | `deepseek-v4.1-flash` |
| Part 1f 单工具作答、拆题和迭代补查、Part 3 全量零样本 | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | `deepseek-v4.1-flash` |
| Part 2c 三个融合候选 | `google/gemma-3n-E4B-it`、`meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`、`OpenAI/gpt-oss-20B` | `deepseek-v4.1-flash`、`qwen3.8`、`gpt-5.6-luna` |
| 简答题裁判 | `google/gemma-3n-E4B-it` | `deepseek-v4.1-flash` |
| Part 4 研究计划与报告综合 | 作业允许自选 | `deepseek-v4.1-flash` |

网关在成功响应的 `model` 字段中分别回报 `deepseek-v4.1-flash`、`qwen3.8-max`、`grok-4.7`。后两项分别对应请求别名 `qwen3.8` 和 `gpt-5.6-luna`。这些名称来自回包，网关背后的实际部署无法独立核实。生成和裁判使用同一个 DeepSeek 标识，这一点也影响与原作业实验的可比性。

## 阅读三份研究报告时可核对的细节

三道原题各有一份单模型回答和一份检索报告。按固态电池、英国移民、核聚变的顺序，报告分别为 5 段/4 个来源、5 段/5 个来源和 4 段/8 个来源。下面的核查补充独立于模型生成的报告。

- **固态电池。** QuantumScape 在 2024 年 10 月 23 日的股东信披露，QSE-5 B 样电芯开始小批生产并交给汽车客户测试。公司报告的体积能量密度超过 800 Wh/L，从 10% 充至 80% 用时不足 15 分钟。该信也列出良率、可靠性和设备产能方面的量产挑战。[QuantumScape 股东信，第 2 页](https://ir.quantumscape.com/static-files/8345451d-6901-42e0-838b-37c95cfa1da8)
- **英国移民政策。** 报告援引的就业率来自 2021 年人口普查，统计范围是英格兰和威尔士的 16–64 岁人口。另一项可以单独核实的政策变化发生在 2024 年 4 月 4 日，适用 Skilled Worker 新规则的一般年薪门槛从 £26,200 升至 £38,700，医疗护理等类别及既有签证持有人另有规定。[英国国家统计局](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/articles/migrationandthelabourmarketenglandandwales/census2021) · [英国内政部](https://www.gov.uk/government/news/new-laws-to-cut-migration-and-put-british-workers-first-in-force)
- **核聚变。** JET 取得 69 MJ 聚变能的实验发生在 2023 年 10 月 3 日，结果于 2024 年 2 月 8 日公布。另一个实际发生于 2024 年的例子是 NIF 在 11 月 18 日用 2.2 MJ 入靶激光能量取得 4.1 MJ 聚变产额。后者按入靶能量计算增益，整座设施的耗电量属于另一项指标。[JET 实验资料](https://www.ipp.mpg.de/5405892/jet_rekord_2024) · [UKAEA 公告](https://www.gov.uk/government/news/jets-final-tritium-experiments-yield-new-fusion-energy-record) · [LLNL 实验记录](https://lmf.llnl.gov/science/achieving-fusion-ignition)
