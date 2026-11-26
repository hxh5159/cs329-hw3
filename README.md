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

原题指定的多个模型端点已经下架。本实现通过 `cs329a_hw3/utils.py` 设置替代模型。通过 `CS329_MODEL` 可以设置通用模型，也可用 `CS329_GENERATION_MODEL`、`CS329_ROUTER_MODEL`、`CS329_DECOMPOSITION_MODEL`、`CS329_REFINEMENT_MODEL`、`CS329_FUSION_MODEL` 和 `CS329_JUDGE_MODEL` 单独指定角色。路由模型需要支持 JSON Schema。三个独立生成候选的模型由 `CS329_CANDIDATE_MODELS` 指定，用英文逗号分隔。修改配置后重启内核。配置单元会打印实际模型，便于记录实验条件。[Together 模型目录](https://docs.together.ai/docs/serverless/models) · [下架记录](https://docs.together.ai/docs/deprecations)

## 结合讲义读代码

| 作业 | 实现入口 | 对应知识 |
| --- | --- | --- |
| Part 0 | `MultiLMAgent.generate`、`evaluation.evaluate_qa` | 第 3、22–24 页。直接作答建立基线，裁判只在回答产生后看到参考答案。 |
| Part 1 | `APIManager.route_query`、四种工具、`single_LM_with_single_API_call` | 第 30–33 页。先决定采取什么行动，再用外部返回的数据作答。 |
| Part 2a–c | `decompose_query`、`_get_synthesis_prompt`、`decompose_and_fuse` | 第 40–44、18 页。独立问题可以并行查证；多个模型阅读同一批证据，再由融合模型核对分歧。 |
| Part 2d、3 | `iterative_refine`、`run_pipeline` | 第 30–31、63–66 页。后一步依赖前一步结果时，先获得观察，再生成下一个请求。 |
| Part 4 | `DeepResearchAgent.research` | 第 64–66、74–75 页。从子问题检索来源，保留时间信息，把报告中的主张与可核查的出处连接起来。 |

## 评估

Part 0–2 的四种方法共用 `COMPARISON_DEBUG` 开关。默认每类取两题，共 8 题。切换为 `False` 并重新运行比较单元，可以得到同一批 15 题上的四个分数和柱状图。Part 3 总是在全部 15 题上评估基线和完整流程，至少答对 11 题才超过原题的 70% 门槛。评审模型未返回有效判定时，Notebook 会暂停，便于检查服务配置后重新测量。

运行下面的命令可单独检查路由、证据传递、轮数上限、报告引用和故障处理。测试使用固定响应，因此在没有 API 凭据的环境也能运行；问答准确率由在线 Notebook 实验给出。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
