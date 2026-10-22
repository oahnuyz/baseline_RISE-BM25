# RISE-BM25 on EnterpriseRAGBench

运行 EnterpriseRAGBench 的 **RISE-BM25：无 TOC、保留 reasoning_content**。支持 10,970 / 511,959 篇文档，两种规模使用相同的 500 道 QA，默认题目并发为 4。

基于 [官方 RISE 固定提交](https://github.com/texttron/RISE/tree/32672b59ead8381decae7d412f05bb53d399946a) 和[论文](https://arxiv.org/abs/2606.06880)。保留官方全文档 BM25（k1=1.5、b=0.75）、每查询 top-1000 召回、top-10 预览、累积工作目录、原无 TOC prompt 及 search/bash/read 工具；不切 chunk、不使用 embedding 或 reranker。

与论文设置的主要区别：使用 DeepSeek 和 EnterpriseRAGBench；开启预算耗尽后的强制回答；采用 0–4 分 judge；使用 Linux 沙箱。另修复了 read 的目录边界校验。保留推理历史本身是官方行为。QA 温度 0、max_tokens=32000、100 轮探索，必要时再追加一轮最终回答；上下文不主动压缩。单次 HTTP 超时 900 秒，3600 秒问答预算在轮次边界检查。

## 安装

需要 Linux x86_64、Python 3.10、可用的 bubblewrap（原环境 0.6.1）以及 Bash / GNU 常用命令。以下命令均在本仓库根目录执行，虚拟环境需放在 `.venv`：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r adapters/requirements.txt
mkdir -p tools
curl -fL https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz -o tools/rg.tar.gz
echo '4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e  tools/rg.tar.gz' | sha256sum -c -
tar -xzf tools/rg.tar.gz -C tools
cp tools/ripgrep-14.1.1-x86_64-unknown-linux-musl/rg tools/rg
bwrap --version
```

`bwrap` 需允许创建用户／网络命名空间；缺失时由服务器管理员安装。运行器会用它隔离模型执行的 bash 命令。

## 数据

**已有 prepared 数据时直接使用其目录，跳过下载和导出。** 目录须含 `corpus/`、`documents.jsonl`、`qa.jsonl`。运行前会校验文档正文哈希和问题集合；gold 和证据标注不会进入 Agent 的检索范围。

从公开数据准备：

```bash
.venv/bin/python -B reproduce.py download --out raw
.venv/bin/python -B reproduce.py prepare --raw raw --scale 10970 --out prepared/enterprise_10970
.venv/bin/python -B reproduce.py prepare --raw raw --scale 511959 --out prepared/enterprise_511959
```

来源固定为 [BM25-Wins-at-Scale](https://github.com/BstWPY/BM25-Wins-at-Scale/tree/0626d4f4019c6a04b891b7453dd93a3491999584) 的[公开语料包](https://huggingface.co/datasets/Bstwpy/BM25-Wins-at-Scale-Corpus/tree/a00241078f59719a06770a0d6eea10067690d76a)，下载时检查 SHA-256。输入 Parquet 约 1.38 GB，全量正文约 2.48 GB、511,959 个文件；运行还需保存输入副本、索引和 trace。

`data/` 只保存复现必需的文档 ID、QA 子集 ID 和校验值，不含历史实验结果。导出保留文档顺序、路径与正文，不转换正文中的字面量 `\n`；两篇 scaffold 文档沿用原导出 `.strip()` 约定。失败时保留 `.part` / `.building` 供检查，不自动覆盖。

## API 与运行

复制并编辑凭据文件，填写自己的服务地址和密钥：

```bash
cp credentials.example.json credentials.local.json
chmod 600 credentials.local.json
```

请求模型为 `deepseek-v4-flash`，响应模型检查值为 `deepseek-v4-flash-ga-260731`。API 需支持 OpenAI 兼容 Chat Completions、工具调用、接收历史 reasoning_content 和完整 usage。不发送推理开关参数，保留服务端返回的推理内容。使用不同模型或服务端版本时，应作为不同模型条件记录。

生成两个 500 QA 运行计划，默认并发 4。若使用已有数据，把 `--data` 换成相应绝对路径；该步骤不调用 API：

```bash
.venv/bin/python -B reproduce.py plan --scale 10970 --subset all500 --data prepared/enterprise_10970 --run-root runs/enterprise_10970_500qa --credentials credentials.local.json --rg tools/rg --out plans/enterprise_10970_500qa.json
.venv/bin/python -B reproduce.py plan --scale 511959 --subset all500 --data prepared/enterprise_511959 --run-root runs/enterprise_511959_500qa --credentials credentials.local.json --rg tools/rg --out plans/enterprise_511959_500qa.json
```

启动实验（会真实调用 API）：

```bash
mkdir -p runs/enterprise_10970_500qa runs/enterprise_511959_500qa
nohup .venv/bin/python -u -B adapters/experiment/launch.py --plan plans/enterprise_10970_500qa.json > runs/enterprise_10970_500qa/launcher.log 2>&1 < /dev/null &
nohup .venv/bin/python -u -B adapters/experiment/launch.py --plan plans/enterprise_511959_500qa.json > runs/enterprise_511959_500qa/launcher.log 2>&1 < /dev/null &
```

两个规模独立运行。启动器先执行真实 API / 工具预检，再完成正式问答和评分。查看各运行目录的 `state.json`、`run.log` 和数据集子目录的 `summary.json`。完成时应为 `accuracy_is_final=true` 且 `successful_qa=scored_questions=500`。

其他选项：`--concurrency` 设置题目并发；`--subset original100` 使用原固定 100 题；`remaining400` 使用其补集。更改配置应使用新的计划文件和运行目录。中断后，待原进程退出，用原计划重新启动即可跳过已完成题；评分失败只补评分。

## 结果与计时

```bash
.venv/bin/python -B reproduce.py summarize --runs runs/enterprise_10970_500qa
.venv/bin/python -B reproduce.py summarize --runs runs/enterprise_511959_500qa
```

汇总器逐题核查完整性；同一规模若分子集运行，可在 `--runs` 后传入多个不重叠的目录。

| 指标 | 口径 |
|---|---|
| accuracy | [固定 judge prompt](assets/judge.txt) 给出 0–4 分，取平均后除以 4；不是全部答对的题目比例 |
| 平均 token | 有效 QA attempt 的 API 输入（含缓存）＋输出；不重复添加已包含的推理 token |
| 平均问答时间 | 本题首次实际 HTTP 发送到最终完整回答返回；包含其间工具执行和重试，不含评分和开始前排队 |
| 平均 iteration | 模型响应轮数，包含最终回答轮；不是工具调用次数 |

执行完成但答错的题仍纳入统计；预检、失败尝试、judge 成本不计入主问答均值。完整 trace 保存在 `runs/<run>/enterprise_rag_bench_<scale>/qa/<题目ID哈希>/attempt_*/`，包括 `native_full.json`、`http.jsonl`、`tools.jsonl`、`retrieval.jsonl` 和 `workspace.jsonl`；上层 `result.json` 保存原题目 ID 和有效结果。

原执行器的 `summary.json.build_time` 只计 BM25。需要完整入库时间时，在实验结束、没有运行进程后执行：

```bash
.venv/bin/python -B reproduce.py measure --plan plans/enterprise_10970_500qa.json
.venv/bin/python -B reproduce.py measure --plan plans/enterprise_511959_500qa.json
```

读取 `runs/<run>/ingestion_timing_v2.json` 的 `total_ingestion_seconds`：重新生成 Parquet＋构建 BM25，不覆盖正式索引，并核对新旧产物和检索结果。不计下载、输入副本、预检与校验时间；不清空 OS 缓存。入库 API token 为 0。模型生成具有随机性，耗时也受服务负载和硬件影响。

## 发布范围与许可

以本目录作为 GitHub 仓库根目录，只提交代码、依赖清单、judge prompt、凭据示例、必要数据 ID／校验值和本 README。`.gitignore` 排除语料、索引、运行结果、trace、虚拟环境、二进制和个人凭据。

源码中保留了被运行器导入的共用模块和 `section_one_doc.py`；无 TOC 运行不会执行章节生成。RISE 代码许可见 [LICENSE](LICENSE)，数据许可见 [data/LICENSE](data/LICENSE)。
