# F1 Circuit Dashboard — 数据说明

基于 `Concept Pitch`、`511_FDS`、`Participant Notes` 整理。  
目标产品：**以赛道为中心的 Streamlit 仪表盘**，核心模块是 **Grid Position Explorer** 与 **Segment Analysis**。

---

## 1. 文档需求 → 数据评估

| 功能（来自文档） | 需要的字段 | 采用的数据源 | 结论 |
|---|---|---|---|
| Grid：发车位 × 赛道的 Win% / Podium% / Avg Finish | `grid`, `position`, `circuit`, `year`, driver/constructor | `data/raw/racedata/`（RaceData = Kaggle 同构，已更新到当前赛季） | **已下载，足够** |
| 赛道 / 车手 / 车队元数据 | circuits, drivers, constructors | 同上 | **已下载** |
| Segment：扇区时间分布、车手 vs 全场 | `Sector1/2/3Time`, driver, circuit, year | `tif1` → `data/raw/tif1_sectors/` | **脚本已就绪，按年下载** |
| 轮胎 / 天气上下文（辅助 Segment） | Compound, TyreLife, AirTemp, TrackTemp, Rainfall | 同上（tif1 laps 自带） | **随扇区表一起有** |
| 圈速历史分布 | lap times | racedata `lap_times.csv` + tif1 sectors | **已有** |
| DNF / 事故（整场级） | `statusId` / status 文本 | racedata `results.csv` + `status.csv` | **已有（整场级）** |
| DNF / 超车 **按赛段定位** | segment-level crash/overtake | 公开源几乎没有稳定历史表 | **暂不做 / 标为缺口** |
| 完整遥测（油门/刹车/轨迹） | telemetry JSON | TracingInsights 年仓库（约 150GB） | **文档已砍掉，不下** |
| OpenF1 API 扇区 | 与 Concept Pitch 原计划一致 | 比赛进行中会 401；平时可作补充 | **备选，不作为主源** |

### 不下载的原因（与文档一致）

- FDS / Zhibo 笔记明确：**Discard detailed engineering / live telemetry（太大太复杂）**
- Participant Notes 评估任务只依赖：发车位指标 + Segment 时间对比
- 按赛段的超车/事故从 1950 年起公开数据不可靠，不纳入 MVP

---

## 2. 已落地的目录结构

```text
finalproject/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── racedata/          # 历史成绩（Grid Explorer 主源）
│   │   │   ├── results.csv
│   │   │   ├── races.csv
│   │   │   ├── circuits.csv
│   │   │   ├── drivers.csv
│   │   │   ├── constructors.csv
│   │   │   ├── lap_times.csv
│   │   │   ├── pit_stops.csv
│   │   │   ├── qualifying.csv
│   │   │   ├── status.csv
│   │   │   └── ...
│   │   └── tif1_sectors/      # Race 扇区圈速（Segment Explorer 主源）
│   │       └── race_sector_laps_YYYY.csv
│   └── processed/             # 后续聚合表（Win%、分段分布等）放这里
└── scripts/
    ├── download_racedata.py
    └── download_sector_laps.py
```

---

## 3. 数据源详情

### A. RaceData（Grid Explorer）

- 来源：https://github.com/TracingInsights/RaceData  
- 发布包：https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip  
- 实质：Kaggle F1 数据集的持续更新镜像（CC0），覆盖约 **1950–当前赛季**
- 与 Concept Pitch 中的  
  https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020  
  **同族表结构**，并已解决「只到 2024」的缺口

**Grid 指标怎么算（与 FDS / 评估任务对齐）**

- Win% / P1% = 从该 `grid` 出发且 `position == 1` 的场次 / 有效样本量  
- Podium% = finish in {1,2,3} / 有效样本量  
- Avg Finish = 对完赛名次求平均（DNF 是否计入需单独处理；评估报告也提到这一点）  
- 务必展示 **样本数（Races）**，避免小样本误导  

**清洗注意：** `results.csv` 里 `grid` / `position` 常为字符串；分析前先 `pd.to_numeric(..., errors="coerce")`。墨西哥站在 circuits 里的 `location` 是 `Mexico City`，官方名是 `Autódromo Hermanos Rodríguez`（评估报告里新人容易看不懂，建议 UI 显示常用地名）。

刷新：

```powershell
py -3 scripts/download_racedata.py
```

### B. tif1 扇区圈速（Segment Explorer）

- 库：`tif1`（TracingInsights CDN，2018 起有 timing/sector）  
- 本地文件：`data/raw/tif1_sectors/race_sector_laps_YYYY.csv`  
- 关键列：`Sector1Time`, `Sector2Time`, `Sector3Time`, `LapTimeSeconds`, `Driver`, `Team`, `Compound`, 天气字段等  
- 对应评估任务：Hamilton 在 Mexico City **Segment 3** 相对全场更快/更慢

下载 / 续传（已存在的年份会跳过）：

```powershell
py -3 scripts/download_sector_laps.py --start 2018 --end 2025
```

强制重下某年：

```powershell
py -3 scripts/download_sector_laps.py --start 2024 --end 2024 --force
```

### C. OpenF1（备选）

- 文档原计划源：https://openf1.org/  
- 有扇区、mini-sector、进站等；**2023 起**  
- 注意：官方比赛窗口内会限制未登录访问（返回 401，需 API key）  
- 当前项目 **不依赖** OpenF1；tif1 已覆盖 Segment 所需扇区时间

---

## 4. 与评估任务的对应关系

| Participant Notes 任务 | 数据路径 |
|---|---|
| Mexico City，P2 胜率 | `racedata/results` ⋈ `races` ⋈ `circuits`，过滤 Mexico，`grid==2` |
| Hamilton，Mexico Segment 3 vs 全场 | `tif1_sectors` 中 Mexico City 的 `Sector3Time` |
| Monaco，P7 平均完赛 | 同 Grid 路径，`grid==7` |

---

## 5. 后续加工建议（尚未生成）

放到 `data/processed/`：

1. `grid_circuit_metrics.csv` — 每条赛道 × 发车位的 Win%/Podium%/AvgFinish/n  
2. `sector_distributions.parquet` — 每条赛道 × 扇区的全场与分车手时间分布  
3. （可选）电路官方名 → 常用地名映射表，解决评估里 “Autódromo Hermanos Rodríguez” 难懂的问题

---

## 6. 环境与启动雏形

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

### 初始原型包含什么

`app.py` 按 `F1_Combined_Dashboard_Mock.html` 的版式与作图编码实现：

1. **Header filters** — Circuit / Years / Driver / Team / Metric / Reset
2. **KPI** — # Crashes · Avg pit stops · Overtakes/race（后一项暂无稳定历史源）
3. **1 Grid Position Explorer** — 顺序色 heatmap · Selected info · small-multiples 水平条 · grid-vs 柱 + 全场虚线均值
4. **2 Segment Analysis** — MultiViewer 赛道图 · 最快纪录水平条 · 时间分布直方图 · DNF 折线（赛道级代理）
5. **3 Strategy** — 轮胎圈数柱 · 进站次数分布 · 轮胎构成堆叠 · 年均进站折线

Cloud 友好预聚合：`py -3 scripts/build_aggregates.py` → `data/processed/*.parquet`（约 2MB）。

本地已验证可用：`streamlit`、`tif1`、`pandas`、`plotly`、`pyarrow`。
