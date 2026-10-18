# 与直接相关工作区别

核查日期2026-10-06；以下来自作者arXiv出版页摘要，不将他人任务指标作为本任务性能目标。

| 文献 | 已有方法/发现 | 本原型的边界 |
|---|---|---|
| Jayalath & Parker Jones, 2026, Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text, https://arxiv.org/abs/2609.40359 | 相邻词窗口的重叠能泄露词间时间，噪声取得接近真实脑信号的词分类结果；改为独立处理词窗 | 我们先冻结同长度、同掩码、固定间隔输入；不将重叠时距当作EEG信息。本原型的图由固定物理支持定义，不从时长估计；任务是中文行级surprisal而非听觉词重建 |
| Banville et al., 2019, Self-supervised representation learning from electroencephalography signals, https://arxiv.org/abs/1911.05419 | 通过判断时间窗是否属于同一时间上下文学习无标签EEG表示，在睡眠评分中验证 | 我们约束同一真实采样时刻在不同重叠窗口中的特征表示，已知支持坐标；目前为监督模型内的残差原型，未做自监督预训练 |

同时间上下文、一致性学习、图平滑和残差更新都已有先例。当前结果不足以确立新颖性，更不能声称首次提出。需要真实支持、随机支持、同容量平滑以及无脑信息控制的证据后，才判断该具体组合是否形成有效算法贡献。当前初筛未通过晋级，不启动额外随机图/噪声训练。
