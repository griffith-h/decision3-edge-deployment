# 非 MLP 线性投影按行分块：独立复现

在已验证 RK3576 / ARM64 Debian13 / RKNN Runtime2.5.0、驱动0.9.8、Torch2.12CPU、Transformers5.17 上运行。先按 `../mlp-length-experiment/REPRODUCTION.md` 准备冻结 graph-half v1 发布与115/128/384行MLP图。新阶段不构建镜像、不转模型、不改厂商库；只复用已有公开RKNN矩阵乘接口。

## 路径与保护

- 线上服务 `decision3-edge-graph:8900` 和冻结发布 `/opt/decision3-edge-graph-20261018-v1` 不修改。
- 上轮目录 `/opt/decision3-edge-mlp-length-20261018` 在容器中只读挂载为 `/mlp`；发布源码/原生库/模型也只读。
- 本轮目录 `/opt/decision3-edge-row-projection-20261018` 可写挂载 `/experiment`，容器 `decision3-row-projection-test` 无发布端口、`--rm`、CPU4–7、三线程、8GiB上限。一次只运行一个板端测试。
- 源码仅覆盖原适配器已选中的42个非MLP语言大型投影；MLP、视觉、小型门控投影及分类头另有原路径。完整名称和K/N/分组见screen.json的inventory。
- 原engine.eligible上限仍512，原生调用每次不超过512。仅线性矩阵独立行拆块，组内原并行线程池与句柄复用；拼回原B/T/N后再执行完整注意力和扫描，不拆attention、不截断输入。
- 原型扩展保护范围为512<B*T<=2048；其他签名/更长输入保留旧路径。这个上限是本轮适配与验证范围，不是芯片硬件上限。

## 操作

本机仓库根目录：

```bash
python3 row-projection-experiment/preserve_service.py before
ssh root@192.168.1.164 'mkdir -p /opt/decision3-edge-row-projection-20261018/logs'
rsync -a --include='*.py' --include='*.sh' --include='config.json' --exclude='*' \
  row-projection-experiment/ root@192.168.1.164:/opt/decision3-edge-row-projection-20261018/
```

板端按顺序运行（每条结束再运行下一条）：

```bash
cd /opt/decision3-edge-row-projection-20261018
bash run.sh test_guards.py > logs/extra-guards.log 2>&1
bash run.sh test_native.py > logs/native-unit.log 2>&1
bash run.sh screen.py > logs/screen.log 2>&1
bash run.sh decompose.py > logs/decomposition.log 2>&1
bash run.sh screen_fused.py > logs/fused-screen.log 2>&1
# 先审阅分解与融合筛选：未通过的路线不能当作验收通过。
bash run.sh formal.py --candidate fused_guarded > logs/formal.log 2>&1
bash run.sh formal_fast_batch.py > logs/fast-batch.log 2>&1
```

模式：

- current：线上fixed115等价路径，其余形状沿原实现。
- mlp_only：上一轮MLP组合图，其他投影仍有512限制。
- single：MLP组合图 + 各非MLP投影独立拆块。
- chunked：MLP组合图 + 同输入投影成组拆块，保留原输出列打包/双核并行。
- fused：chunked + 恢复已有注意力预处理融合。只将其投影资格检查和group调用接入分块门面，原engine、raw MLP pipeline的资格上限不放宽，B=1等原融合保护仍保留。
- linear_only：保留原MLP实现，仅分块其他投影，用于定位舍入差异来源。
- fused_guarded：fused，另外对实际MLP张量B>1且B*T>512保留原MLP。该策略源自跨512多问题样本的预测偏差，不是将实验失败隐藏为通过。原始无保护路线和失败字段仍保留。

decompose.py对失败多问题输入保存完整输出、差异最大的字段和不同策略；另测自然语言B=2跨512输入。formal.py对115-token做20组交替对照；对513/615行各测5次MLP-only控制与20次候选（分5段交错），多问题保护路径各5次配对。两种样本量不可说成“所有长度20组配对”。每个输入重新计算current参照，首次准备和暖机不计入正式分布。另12个自然语言115-token请求各2次回归。

本机收尾（任何哈希扫描不得与计时重叠）：

```bash
rsync -a --exclude='__pycache__' root@192.168.1.164:/opt/decision3-edge-row-projection-20261018/logs/ row-projection-experiment/logs/
python3 row-projection-experiment/preserve_service.py after
python3 row-projection-experiment/capture_artifacts.py
python3 row-projection-experiment/validate_records.py
python3 row-projection-experiment/update_report.py
```

`update_report.py`先由上一阶段生成器重建旧报告，再插入本轮结果到`../mlp-length-experiment/report.html#row-projection`，保留先前失败路线、原始计时与精度记录。本轮不自动部署/提交/推送。

## 判读

- 每次测量核对answers预测字段：概率/score/noul最大绝对差<=0.01且choice不变；confidence和全部受检数值另列，排除legend元数据。score有自身量程，不能把score差异直接写成概率百分点。没有标注业务数据，不宣称业务准确率无损。
- 每次保存实际MLP B/T、投影形状、各块真实rows、42投影覆盖、范围/运行/保护回退、原engine CPU回退数。保护版多问题允许12层原MLP（36个旧投影）的CPU调用；其他42投影不得因为超512回退。
- 记录首次分配、常驻句柄数和校准数据；不缓存答案。不强制CPU/NPU/内存频率，分阶段绝对延迟可波动，只用同阶段比较；模型方法计时不是HTTP/排队/多请求SLA。
- 每块失败都等待原group线程完成，丢弃全部局部结果，整组用原CPU实现；运行失败禁用故障成员。范围转换、模型权重/偏置身份与版本、钩子、训练/梯度、不支持的dtype和形状均保留保护。
- 已测输入不能保证所有新文本均<=0.01。视频、并发吞吐、其他芯片、更换模型/SDK尚未验证。

快路径多问题另实跑20次，评估参照沿用formal.json同一冻结模型的current输出，不缓存候选答案。该样本分类概率差0.00971445、score差0.01745270（量程0–2），因此仅概率门槛通过、沿用的完整score门槛未通过；快路径与严格score保护版并列报告，不默认以较慢策略替用户决定部署。
