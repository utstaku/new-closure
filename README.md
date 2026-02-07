## Background
- 1D1V Vlasov–Poisson system
- Landau damping
- ML-based heat-flux closure (AE / FNO / Transformer)

## Method
1. Vlasov simulationでデータ生成
2. モーメント量の時間履歴を入力として ML 学習
3. 学習済みクロージャーを流体方程式に埋め込み

## Experiments
- Linear / nonlinear Landau damping
- Comparison with HP closure
