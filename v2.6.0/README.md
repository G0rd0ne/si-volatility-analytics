# Si Volatility Analytics v2.6.0

Auto-exported from Google Colab  
**Timestamp:** 2026-04-16 10:12:42  
**Total files:** 37

## Структура
v2.6.0/
├── core/               # Core модули
├── har_backtest/       # HAR Backtest
├── optimization/       # Optimization
├── thursday_analysis/  # Thursday Analysis
└── schemas/            # Pydantic Schemas

### core/ (10 files)
- `01_imports_config_utils.py` (Cell: EHJdFwWrZBWa)
- `02_moex_client_contracts.py` (Cell: eVcfSVEFZJlh)
- `03_volatility_estimators.py` (Cell: lw7Cwpd3ZRFy)
- `04_analytics_classification.py` (Cell: IkEQx5x8ZYpV)
- `05_1_vol_metrics.py` (Cell: 3_ubpWu6ZeZX)
- `05_2_self_tests.py` (Cell: zvKoyi6IMICM)
- `05_3_dashboard_viz.py` (Cell: zUZl2oMsMNe6)
- `05_4_rv_forecast_comp.py` (Cell: ioODNCI2MSV8)
- `05_5_main_pipeline.py` (Cell: N4wV10wPMXCE)
- `06_data_summary_report.py` (Cell: GqDtzE_6k8cR)

### har_backtest/ (14 files)
- `07_1_backtest_config.py` (Cell: CMshlQIJOpAR)
- `07_2_1_contract_id.py` (Cell: 8vPWTu_7YV9k)
- `07_2_2_hist_data_loader.py` (Cell: Wk3l9M3XYc_M)
- `07_2_3_rv_forward_comp.py` (Cell: YeeeE3JfYkpw)
- `07_2_4_har_forecasting.py` (Cell: t_U-l4bLYuPj)
- `07_2_5_ewma_forecasting.py` (Cell: vjYxQ94tY0Da)
- `07_2_6_forecast_evaluation.py` (Cell: t3iEvDZ5Y5ti)
- `07_2_7_beta_optimization.py` (Cell: 7b4i090EY_w0)
- `07_2_8_backtest_engine_load.py` (Cell: WAGiEd9iZgjY)
- `07_2_9a_data_pipeline.py` (Cell: xZxqEXuxCI6U)
- `07_2_9b_backtest_execution.py` (Cell: EAkYYJ81CK0i)
- `07_3_backtest_viz.py` (Cell: os-mMOfBO0eN)
- `07_4_backtest_entrypoint.py` (Cell: wtwa8dJkWHzX)
- `07_5_backtest_self_tests.py` (Cell: EhqRImXrGVrL)

### optimization/ (7 files)
- `07_6_1_opt_config.py` (Cell: Eqw_Dv4zPhvW)
- `07_6_2_single_horizon_opt.py` (Cell: WD6L1TtYPkLp)
- `07_6_3_portfolio_runner.py` (Cell: UaJZgbXFPp-C)
- `07_6_4_opt_validation.py` (Cell: j5fyjgzjPuvf)
- `07_6_5_tuning_config.py` (Cell: ZocF4QjFbBYu)
- `07_6_6_tuning_engine.py` (Cell: qlxxGXLHfvbw)
- `07_6_8_unified_tuning_runner.py` (Cell: H9jPKDsaf0ho)

### schemas/ (1 files)
- `11_pydantic_schemas.py` (Cell: 45178ceb)

### thursday_analysis/ (5 files)
- `10_1_thursday_helpers.py` (Cell: u4HgpZTSWjRc)
- `10_2_5d_thursday_opt.py` (Cell: f96pbxZwWxiP)
- `10_3_runner_comparison.py` (Cell: X4TsUjodW-Tr)
- `10_3b_cache_comparison.py` (Cell: W94CamopX0yn)
- `10_4_execute_thursday_opt.py` (Cell: 4zfJSpR6ZlRZ)
