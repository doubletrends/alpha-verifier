.PHONY: install data features pipeline matrix dxy_matrix dxy_ret_5_matrix dxy_ret_10_matrix dxy_ret_30_matrix dxy_ret_50_matrix dxy_ret_100_matrix volatility_matrix halving_matrix halving_cycle_transition rsi_matrix williams_r williams_r_plain_7 williams_r_plain_14 williams_r_pair_state williams_r_deep_os williams_r_ob_cont mean_reversion thresholds clean

# ── setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── data pipeline ─────────────────────────────────────────────────────────────
data:
	python pipeline/step_1_data_ingestion.py

features:
	python pipeline/step_2_feature_engineering.py

pipeline: data features

# ── research ──────────────────────────────────────────────────────────────────
matrix:
	python research/residual/matrix.py

dxy_matrix:
	python research/dxy/matrix.py

dxy_ret_5_matrix:
	python research/dxy/ret_5_matrix.py

dxy_ret_10_matrix:
	python research/dxy/ret_10_matrix.py

dxy_ret_30_matrix:
	python research/dxy/ret_30_matrix.py

dxy_ret_50_matrix:
	python research/dxy/ret_50_matrix.py

dxy_ret_100_matrix:
	python research/dxy/ret_100_matrix.py

volatility_matrix:
	python research/volatility/matrix.py

halving_matrix:
	python research/halving/matrix.py

halving_cycle_transition:
	python research/halving/cycle_transition.py

rsi_matrix:
	python research/rsi/matrix.py

williams_r:
	python research/williams_r/composite.py

williams_r_plain_7:
	python research/williams_r/plain_7.py

williams_r_plain_14:
	python research/williams_r/plain_14.py

williams_r_pair_state:
	python research/williams_r/pair_state.py

williams_r_deep_os:
	python research/williams_r/deep_oversold_stabilization.py

williams_r_ob_cont:
	python research/williams_r/overbought_continuation.py

mean_reversion:
	python research/residual/mean_reversion.py

thresholds:
	python research/residual/thresholds.py

# ── housekeeping ──────────────────────────────────────────────────────────────
clean:
	rm -rf output
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
