.PHONY: data features train train_tabnet dashboard all

data:        ## Scrape Wikipedia + rebuild raw CSV
	python3 run_pipeline.py --scrape

features:    ## Re-engineer features + mirror dataset
	python3 run_pipeline.py --features

train:       ## Train all models, save best to models/best_model.pkl
	python3 src/train_lgbm.py && python3 src/train_catboost.py && python3 src/train_xgb.py && python3 src/train_ensemble.py

train_tabnet: ## Train TabNet and re-run ensemble selection
	python3 src/train_tabnet.py && python3 src/train_ensemble.py

dashboard:   ## Launch Streamlit app
	streamlit run app.py

all:         ## Full pipeline end-to-end
	python3 run_pipeline.py --all

simulate:    ## Run German Open 2026 Monte Carlo simulation
	python3 src/simulate_german_open.py

cv:          ## Run rolling 3-fold temporal cross-validation
	python3 src/temporal_cv.py

tune:        ## Optuna hyperparameter search (50 trials) + retrain best models
	python3 src/tune_hyperparams.py --model all --trials 50 --retrain

help:        ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
