# Amelia Tang 

# This driver script complete the final report on the used car price prediction analysis 

# Example usage:
# make all 

all : doc/car_price_prediction_report.Rmd

# Model selection and scores 
results/model_selection.csv : src/model_selection.py 
	python src/model_selection.py --csv_path=results/model_selection.csv
	
# Train, tune and save the final model 
results/final_model.pickle : src/final_model.py
	python src/final_model.py --model_path=results/final_model.pickle
	
# Create the shap plot 
results/shap.svg : src/shap_plot.py
	python src/shap_plot.py --train_x=data/raw/X_train.csv --train_y=data/raw/y_train.csv --shap_path=results/shap.svg
	
# Generate the test scores 
results/test_scores.csv : src/test_result.py 
	python src/test_result.py --test_x=data/raw/X_test.csv --test_y=data/raw/y_test.csv --model_path=results/final_model.pickle --csv_path=results/test_scores.csv

# write the report
doc/car_price_prediction_report.Rmd : results/target_distribution_plot.svg results/price_by_brand.svg results/price_year_brand.svg results/model_selection.csv results/final_model.pickle results/shap.svg results/test_scores.csv      
	Rscript -e "rmarkdown::render('doc/car_price_prediction_report.Rmd', output_format = 'html_document')"

# Clean the created files
clean: clean_selection clean_model clean_shap clean_test
#clean_eda : 
#	rm -rf results/target_distribution_plot.svg
#	rm -rf results/price_by_brand.svg 
#	rm -rf results/price_year_brand.svg

clean_selection : 
	rm -rf results/model_selection.csv 
	
clean_model :
	rm -rf results/final_model.pickle
	
clean_shap :
	rm -rf results/shap.svg

clean_test :
	rm -rf results/test_scores.csv