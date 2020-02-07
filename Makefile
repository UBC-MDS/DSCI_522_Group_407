# national collision data pipe
# author: DSCI_522_Group407
# date: 2020-01-30

all: data/raw_data.csv data/cleaned_test_data.csv results/eda.png results/ml_reports.csv doc/final_report.md doc/eda_report.md

# download data
data/raw_data.csv : src/data_read.R
	Rscript src/data_read.R --filepath='https://opendatatc.blob.core.windows.net/opendatatc/NCDB_2017.csv'

# pre-process data (e.g. one-hot-encoding and split into train & test)
data/cleaned_test_data.csv : data/raw_data.csv src/data_clean.py
	python src/data_clean.py --read_path=data/raw_data.csv --write_path=data/

# exploratory data analysis - visualize predictor distributions across classes
results/eda.png : data/cleaned_train_data.csv src/eda.py
	python src/eda.py --read_path=data/cleaned_train_data.csv --write_path=results/

# build models (random forest and logistic regression) and 
# test results on the test data
results/ml_reports.csv : src/ml_pipeline_3.py
	python src/ml_pipeline_3.py

# render report
doc/eda_report.md : doc/eda_report.md
	Rscript -e "rmarkdown::render('doc/eda_report.Rmd')"
doc/final_report.md : doc/final_report.md doc/refs.bib
	Rscript -e "rmarkdown::render('doc/final_report.Rmd')"

# clean all the output
clean :
	rm -f data/*.csv
	rm -f results/*.png
	rm -f results/*.csv
	rm -f doc/*.md