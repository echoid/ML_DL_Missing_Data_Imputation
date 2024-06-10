# Missing Data Imputation: Do Advanced ML/DL Techniques Outperform Traditional Approaches?

This repository contains the code and data for the paper "Missing Data Imputation: Do Advanced ML/DL Techniques Outperform Traditional Approaches?".

## Repository Structure

- `create_data.ipynb`: Script to load raw data, perform normalization, and split the data.
- `create_missing.ipynb`: Script to create missing data masks with different missing rates.
- `create_visualization.ipynb`: Script to create visualizations including missing mechanism scatter plots, missing rate plots, and missing distribution plots.
- `models/`: Folder containing various models for data imputation.
- `evaluation/`: Folder containing scripts for evaluating imputation performance and downstream tasks.
- `results/`: Folder for storing results and visualizations.

## Datasets

The datasets used in this study are listed below:

1. [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
2. [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
3. [Wine Quality (White)](https://archive.ics.uci.edu/dataset/186/wine+quality)
4. [Wine Quality (Red)](https://archive.ics.uci.edu/dataset/186/wine+quality)
5. [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
6. [Climate Model Simulation Crashes](https://archive.ics.uci.edu/dataset/252/climate+model+simulation+crashes)
7. [Connectionist Bench (Sonar, Mines vs. Rocks)](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
8. [QSAR Biodegradation](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation)
9. [Yeast](https://archive.ics.uci.edu/dataset/110/yeast)
10. [Yacht Hydrodynamics](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics)

## How to Use

### Data Preparation

1. **Load and Prepare Data**:

   ```bash
   jupyter notebook create_data.ipynb
   ```

2. **Create Missing Data**:

   ```bash
   jupyter notebook create_missing.ipynb
   ```

### Model Training and Evaluation

1. **Train and Test Models**:

   - Navigate to the `models` folder and run the respective model notebooks:

     ```bash
     jupyter notebook models/mean.ipynb
     jupyter notebook models/knn.ipynb
     jupyter notebook models/missforest.ipynb
     jupyter notebook models/XGB.ipynb
     jupyter notebook models/MICE.ipynb
     ```

2. **Run Evaluation**:

   ```bash
   jupyter notebook evaluation/rmse.ipynb
   jupyter notebook evaluation/mltask.ipynb
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the contributors of the datasets used in this study.
