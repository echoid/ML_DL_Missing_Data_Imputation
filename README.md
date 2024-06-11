# Missing Data Imputation: Do Advanced ML/DL Techniques Outperform Traditional Approaches?

This repository contains the code and data for the paper "Missing Data Imputation: Do Advanced ML/DL Techniques Outperform Traditional Approaches?" by Youran Zhou, Mohamed Reda Bouadjenek, and Sunil Aryal

## Repository Structure

- `datasets/`: Folder containing the datasets used in the study.
- `generation/`: Folder containing scripts for generating data and create missing data.
- `missing_process/`: Folder containing scripts for handling and processing missing data.
- `models/`: Folder containing various models for data imputation.
- `evaluation/`: Folder containing scripts for evaluating imputation performance RMSE, Correlation and downstream tasks.
- `results/`: Folder for storing results.

## Installation Requirements

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

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


Task indicates the downstream task associated with datasets (C-Classification, R-Regression. Classification datasets can also be used for clustering).

| Dataset | Bank | Cali | Climate | Concre | Qsar | Red | Sonar | White | Yachts | Yeast |
|---------|------|------|---------|--------|------|-----|-------|-------|--------|-------|
| #Inst   | 1372 | 20640| 540     | 1030   | 1055 | 1500| 208   | 4898  | 308    | 1484  |
| #Dim    | 5    | 9    | 20      | 8      | 41   | 11  | 60    | 11    | 6      | 8     |
| Task    | C    | R    | C       | R      | C    | R   | C     | R     | R      | C     |

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

   - Navigate to the `models` folder and run the respective model script:

     ```bash
     python [modelname]_main.py --data_name [dataname] --miss_type [misstype]
     ```

   - Parameters:
   --data_name: Specifies the dataset to use. 
   --miss_type: Specifies the type of missing data mechanism to apply. 

   #### Available Options
   #### modelname
   - `gain` [gain](https://github.com/jsyoon0823/GAIN)
   - `hyper`[HyperImputer](https://github.com/vanderschaarlab/hyperimpute)
   - `knn` [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
   - `mcflow`[MCFlow](https://github.com/trevor-richardson/MCFlow)
   - `mean`[Mean](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
   - `mf`[MatrixFactorization](https://pypi.org/project/fancyimpute/)
   - `mice`[MICE](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
   - `missforest`[Missforest](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
   - `MIWAE`[MIWAE](https://github.com/pamattei/miwae)
   - `notMIWAE`[not-MIWAE](https://github.com/nbip/notMIWAE)
   - `ot`[OptimalTransport](https://github.com/BorisMuzellec/MissingDataOT)
   - `tabcsdi`[TabCSDI](https://github.com/pfnet-research/TabCSDI)
   - `XGB`[XGB](https://pypi.org/project/xgbimputer/)

   #### dataname
   - `banknote`
   - `concrete_compression`
   - `wine_quality_white`
   - `wine_quality_red`
   - `california`
   - `climate_model_crashes`
   - `connectionist_bench_sonar`
   - `qsar_biodegradation`
   - `yeast`
   - `yacht_hydrodynamics`

   #### misstype
   - `quantile`
   - `diffuse`
   - `logistic`
   - `mcar`
   - `mar`

   #### Example
   ```bash
   python hyper_main.py --data_name banknote --miss_type quantile
   ```
   
2. **Run Evaluation**:

   ```bash
   jupyter notebook evaluation/rmse.ipynb
   jupyter notebook evaluation/downstream.ipynb
   jupyter notebook evaluation/correlation.ipynb
   jupyter notebook evaluation/clustering.ipynb
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the contributors of the datasets used in this study.
