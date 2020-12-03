# Classification of heart disease presence using Decision Trees, Random Forests, Gradient Boosting and Adaptive Boosting

### Report
Report directory contains the .pdf-file for the report and also the .tex-file.

### Results
Results directory contains all selected figures and results used in the report.

### Tests
Tests directory contains several examples of commands to run with in the files in the src directory, and also the results you should expect.

### src
src directory contains all source code used to produce the results in the report.

- **data/heart_disease.csv**
    - Contains the data used to produce the results. The .csv-file uses comma ( , ) sepearation and the first line contains the column names, with the last column containing the target data.

- **heart_disease_data_analysis.py**
    - Contains the code for produce the histogram plot, correlation matrix plot, and explained variance ratios plot.

        How to use:
        <pre><code> $ python3 heart_disease_data_analysis.py (plot_type) </code></pre>
        Possible plot type arguments are:  
        - **correlation**
        - **pca**
        - **histogram**

- **project3.py**
    - Contains the code used to produce results for each individual classification method. For example the top 20 results of a grid search in a table, and the ROC-curve and confusion matrix for the best model

        How to use:
        <pre><code> $ python3 heart_disease_data_analysis.py (method) [grid] </code></pre>  
        
        Possbile method arguments are:
        - tree
        - forest
        - gradientboost
        - adaboost

        Method argument is required. If grid is specified, the code will perform a grid search with the specified classification method.

