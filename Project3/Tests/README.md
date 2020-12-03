# Tests

## Test runs

The following commands should be run in the src directory. Below the commands, is the result you should expect.

Note: there may be some small difference between the provided result and what you get due to the randomness of k-fold cross validation and the way the data is split.

Example 1:
<pre><code> $ python3 project3.py tree  </code></pre>

Result 1:  
![](dt_max_depth_vs_accuracy.png)  

Example 2:  
<pre><code> $ python3 project3.py tree grid </code></pre>  

Result 2:  
![](dt_confusion_matrix_max_depth=5_min_samples_leaf=1_min_samples_split=5_splitter=random.png)
![](dt_roc_curve_max_depth=5_min_samples_leaf=1_min_samples_split=5_splitter=random.png)
(See **dt_best_results_grid.txt** for top 20 results from grid search)

Example 3:
<pre><code> $ python3 heart_disease_data_analysis.py pca </code></pre>

Result 3:  
![](pca_variance_ratio.png)


