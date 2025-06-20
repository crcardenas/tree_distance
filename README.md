TO DO:

ensure PCA plots are being made correctly

make pdf & SVG plots

add prefix to every step!

Have a think:

should we instead consider the number of taxa in a locus instead of collapsing nodes of uncertainty?


------------------------------


Perform a PCA on gene tree distances to explore gene tree discordance 

Create compatable conda environment using:
`conda/mamba create -f environment.yml`

1 - filter_trees.py based on the prefered filtering method (filter_trees.py); this will remove gene trees with uncertainty and speed up pairwise comparision. The tree is also rerooted! Here the rerooted tree uses phyx pxrr with a comma seperated list. More than one outgroup is suggested.

There are a number of options here to create polytomies based on branch length, branch support, or not at all. In all cases, a keep value is selected for the trees with the fewest polytomies (-tol flag) or highest overal mean branch support (--sup flag or no flag). The support flag calls phyx pxcolt to collapse nodes with poor branch support.

```
# basic example
python ../filter_trees.py -t subset100.treefile --ranked_og GCA_048127345.1,GCA_022063505.1,SRR2083640,GCA_044734075.1,GCA_044734065.1,CBX0472,CBX0471,CBX0473 --prefix example --keep 0.50
```

2 - run pairwise distance matirices on the rerooted locus trees (compute_distance_matricies.py). The method here needs to be selected based on robinsons fould distance, normalized robinsons foulds, and Kuhner–Felsenstein branch-score distance This is calculated using... 
```
python compute_distance_matrices.py filtered_trees.rr.treefile --rf --threads 4
```

3 - plot PCA's based on the pairwise matrix; standard PCA analysis using scikitlearn
This includes a scree plot, histogram of distances (this should be normal, if a cluster appears in the PCA, some pattern has emerged in tree distances!)

```
python summarize_distances.py RFmatrix_trees.txt --hist --comps 10 --pcas 2
```

4 - optional 3D pca that may reveal clusters that the 2d plots didnt. Helpful in subset of treefiles if a cluster emerges. 500+ trees may cause the output HTML file to load slowly
```
python 3d_pca.py RFmatrix_trees_eigenvectors.csv
```

5 - Create a subset of trees with a filter based on the PCA axis values found in plotly; this requires the user to examine the 2d or 3d output and selecting a cut off to create two subsets of tree files for examination
```
python filter_by_pca.py  --eigenvec RFmatrix_trees_eigenvectors.csv --treefile filtered_t
rees.rr.treefile  --filter 'PC1>=0.033' --filter 'PC1<=0.47' --filter 'PC2<0' --prefix example
```


Example 3D plot showing three clusters of trees with similar RF distances, suggesting similar topologies:
![plotly 3D plot of a PCA's 1, 2, & 3rd axes.](https://github.com/crcardenas/tree_distance/blob/main/Robinson-Foulds_Distance_PCA.jpg)
