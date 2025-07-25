TO DO:

make pdf & SVG plots

add prefix to every step!

------------------------------


Perform a PCA on gene tree distances to explore gene tree discordance 

Create compatable conda environment using:
`conda/mamba create -f environment.yml`

1 - filter_trees.py based on the prefered filtering method (filter_trees.py); this will remove gene trees with uncertainty and speed up pairwise comparision. The tree is also rerooted! Here the rerooted tree uses phyx pxrr with a comma seperated list. This comma seperated list is ranked by the prefred outgroup (first in the list) up to the user defined outgroup. Using more than one outgroup is suggested.

There are a number of options here to create polytomies based on branch length, branch support, or not at all. In all cases, a keep value is selected for the trees with the fewest polytomies (-tol flag) or highest overal mean branch support (--sup flag or no flag). The support flag calls phyx pxcolt to collapse nodes with poor branch support.

```
# basic example
python ../filter_trees.py -t subset100.treefile --ranked_og GCA_048127345.1,GCA_022063505.1,SRR2083640,GCA_044734075.1,GCA_044734065.1,CBX0472,CBX0471,CBX0473 --prefix example --keep 0.50
```

2 - run pairwise distance matirices on the rerooted locus trees (compute_distance_matricies.py). This wraps IQTree2's RF distance calculation and is much faster
```
python compute_distance_matrices.py -t example.rr.treefile -p RFmatrix
```

3 - plot PCA's based on the pairwise matrix; standard PCA analysis using scikitlearn
This includes a scree plot, histogram of distances (this should be normal, if a cluster appears in the PCA, some pattern has emerged in tree distances!)

```
python summarize_distances.py RFmatrix.tsv --hist --comps 10 --pcas 2 --supp subset100_statistics.tsv
```

4 - optional 3D pca that may reveal clusters that the 2d plots didnt. Helpful in subset of treefiles if a cluster emerges. 500+ trees may cause the output HTML file to load slowly

Both step 3 and 4 have an option to color the tree in PCA space by mean support (--sup flag) or proportion of polytomy (--tol).
```
python 3d_pca.py RFmatrix_trees_eigenvectors.csv --supp subset100_statistics.tsv
```

5 - Create a subset of trees with a filter based on the PCA axis values found in plotly; this requires the user to examine the 2d or 3d output and selecting a cut off to create two subsets of tree files for examination. This extracts the information from an IQTree2 log so if necessary, new trees can be constructed.
```
python filter_by_pca.py  --eigenvec RFmatrix_trees_eigenvectors.csv --treefile filtered_trees.rr.treefile  --filter 'PC1>=0.033' --filter 'PC1<=0.47' --filter 'PC2<0' --prefix example
```


Example 3D plot showing three clusters of trees with similar RF distances, suggesting similar topologies:
TODO!
