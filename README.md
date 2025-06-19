Create compatable conda environment using:
`conda/mamba create -f environment.yml`

With the python scripts use:
```
#!/bin/bash

source /local/anaconda3/bin/activate
conda activate /YOUR_PATH/envs/RF_distance

# fewer comparisons to make (and trees with fewer polytomys) with a lower keep threshold
python filter_trees.py pentheri_subset2_bb1000_allnni_locustrees.treefile --keep 0.25 --supp 0.8

# should probably add a rooting step HERE
# reroot like sortadate... with prefered order of outgroups AFTER filtering by 
# overall branch support (so the outgroups dont collapse the entire root)
# ...

# once rerooted, do your next steps with the prefered distance metric
python compute_distance_matrices.py filtered_trees.treefile --rf --threads 4
python summarize_distances.py RFmatrix_trees.txt --hist --comps 10 --pcas 2
```

To generate a 3D pca use your prefered eigenvector file (rf, normalized rf, or kf tree distance measure) and which PC's to use. 

see *.py --help 


To do
- confirm collapsed nodes by support
- confirm collapsed branches by length
- create svg or pdf outputs
- create a filter for clusters if observed
- root trees?


Example 3D plot showing three clusters of trees with similar RF distances, suggesting similar topologies:
![plotly 3D plot of a PCA's 1, 2, & 3rd axes.](https://github.com/crcardenas/tree_distance/blob/main/Robinson-Foulds_Distance_PCA.jpg)
