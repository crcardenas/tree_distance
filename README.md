Create compatable conda environment using:
`conda/mamba create -f environment.yml`

With the python scripts use:
```
#!/bin/bash

source /local/anaconda3/bin/activate
conda activate /YOUR_PATH/envs/RF_distance

# fewer comparisons to make (and trees with fewer polytomys) witha lower keep threshold
python filter_trees.py pentheri_subset2_bb1000_allnni_locustrees.treefile --keep 0.25 --tol 1e-6
python compute_distance_matrices.py filtered_trees.treefile --rf --threads 4
python summarize_distances.py RFmatrix_trees.txt --hist --comps 10 --pcas 2
```

To generate a 3D pca use your prefered file (rf, rnorm, or kf tree distance measure) and which PCA's to use. 

see *.py --help 


To do
1 - create svg or pdf outputs
2 - create a filter for clusters if observed
