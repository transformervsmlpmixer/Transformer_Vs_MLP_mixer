# NIPS_Experiments
To launch the experiments, run trainer, to run on slurm environment, the following command will launch the full 
experiment as it appears on the article body

! sbatch --array=1-144 ./meta_trainer.sh

When the run will end all the results, are saved into the ./results directory, as .pickle files. 
And the all_results_table_printer.py file, can be run, in order to get the final graphs as presented in the article.


