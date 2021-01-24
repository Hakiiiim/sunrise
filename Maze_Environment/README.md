# SUNRISE: Maze Environment

This codebase was originally forked from [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow).  

## install

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate rainbow` to activate the environment.


## Run experiments

### Rainbow on Atari
```
./scripts/run_rainbow.sh maze
```

### SUNRISE on Atari
```
./scripts/run_sunrise.sh maze [beta] [temperature] [lambda]
```

### To test or change maze settings

You can run the file *my_env.py* to create a random maze and run (also render) a random walk through it.

To change the maze size and/or the number of traps, please walk through *main.py* or *sunrise.py* and change the variables **SIZE** and **NUM_TRAP**.

Please notice that the number of traps chosen should be feasible with respect to the maze size. Indeed, the maze is assured to leave space for the agent to move between the traps (see *my_env.py* for implementation details).

