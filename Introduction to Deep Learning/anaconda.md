# Command Lists of Anacondas

### Install Multiple packages
```
conda install numpy pandas
```

### Search Packages
```
conda search *numpy*
```

### Create Env with packages
```
conda create -n my_env python=3.6 numpy
```

### Save and Load

- Export Package List
```
conda env export > environment.yaml
```

- Create env and load pacakges
```
conda env create -f environment.yaml
```

### Remove Env
```
conda env remove -n my_env
```

# Cheet Sheet
[CheetSheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)