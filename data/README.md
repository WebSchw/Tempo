## Dataset

This folder contains two scripts to download the BRISE data.

### Script 1: `get_data.py`

This python script offers the possibility to either just download the three baseline csv files from github, just preprocess already downloaded data or download the data and directly preprocess it.

- `python get_data.py -d`: just download the csv files
- `python get_data.py -p`: just preprocess the data
- `python get_data.py -dp`: download the data and preprocess it right after

Note: Preprocessing might take a while. If you are executing scripts that require downloaded and preprocessed data, `python get_data.py -dp` is  called automatically.

### Script 2: `download.sh`


This script uses `svn export` to copy the data directory of the brise-plandok `https://github.com/recski/brise-plandok/tree/main/data`. This can also be done in command-line using the command
```bash
svn export https://github.com/recski/brise-plandok/trunk/data
```

The resulting folder contains all csv files. Also the ones that are to be created in the preprocessing step.

Note: It might be necessary for a user to first install svn (like sliksvn, `https://sliksvn.com/download/`) and/or make the script executable (`chmod +x download_svn.sh`)
