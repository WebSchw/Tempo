## Dataset

This folder contains two scripts to download the BRISE data.

### Script 1: `download.py`

This script just downloads the three baseline csv files from github that do not contain features or labels. These labels and features are then extracted in a preprocessing-step.

### Script 2: `download.sh`


This script uses `svn export` to copy the data directory of the brise-plandok `https://github.com/recski/brise-plandok/tree/main/data`. This can also be done in command-line using the command
```bash
svn export https://github.com/recski/brise-plandok/trunk/data
```

The resulting folder contains all csv files. Also the ones that are to be created in the preprocessing step.

Note: It might be necessary for a user to first install svn (like sliksvn, `https://sliksvn.com/download/`) and/or make the script executable (`chmod +x download_svn.sh`)
