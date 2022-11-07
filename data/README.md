## Dataset

This folder contains a script to download the BRISE data. It uses `svn export` to copy the data directory of the brise-plandok `https://github.com/recski/brise-plandok/tree/main/data`. This can also be done in command-line using the command
```bash
svn export https://github.com/recski/brise-plandok/trunk/data
```
Note: It might be necessary for a user to first install svn (like sliksvn, `https://sliksvn.com/download/`) and/or make the script executable (`chmod +x download_svn.sh`)
