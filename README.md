# Running camera calibration
General: Scripts were tested using Python 3.12 and virtual environment.  
## Setting up virtual environment in Windows and Power Shell:  
1. Go to the top directory of the package (the one where the requirements.txt file resides).  
2. In Windows Power Shell verify the Python version:
```bash
python --version
```
3. Create virtual environment:
```bash
python -m venv venv
```
4. Activate virtual environment:
```bash
venv\Scripts\Activate.ps1
```
5. Install required packages:
```bash
pip install -r requirements.txt
```
(You might want to upgrade pip first). This step you can execute only once, when you are setting up virtual environment for the first time.

6. Don't forget to deactivate virtial environment after you finished working with script:
```bash
deactivate
```

## Setting up virtual environment in Linux:
1. Go to the top directory of the package (the one where the requirements.txt file resides).
2. In Linux terminal verify the Python version:
```bash
python3 --version
```

3. Create virtual environment:
```bash
python3 -m venv venv
```

4. Activate virtual environment:
```bash
source venv/bin/activate
```

5. Install required packages:
```bash
pip install -r requirements.txt
```
(You might want to upgrade pip first). This step you can execute only once, when you are setting up virtual environment for the first time.

    
6. Don't forget to deactivate virtial environment after you finished working with script:
```bash
deactivate
```
## Running script:
To calibrate camera use calibrate_camera.py script.
Run the script with -h argument to get the list of all command line arguments.

To calibrate camera just copy your chessboard calibratoin images to one directory where you have read/write access.
Make sure you are in the main repo directory. Run the calibration script (Linux command):
```bash
python3 src/calibrate_camera.py -d path/to/image/dirctory
```

Windows command:
```bash
python src/calibrate_camera.py -d path/to/image/dirctory
```

The calibration script produces 4 results:
1. `corners` directory containing images with detected chessboard corners. If you see something wrong with corner detection, remove respective image from the input image set.
2. `calibration.yaml` file - the OpenCV data storage for computed calibration parameters. Use this file to conveniently load calibration parameters in your OpenCV code.
3. `calibration_report.txt` - Contains some more statistics, like reprojection errors and standard deviations of estimated parameters.
4. `orbslam3_mono_camera_config.yaml` file contains calibration results formatted as ORB SLAM 3 config file to conveniently use with ORB SLAM 3.

