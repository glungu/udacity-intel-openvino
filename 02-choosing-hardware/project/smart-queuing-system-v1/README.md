 # Smart Queue Monitoring System
This project deals with detecting people in queues (in order to redirect them to shortest queue) 
using inference on pre-trained neural network with Intel OpenVINO framework. 
The idea is to choose hardware most suited for particular task (scenario). 

## Proposal Submission
The type of hardware chosen for each of the scenarios:
- Manufacturing: CPU + FPGA
- Retail: CPU + Integrated GPU
- Transportation: CPU + VPU

## Project Set Up and Installation
Project setup procedure included the following steps:
* Connect to Intel DevCloud, setup developer account, and create or upload jupyter notebook in the DevCloud 
  to have access to the DevCloud environment
* Downloading the pre-trained model from OpenVINO model zoo: person-detection-retail-0013. 
  IR (Intermediate Representation) consists of two files: 
  [xml](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.xml),
  [bin](https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.bin). 
  Downloading can be performed using OpenVINO's Model Downloader, or by using `wget` and direct links:
  
  `!wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.xml`
  
  `!wget https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/person-detection-retail-0013/FP16/person-detection-retail-0013.bin`
  
  The downloaded model was placed in the `model` sub-directory of the home dir in the virtual environment 
  running jupyter notebook. This ensures that it will be made available to the virtual host running the actual job.  
   
* Check that command line tools like `qsub`, `qstat`, `qdel` etc. are available 
  to submit, delete and query status of DevCloud Jobs
* Install dependencies required by the project using `pip` from the jupyter notebook, for example:
  `!pip3 install ipywidgets matplotlib`  

## Documentation
The project code is mainly located in the following files:
* `person_detect.py` - Python script containing the main inference code: 
  initialising OpenVINO core, loading the network to make it ready for inference, 
  loading and processing the input MP4 file, and outputting the resulting MP4 file with bounding boxes.
  Two sets of bounding boxes are added to the original file: green - for the location of the queuing areas, 
  red - for the location of the people found in frame.
  Additionally, the script outputs `stats.txt` file with information on total people and the number of 
  people in each queue area per frame. 
  Measures of loading and total inference time are also written at the end of this file.

The three jupyter notebooks for each of the scenarios:
* `people_deployment-manufacturing.ipynb` - Manufacturing (worker queues at conveyor belt on the factory floor)
* `people_deployment-retail.ipynb` - Retail (customer queues at cashier counters at the grocery store) 
* `people_deployment-transportation.ipynb` - Transportation (passenger queues at the busy metro station)

Each notebook make use of the `person_detect.py` script to make inference. It follows the same pattern:
* Creates job script
* Submits 4 jobs using this script to the DevCloud, using same video for particular scenario, but different hardware: 
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU)
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU + Integrated Intel® HD Graphics 530 card GPU)
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE, with Intel Neural Compute Stick 2 (Myriad X)
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE, with IEI Mustang-F100-A10 card (Arria 10 FPGA).
* Shows results of each job:
  * Video with bounding boxes
  * Model loading time 
  * Average Inference time per frame
  * Inference FPS (frames per second)
     
## Results

#### Manufacturing

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  40.192                                      |  01.679189                 |  FP16                   |
| GPU              |  42.398                                      |  35.238801                 |  FP16                   |
| VPU              |  156.621                                     |  02.530733                 |  FP16                   |
| FPGA             |  32.942                                      |  29.014384                 |  FP16                   |


#### Retail
 
| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  27.043                                      |  01.587233                 |  FP16                   |
| GPU              |  32.928                                      |  36.118535                 |  FP16                   |
| VPU              |  146.740                                     |  02.566522                 |  FP16                   |
| FPGA             |  19.558                                      |  29.120876                 |  FP16                   |


#### Transportation

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |  50.998                                      |  01.521313                 |  FP16                   |
| GPU              |  52.103                                      |  35.303533                 |  FP16                   |
| VPU              |  149.853                                     |  02.584625                 |  FP16                   |
| FPGA             |  41.968                                      |  29.428855                 |  FP16                   |


## Conclusions
The fastest inference is on FPGA, although takes longer to load the model.
CPU and GPU are almost the same on inference, with GPU a little longer, and much longer to load the model.
Finally, VPU is the slowest on inference. 

