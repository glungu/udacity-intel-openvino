# Smart Queuing System

In this project you will build a people counter app to reduce congestion in queuing systems by guiding people to the least congested queue. You will have to use Intel's OpenVINO API and the person detection model from their open model zoo to build this project.

## What you will do?
You will be given three videos of different scenarios. Each scenario will have people in queues. Your task is to detect and count the number of people in each queue.

You can use the `person-detection-retail-0013` model for identifying the people. Furthermore you will have to use simple image segmentation to identify the queues and count the number of people in it. Finally, you will deploy the model on four different sets of hardware
 
- CPU
- Integrated GPU
- FPGA
- VPU

The purpose of this project is to get you acquainted with real-world scenarios and use the knowledge and skills that you have acquired in the course to choose the correct hardware for each scenario. Furthermore, you will have to deploy your project on the hardware devices that you have chosen. 

You will be given three different proposals (one for each scenario). Each proposal contains information that will help you choose the hardware for that scenario. Each scenario is based on different sectors where edge devices are typically deployed. The different sectors and their respective proposals are available:

Once you make an informed decision of matching each sector with a particular set of hardware, complete the [project proposal](https://github.com/udacity/nd131-c2-choose_the_right_hardware/blob/master/project/starter/proposals/proposal_outline.md) document for each sector detailing the reason for selecting the hardware.


## Useful Links

Do use these links for solving different tasks of the project
- Inference Engine API Docs - [link](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html)
- Model Documentation - [link](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)
- Intel DevCloud for the Edge Access Docs - [link](https://devcloud.intel.com/edge/get_started/guide/#to-get-started-with-jupyter-notebooks-in-the-intel-xae-devcloud)

## Project Structure
This project has one file that you will have to complete: `person_detect.py`. This file contains code that can help you get input from a video file, and also has a skeleton of a working Inference Engine pipeline. You will need to complete the model specific functions to be able to run it.

The project also has proposals present in the `proposals` folder. You will need to complete the `proposals/proposal_outline.md` file based on your conclusions after reading the other proposal document.

The `resources` folder contains the videos for each scenario and the `bin\queue_param` folder contains `.npy` files that you will need to run your jobs.

## Getting Started
Before you actually get started with building your project, it might be helpful to check that you have done the following:

- [ ] Downloaded the starter files
- [ ] Uploaded the starter files to DevCloud directory
- [ ] Make sure that you check out the project rubrics. The rubrics state what the reviewers will be looking for when reviewing your project.

## Main Tasks
The broad level tasks that you have to finish are:

- [ ] Finish the three proposals: You have to complete the proposal doc file foud in `Part 1` of the Classroom for the three given scenarios.
- [ ] Download the model: You will need to download the `person-detection-retail-0013` model available in the open model zoo
- [ ] Complete the `person_detect.py` file: This is the file with the main prediction code. You will have to complete this file.
- [ ] Complete the 3 Jupyter Notebooks (one for each scenario) and run the Deployment on the four hardware sets via Intel DevCloud for the Edge.
