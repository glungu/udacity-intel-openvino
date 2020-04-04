 *Template for Your ReadMe File*
 
 A ReadMe file helps reviewers in accessing and navigating through your code. This file will serve as a template for your own ReadMe file. Do fill out the sections carefully, in a clear and concise manner
 
 # Smart Queue Monitoring System
*TODO:* Write a short introduction to your project.

## Proposal Submission
*TODO* Specify the type of hardware you have chosen for each sector 
Mention the type of hardware you have chosen for each of the scenarios:
- Manufacturing:
- Retail:
- Transportation: 

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Documentation
*TODO* In this section, you can document your project code. You can mention the scripts you ran for your project. This could also include scripts for running your project and the scripts you ran on Devcloud too. Make sure to be as informative as possible!

## Results
*TODO* In this section you can write down all the results you got. You can go ahead and fill the table below.

| Type of Hardware | Time required for inference (on average)(ms) | Time for loading the model | Type of Model Precision |
|------------------|----------------------------------------------|----------------------------|-------------------------|
| CPU              |                                              |                            |                         |
| GPU              |                                              |                            |                         |
| FPGA             |                                              |                            |                         |
| VPU              |                                              |                            |                         |

## Conclusions
*TODO* In this section, you can give an explanation for your results. You can also attach the merits and demerits observed for each hardware in this section

## Stand Out Suggestions
*TODO* If you have implemented the standout suggestions, this is the section where you can document what you have done.
- Fallback Policy: Use OpenVINO's Hetero plugin to use a fallback device like CPU when the main prediction device fails. Write about the changes you had to make in the code to implement these changes. More information regarding Fallback Policy can be found [here](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_HETERO.html)
- MultiDevice Plugin: Use the multi device plugin available as a part of the OpenVINO toolkit. Write about the changes that you had to make in your code to implement this.
