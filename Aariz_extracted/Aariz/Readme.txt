The dataset directory is structured as follows:

Aariz
    |- train
    |   |- Cephalograms
    |   |   |- cks2ip8fq2a0j0yufdfssbc09.png
    |   |   |- cks2ip8fq2a0t0yufgab484s9.png
    |   |   |_ ....
    |   |_ Annotations
    |      |- Cephalometric Landmarks
    |      |   |- Junior Orthodontists
    |      |   |   |- cks2ip8fq2a0j0yufdfssbc09.json
    |      |   |   |- cks2ip8fq2a0t0yufgab484s9.json
    |      |   |   |_ ....
    |      |   |_ Senior Orthodontists
    |      |      |- cks2ip8fq2a0j0yufdfssbc09.json
    |      |      |- cks2ip8fq2a0t0yufgab484s9.json
    |      |      |_ ....
    |      |_ CVM Stages
    |         |_ cks2ip8fq2a0j0yufdfssbc09.json
    |         |_ cks2ip8fq2a0t0yufgab484s9.json
    |         |_ ....
    |- valid
    |- test
    |- cephalogram_machine_mappings.csv
    |  In this .csv file, you can find the resolutions of the X-ray imaging devices from which the given cephalogram is extracted. 
    |  The pixel size will used to transform the mean radial error (MRE) into milimeters to assess whether the predicions are in clinically accepted range or not.
    |_ Readme.txt

Please visit the github repository (https://github.com/manwaarkhd/aariz), where you can find the inital code to read the cephalograms with their corresponding annotations from directories.
In case of any questions/queries, feel free to contact us at cepha29.challenge@gmail.com 
