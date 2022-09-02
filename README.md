# Unified Framework for Real-world Skeleton Action Recognition 

[SSTA-PRS: Selective Spatio-Temporal Aggregation Based Pose Refinement System](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_Selective_Spatio-Temporal_Aggregation_Based_Pose_Refinement_System_Towards_Understanding_Human_WACV_2021_paper.pdf) | [Project page](https://github.com/YangDi666/SSTA-PRS)  
[UNIK: A Unified Framework for Real-world Skeleton-based Action Recognition](https://arxiv.org/pdf/2107.08580) | [Project page](https://yangdi666.github.io/UNIK-project/)  
[ViA: View-invariant Skeleton Action Representation Learning via Motion Retargeting](https://arxiv.org/pdf/2209.00065.pdf) | [Project page](https://walker-a11y.github.io/ViA-project/)

## Evironment

-- Python3 with PyTorch version >=Pytorch0.4.

## Data Preparation

 - `mkdir data`
 - Posetics: please contact us ([di.yang@inria.fr](di.yang@inria.fr)) for Data Request.
   ![ad](https://github.com/YangDi666/UNIK/blob/main/demo/demo.png)
 - [Toyota Smarthome](https://github.com/YangDi666/SSTA-PRS#refined-pose-data): download the raw data (skeleton-v2.0 refined by [SSTA-PRS](https://github.com/YangDi666/SSTA-PRS#refined-pose-data)). 
 - Penn Action: download the raw [skeleton data](https://drive.google.com/file/d/13RUvRrNFOlyKSVwNuQAYqg3Vib7Ffbn8/view?usp=sharing).
 - For other datasets: [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D)/[Skeleton-Kinetics](https://github.com/yysijie/st-gcn).
 - Put them under the data directory:

        -data\
         -smarthome_raw\
            -smarthome_skeletons\
             - ... .json
               ... .json
               ...               
         -pennAction_raw\
            -skeletons\
             - ... .json
               ...
         -posetics_raw\
            -posetics_train_label.json
            -posetics_val_label.json  
            -posetics_train\
             - ... .json
               ...        
            -posetics_val\
             - ... .json
               ... 
         -nturgbd_raw\
            -samples_with_missing_skeletons.txt
            -nturgb+d_skeletons\
             - ... .skeleton
             - ...
         -kinetics_raw\             
                ...
         -...
             
 - Preprocess the data with
    ```
    cd data_gen
    python smarthome_gendata.py
    python penn_gendata.py
           ...
    ```
    
 - Generate the bone data with:

    ```python gen_bone_data.py```

## Pre-training on Posetics

    python run_unik.py --config ./config/posetics/train_joint.yaml
    
 
Pre-trained model is now avalable [here](https://drive.google.com/file/d/1K6RVaV02oy0gy8swab8V0s6T7a9YPuxS/view?usp=sharing). Move it to 

    ./weights/

## Training (Fine-tuining) & Testing

Change the config file depending on what you want (e.g., for Smarthome).

    python run_unik.py --config ./config/smarthome-cross-subject/train_joint.yaml
    
    python run_unik.py --config ./config/smarthome-cross-subject/train_bone.yaml

To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer.

    python run_unik.py --config ./config/smarthome-cross-subject/test_joint.yaml
    
    python run_unik.py --config ./config/smarthome-cross-subject/test_bone.yaml

Then combine the generated scores with:

    python ensemble.py --datasets smarthome/xsub

For evaluation on Smarthome:

 - Cross-subject:
 
       python evaluation-cs.py runs/smarthome/smarthome_cs_unik_test_joint_right.txt 31
	
 - Cross-view:
 
       python evaluation-cv.py runs/smarthome/smarthome_cv2_unik_test_joint_right.txt 19
       python evaluation-cv.py runs/smarthome/smarthome_cv1_unik_test_joint_right.txt 19

## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@article{yang2021unik,
      title={UNIK: A Unified Framework for Real-world Skeleton-based Action Recognition}, 
      author={Di Yang and Yaohui Wang and Antitza Dantcheva and Lorenzo Garattoni and Gianpiero Francesca and Francois Bremond},
      year={2021},
      journal={BMVC}
}

@article{yang2022via,
      title={ViA: View-invariant Skeleton Action Representation Learning via Motion Retargeting}, 
      author={Di Yang and Yaohui Wang and Antitza Dantcheva and Lorenzo Garattoni and Gianpiero Francesca and Francois Bremond},
      year={2022},
      journal={arXiv preprint arXiv:2209.00065}
}
```
