## ___***Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control***___
<div align="center">
<img src='img/logo.png' style="height:75px"></img>

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='http://fuxiao0719.github.io/projects/robomaster'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://drive.google.com/file/d/1GSfB3UbrUtJvgHkNrzm4TwLjqdR6H0-n/view'><img src='https://img.shields.io/badge/arXiv-2506.XXXX-b31b1b.svg'></a> &nbsp;
 <a href='https://huggingface.co/KwaiVGI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;

**[Xiao Fu<sup>1 &dagger;</sup>](https://fuxiao0719.github.io/), 
[Xintao Wang<sup>2 &#9993;</sup>](https://xinntao.github.io/), 
[Xian Liu<sup>1</sup>](https://alvinliu0.github.io/), 
[Jianhong Bai<sup>3</sup>](https://jianhongbai.github.io/), 
[Runsen Xu<sup>1</sup>](https://runsenxu.com/), <br>
[Pengfei Wan<sup>2</sup>](https://scholar.google.com/citations?user=P6MraaYAAAAJ&hl=en),
[Di Zhang<sup>2</sup>](https://openreview.net/profile?id=~Di_ZHANG3),
[Dahua Lin<sup>1&#9993;</sup>](http://dahua.site/)** 
<br>
<sup>1</sup>The Chinese University of Hong Kong
<sup>2</sup>Kuaishou Technology
<sup>3</sup>Zhejiang University
<br>
&dagger;: Intern at KwaiVGI, Kuaishou Technology, &#9993;: Corresponding Authors

</div>

## 🔥 Reproduce Website Demos

1. **[Environment Set Up]** Our environment setup is identical to [CogVideoX](https://github.com/THUDM/CogVideo). You can refer to their configuration to complete the environment setup.
    ```bash
    conda create -n robomaster python=3.10
    conda activate robomaster
    ```
2. Robotic Manipulation on Diverse Out-of-Domain Objects.
    ```bash
    python inference_inthewild.py \
        --input_path demos/diverse_ood_objs \
        --output_path samples/infer_diverse_ood_objs \
        --transformer_path ckpts/RoboMaster \
        --model_path ckpts/CogVideoX-Fun-V1.5-5b-InP
    ```
3. Robotic Manipulation with Diverse Skills
    ```bash
    python inference_inthewild.py \
        --input_path demos/diverse_skills \
        --output_path samples/infer_diverse_skills \
        --transformer_path ckpts/RoboMaster \
        --model_path ckpts/CogVideoX-Fun-V1.5-5b-InP
    ```
4. Long Video Generation in Auto-Regressive Manner
    ```bash
    python inference_inthewild.py \
        --input_path demos/long_video \
        --output_path samples/long_video \
        --transformer_path ckpts/RoboMaster \
        --model_path ckpts/CogVideoX-Fun-V1.5-5b-InP
    ```
    
## 🚀 Benchmark Evaluation (Reproduce Paper Results)
  ```
├── RoboMaster
    ├── eval_metrics
        ├── VBench
        ├── common_metrics_on_video_quality
        ├── eval_traj
        ├── results
            ├── bridge_eval_gt
            ├── bridge_eval_ours
            ├── bridge_eval_ours_tracking
  ```
**(1) Inference on Benchmark & Prepare Evaluation Files**
1. Generating `bridge_eval_ours`. (Note that the results may vary slightly across different computing machines, even with the same seed. We have prepared the reference files under `eval_metrics/results`)
    ```bash
    cd RoboMaster/
    python inference_eval.py
    ```
1. Generating `bridge_eval_ours_tracking`: Install [CoTracker3](https://github.com/facebookresearch/co-tracker), and then estimate tracking points with grid size 30 on `bridge_eval_ours`. 
**(2) Evaluation on Visual Quality**

1. Evaluation of VBench metrics.
    ```bash
    cd eval_metrics/VBench
    python evaluate.py \
        --dimension aesthetic_quality imaging_quality temporal_flickering motion_smoothness subject_consistency background_consistency \
        --videos_path ../results/bridge_eval_ours \
        --mode=custom_input \
        --output_path evaluation_results
    ```
2. Evaluation of FVD and FID metrics.
    ```bash
    cd eval_metrics/common_metrics_on_video_quality
    python calculate.py -v1_f ../results/bridge_eval_ours -v2_f ../results/bridge_eval_gt
    python -m pytorch_fid eval_1 eval_2
    ```

**(3) Evaluation on Trajectory (Robotic Arm & Manipulated Object)**

1. Estimation of TrajError metrics. (Note that we exclude some samples listed in `failed_track.txt`, due to failed estimation by [CoTracker3](https://github.com/facebookresearch/co-tracker))
    ```bash
    cd eval_metrics/eval_traj
    python calculate_traj.py \
        --input_path_1 ../results/bridge_eval_ours \
        --input_path_2 ../results/bridge_eval_gt \
        --tracking_path ../results/bridge_eval_ours_tracking \
        --output_path evaluation_results
    ```
2. Check the visualization videos under `evaluation_results`. We blend the trajectories of robotic arm and object throughout the entire video for better illustration.

####

## 🔗 Citation
If you find this work helpful, please consider citing:
```BibTeXw
@article{fu2025robomaster,
  title={Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control},
  author={Fu, Xiao and Wang, Xintao and Liu, Xian and Bai, Jianhong and Xu, Runsen and Wan, Pengfei and Zhang, Di and Lin, Dahua},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2025}
}
```
