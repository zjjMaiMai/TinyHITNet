## HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching

This is a **Pytorch** implementations of *"HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"*.

-----------------

### Accuracy
  
| Model                         |  **Sceneflow Finalpass**, EPE | δ<0.1(%) | δ<1.0(%) | δ<3.0(%) | GMac(G) | Checkpoint |     |
| ---                           |  ---                          | ---   | ---   | ---   | ---  | --- | --- |
| **HitNet-XL**                 | 0.3762 | 82.1971 | 96.2759 | 98.1472 | 386.6757 | [ckpt](ckpt/hitnet_xl_sf_finalpass_from_tf.ckpt) | converted copy of [original](https://github.com/google-research/google-research/tree/master/hitnet) tensorflow model |
| **HitNet**                    | 0.5486 | 75.1038 | 94.5830 | 97.3138 | 50.5048  | [ckpt](ckpt/hitnet_sf_finalpass.ckpt) |  |
| **StereoNet**                 | 0.7566 | 50.0250 | 91.0111 | 96.2597 | 106.7765 | [ckpt](ckpt/stereo_net.ckpt) | 8x downsample |


### Training

1) Compile and install cuda op
    ```shell
    pip install ./ext_op
    ```

2) Replace dataset path in **preprocess/plane_fitting.py** and **script/hitnet_sf_finalpass.sh**

3) Robust plane fitting 
    ```
    python preprocess/plane_fitting_sf.py
    ```

2) Training
    ```shell 
    bash script/hitnet_sf_finalpass.sh
    ```

### Evaluation

1) Replace dataset path in **eval.py**

1) Evaluation
    ```shell
    python eval.py --model HITNet --ckpt ckpt/{ckpt_name} --data_type SceneFlow --data_root_val {path} --data_list_val lists/sceneflow_test.list
    ```

### Predict

```shell
python predict.py --model HITNet --ckpt ckpt/{ckpt_name} --images {left.png} {right.png} --output {disp.png}
```

## Citation
```
@article{tankovich2020hitnet,
  title={HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching},
  author={Tankovich, Vladimir and H{\"a}ne, Christian and Fanello, Sean and Zhang, Yinda and Izadi, Shahram and Bouaziz, Sofien},
  journal={arXiv preprint arXiv:2007.12140},
  year={2020}
}
```

However, if you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{hang2021tinyhitnet,
  title={TinyHITNet},
  author={zjjMaiMai},
  howpublished={\url{https://github.com/zjjMaiMai/TinyHITNet}},
  year={2021}
}
```