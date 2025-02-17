# Polishing Every Facet of the GEM: Testing Linguistic Competence in LLMs and Humans

[![KOGL TYPE 1][kogl-1-shield]][kogl-1]  [![CC BY 4.0][cc-by-shield]][cc-by]


[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[kogl-1]: https://www.kogl.or.kr/info/license.do
[kogl-1-shield]: https://www.kogl.or.kr/static/kogl/img/sub/number1.jpg


This repository is associated with a paper currently under review for ACL 2025. <br/>
<br/><br/>


## Environment Settings
1. Create the virtual environment.
    ```
    conda create -n env_name python=3.9.12
    ```

2. Activate the virtual environment.
    ```
    conda activate env_name
    ```

3. Install the packages from <code>requirements.txt</code>
    ```
    pip install -r requirements.txt
    ```

4. Install the torch toolkit</br>
  Before downloading this Torch package, you should check the compatibility of your CUDA settings.<br/>
  ( I used CUDA 12.2 and cuDNN 8.9.6 with NVIDIA driver 535.183.01 )
    ```
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    ```


<br/><br/>



## KoGEM (<ins>Ko</ins>rean <ins>G</ins>rammar <ins>E</ins>valuation Bench<ins>M</ins>ark)

~~You can download the dataset from [here]().~~
<br/>
Notice) We plan to upload our benchmark and generation results upon acceptance.
<br/><br/>

This presents the components and statistics of our proposed dataset, KoGEM. Our benchmark consists of a total of 1,524 annotated QA pairs. More detailed information about KoGEM can be found in our paper.

</br>
<img src='analysis/assets/figures/texanomy_distributions.png' width='70%'>


</br></br>
## Zero-shot Evaluation Results for Each Subcategory
A closer examination of individual subcategories. These results reveal distinct strengths and weaknesses, as LLMs and humans excel in different areas, underscoring the need for a fine-grained evaluation of linguistic competence at the subcategory level.

</br>
<img src='analysis/assets/figures/subcategory_results.png' width='80%'>


</br></br>
## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

This work is used according to [Korea Open Government License (KOGL) Type 1](https://www.kogl.or.kr/info/license.do).

