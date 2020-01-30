# Implicit Attention Mechanism Data Augmentation:
CNNs are able to autonomously find the critical regions of the input data and discriminate between foreground and background features. However, they demand huge volumes of data, which can be hard to collect and (particularly) annotate for usage in supervised tasks. In this paper, we develop an augmentation technique that empowers the network to improve this autonomous \emph{discovery skill}.

# Paper:
An Implicit Attention Mechanism for Deep Learning Pedestrian Re-identification Frameworks. ICIP2020, (Submitted).
Paper link at arxiv [[https://to_be_inserted_soon]](https://arxiv.org/)
* This project is is a forked version of https://github.com/michuanhaohao/reid-strong-baseline

## The proposed attention mechanism:
<div align=center>
<img src='imgs/attention.png' width='700'>
</div>

## Samples:
Examples of generated images when developing the attention on upper-body (First two rows and full-body (3th and 4th rows) are as follows.
When enabling the attention on the upper-body region, fake samples are different in the human lower-body and the environment, while they resemble each other in the person's upper-body and identity label. By selecting the full-body as the Region of Interest, the generated images will be composed of similar body silhouettes with different surroundings.
<div align=center>
<img src='imgs/samples1.png' width='900'>
</div>
<div align=center>
<img src='imgs/samples2.png' width='700'>
</div>

## Comparison results on the Richly Annotated Pedestrian (RAP) dataset is as follows:
<div align=center>
<img src='imgs/comparison_chart.png' width='800'>
</div>
<div align=center>
<img src='imgs/tabel.png' width='800'>
</div>



