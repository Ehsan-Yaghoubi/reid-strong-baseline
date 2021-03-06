# Person Re-identification: Implicitly Defining the Receptive Fields of Deep Learning Classification Frameworks:
CNNs are able to autonomously find the critical regions of the input data and discriminate between foreground and background features. However, they demand huge volumes of data, which can be hard to collect and (particularly) annotate for usage in supervised tasks. In this paper, we develop an augmentation technique that empowers the network to improve this autonomous \emph{discovery skill}.

* We built our method on a re-id baseline available at https://github.com/michuanhaohao/reid-strong-baseline. 

## The proposed attention mechanism:
<div align=center>
<img src='imgs/main.png' width='900'>
</div>

## Samples:
Examples of the generated images are as follows. The first two rows show the fake samples with attention on the upper-body, the 3th and 4th rows show the fakes samples with attention on the full-body.
When enabling the attention on the upper-body region, fake samples are different in the human lower-body and the environment, while they resemble each other in the person's upper-body and identity label. By selecting the full-body as the Region of Interest, the generated images will be composed of similar body silhouettes with different surroundings.
<div align=center>
<img src='imgs/samples1.png' width='800'>
</div>
<div align=center>
<img src='imgs/samples2.png' width='580'>
</div>

## Comparison results on the Richly Annotated Pedestrian (RAP) dataset is as follows: Note that, we stopped training the modles when all of them reached the same loss value.
<div align=center>
<img src='imgs/comparison_chart.png' width='600'>
</div>
<div align=center>
<img src='imgs/tabel.png' width='600'>
</div>



