#import "@preview/polylux:0.3.1": *
#import "@preview/pinit:0.1.3": *

#set text(size: 25pt, font: "EB Garamond", fallback: false)
#show heading: it => {
  text(fill: blue, )[
    #it.body
  ]
}
#set page(
  paper: "presentation-16-9",
  margin: 16pt,
  footer: [
    #set text(size: 12pt, fill: blue, weight: "bold")
    #grid(columns: (2fr, 3fr, 2fr),
      [#align(left)[Dept. CSE, UT Arlington]],
      [#align(center)[Scalable Modeling & Imaging & Learning Lab (SMILE)]],
      [#align(right)[
        #counter(page).display(
          "1/1",
          both: true,
        )]
      ]
    )
  ]
)

#polylux-slide[
  #align(horizon + center)[
    #heading("Segment Any Cell: A SAM-based Auto-prompting Fine-tuning Framework for Nuclei Segmentation", outlined: false)

    Saiyang Na, Yuzhi Guo, Feng Jiang, Junzhou Huang, Hehuan Ma

    University of Texas at Arlington
  ]
]

#show heading: it => {
  text(fill: blue, bottom-edge: "bounds", baseline: 14pt)[
    *#it.body*
    #line(length: 100%, stroke: 2pt+blue)
  ]
}

#show outline: it => {text(fill: blue, it)}

#polylux-slide[
  *#outline(fill: none, depth: 1)*
]

#polylux-slide[
  = Introduction

  *Background -- AI Research Evolution*

  #v(30pt)

  - Past: A well defined task, a well defined model --- LeNet, AlexNet.
  - Now:
    - Data-driven, large-scale model --- BERT, GPT, ViT.
    - Those foundamental models can be adapted across a wide range of tasks and domains.

]

#polylux-slide[
  #side-by-side[
    == Introduction

    *SAM --- Segment Anything Model*

    - *Purpose*: SAM is designed to perform image segmentation tasks with
    a high degree of versatility, enabling it to understand and
    segment virtually any object or region within an image as
    instructed by user-generated textual prompts.

  ][

    #v(40pt)
    #figure(
      image("./p1.png"),
      caption: [Promptable segmentation, figure from SAM.],
      numbering: none
    )
  ]
]

#polylux-slide[
  #side-by-side[
    == Introduction

    *SAM --- Segment Anything Model*

    - *Foundation*: At its core, SAM utilizes a transformer-based
    architecture --- VIT, similar to those found in state-of-the-art natural
    language processing (NLP) models. This design choice allows SAM to
    effectively process and interpret the complex relationships
    between visual elements and textual prompts.

  ][

    #v(40pt)
    #figure(
      image("./p2.png"),
      caption: [Segment Anything Model, figure from SAM.],
      numbering: none
    )

  ]
]

#polylux-slide[
  == Introduction

  *SAM Challenges in specialized areas like nuclei segmentation*

  #figure(
    image("./p3-a.png", height: 8cm),
    caption: [Natural Image Segmentation.],
    numbering: none
  )

]

#polylux-slide[
  == Introduction

  *SAM Challenges in specialized areas like nuclei segmentation*

  #figure(
    image("./p3-b.png", height: 8cm),
    caption: [Nuclei Segmentation.],
    numbering: none
  )

]

#polylux-slide[
  = Related works

  #cite(<hörst2023cellvit>, form: "full")
  - Which everages pre-trained ViT encoders, such as ViT256 and SAM,
  and its own decoder to do the cell segmentation.
  - No prompt for the mask decoder.

  #cite(<MedSAM>, form: "full")
  - Which fine-tuning SAM with more than one million medical image-mask pairs.
  - Still need professionals' to point out the nuclei in the prompt.

]

#polylux-slide[
  = Methodology

  #figure(
    image("./p4.png", height: 11cm),
    caption: text(size: 16pt)[*Segment any cell framework.*],
    numbering: none
  )

]

#polylux-slide[
  == Methodology

  *Auto-prompt generator*

]

#polylux-slide[
  == Methodology

  *Discriminating Strategy*

]


#polylux-slide[
  == Methodology

  *LoRA*

]

#polylux-slide[
  = Experiments

  *Dataset*

  - MoNuSeg
  - 2018 Data Science Bowl (DSB)
  - PanNuke

]

#polylux-slide[
  == Experiments

  *MoNuSeg*

]

#polylux-slide[
  == Experiments

  *DSB*

]

#polylux-slide[
  == Experiments

  *PanNuke*

]

#polylux-slide[
  == Ablation Studies

  *Effectiveness of Prompts in SAM Fine-Tuning*

]

#polylux-slide[
  == Ablation Studies

  *Centroid-based prompt selection vs. Direct probability-based prompt selection*

]

#polylux-slide[
  == Ablation Studies

  *Efficiency Analysis*

]

#polylux-slide[
  == Ablation Studies

  *Enhancement Through Incremental Prompt Amplification*

]

#polylux-slide[
  == Ablation Studies

  *SAM VIT Backbone Comparison*

]



#polylux-slide[

  = Conclusion

  - Segment Any Cell Framework
  - Auto Prompting
  - LoRA

]


#polylux-slide[

  #bibliography("main.bib", style: "nature", title: "Reference", full: true)

]
