#import "@preview/touying:0.3.3": *
#import "@preview/pinit:0.1.3": *

#let pinit-highlight-equation-from(
  height: 2em, pos: bottom, fill: rgb(0, 180, 255), dx: 0em,
  highlight-pins, point-pin, body,
) = {
  pinit-highlight(
    ..highlight-pins, dy: -0.6em, fill: rgb(..fill.components().slice(0, -1), 40),
  )
  pinit-point-from(
    fill: fill, pin-dx: -0.6em, pin-dy: if pos == bottom { 0.8em } else { -0.6em },
    body-dx: dx, body-dy: if pos == bottom { -1.7em } else { -1.6em }, offset-dx: -0.6em,
    offset-dy: if pos == bottom { 0.8em + height } else { -0.6em - height }, point-pin,
    rect(inset: 0.5em, stroke: (bottom: 0.12em + fill), {
      set text(fill: fill)
      body
    }),
  )
}

#let s = themes.simple.register(
  aspect-ratio: "16-9", footer: context{
    set text(size: 12pt, fill: rgb("#2e59a7"), weight: "medium")
    v(1.5em)
    grid(
      columns: (2fr, 3fr, 2fr), align: (left, center, right), [Dept. CSE, UT Arlington],
      [Scalable Modeling & Imaging & Learning Lab (SMILE)], states.slide-counter.display() + " / " + states.last-slide-number,
    )
  }, footer-right: none, foreground: rgb("#526069"),
  primary: rgb("#526069"),
  // background: rgb("#F2ECDE"),
  background: rgb("#FFF"),
)

// Extract methods
#let (init, slides, touying-outline, alert) = utils.methods(s)

#show: init

// Place global settings here
#set text(size: 25pt, font: "EB Garamond", fallback: true)

#show heading.where(level: 2): it => {
  text(bottom-edge: "bounds")[
    *#it.body*
    #v(-30pt)
    #line(length: 100%, stroke: 2pt + gradient.linear(..color.map.mako))
    #v(-20pt)
  ]
}

#set math.equation(numbering: "(1)")

#show link: underline

#show strong: alert

// Extract slide functions
#let (slide, empty-slide, title-slide, centered-slide, focus-slide) = utils.slides(s)
#show: slides.with(slide-level: 1)

// Title slide
#title-slide[
  = Mamba
  #v(30pt)

  Nasy

  #datetime.today().display("[month repr:short] [day], [year]")
]

#slide[
  == Table of Contents
  #touying-outline()
]

= Introduction

*Motivation*:
- Most current works ared based on Transformer architecture.
- However, transformers is inefficiency on long sequences.
- Current efficiency works not perform as well as transformers.
- Thus, they propose a new model, mamba, to achieve the modeling power of
  Transformers while scaling linearly in sequence length.
  - *New Selection Mechanism, Hardware-aware Algorithm, Simpler Architecture* on the
    prior works.

= SSM

== Content

#align(horizon)[
  - What is State Space?
  - What is SSM?
  - Why SSM?
  - Motivation
  - SSM to NN Layer -- s4 Model
  - SSM to NN Architectures -- H3 Layer
]

#slide(title: "What Is State Space?", composer: (2fr, 1fr))[
  #align(center, image("ssmap.png", height: 80%))
][
  #align(horizon)[
    A State Space contains the minimum number of variables that fully describe a system
  ]
  #pause

  The “state space” is the map of all possible locations (states)
]

== What Is State Space Model (SSM)?

#grid(columns: 2)[
  #align(center, image("ssm.png", height: 38%))
][
  #align(horizon + center)[
    $ x'(t) &= A x(t) + B u(t) \
    y(t)  &= C x'(t) + D u(t) $
  ]
]

#v(-34pt)

#align(center)[
  #grid(columns: 2, align: left, column-gutter: 4em)[
    - *$u(t)$*: Input
    - *$y(t)$*: Ouput
    - *$x(t)$*: State
    - $x(t)' = d /(d t)x(t))$:\
      The differential of the state
  ][
    - *$A$*: State matrix
    - *$B$*: Input matrix
    - *$C$*: Output matrix
    - *$D$*: Feedforward matrix
  ]
]

== Why SSM?

#text(
  size: 22pt,
)[

  *Deep sequence models meet three challenges* (at the time paper was proposed):

  - *Generalization*
    - RNN, CNN, Transformer, Neural Differential Equations (NDEs) in different field
  - *Computational Efficiency*
    - CNNs and Transformers are not efficient autoregressive inference
  - *Long range dependencies*
    - Gradient vanishing/exploding problem
    - Limitation of the length of the context.
]

== Motivation

A simple and base model to archive the above challenges is the State Space Model
(SSM).

- SSM is continuous (differential equations)
- SSM is recurrent (after discretization, if time invariant)
- SSM is convolutional (time invariant)

== SSM in Deep Learning

In a continuous system, how to map a function $x(t) in bb(R)$ to anoher function $y(t) in bb(R)$ through
an implicit latent state $h(t) in bb(R)^N$?

#v(20pt)

#grid(
  columns: (1fr, 1fr),
)[

  - *Input*: $x(t) in bb(R)^(1 times D)$
  - *Output*: $y(t) in bb(R)^(1 times D)$
  - *Latent State*: $h(t) in bb(R)^(N times D)$
  - *Derivative of $h(t)$*: $h'(t)$
  - $bold(Delta)$: Time step. for discretization

][

  #text(
    22pt,
  )[

    #v(20pt)
    $ h'(t) &= bold(A)h(t) + bold(B)x(t) \
    y(t)  &= bold(C) h'(t) + bold(D)x(t) $

    #v(-15pt)
    where, $bold(A)$, $bold(B)$, $bold(C)$, and $bold(D)$ are the learnable
    parameters. ]
]

== S4 Model -- Structure State Space Sequence Model

#align(center + horizon)[

  #text(32pt)[

    $"SSM" attach(-->, t: "discrete") cases("Recurrent representation" \
    "Convolutional representation") $
  ]
]

== Discretization? Recurrent? Convolutional?

#text(
  20pt,
)[

  #grid(
    columns: (1.2fr, 1fr), column-gutter: 40pt,
  )[

    $ h'(t) &= bold(A)h(t) + bold(B)x(t) \
    y(t)  &= bold(C) h'(t) + bold(D)x(t) $ <eq:1>

    Here, $bold(A)$, $bold(B)$, $bold(C)$, $bold(D)$ are continuous parameters.

    We can discretize them to discrete parameters $macron(bold(A))$, $macron(bold(B))$, $macron(bold(C))$, $macron(bold(D))$ by
    the step parameter $Delta$:

    #v(-10pt)
    $ h_t &= macron(bold(A))h_(t-1) + macron(bold(B))x_t \
    y_t &= macron(bold(C)) h_t + macron(bold(D))x_t $

    #v(-15pt)
    #text(
      16pt,
    )[
      where $macron(bold(A)) = exp(bold(Delta A))$, \
      $macron(bold(B)) = (bold(Delta A))^(-1)(exp(bold(Delta A - I)) dot.c bold(Delta B))$,
      \
      $macron(bold(C)) = bold(C)$, $macron(bold(D)) = bold(D)$.
    ]
  ][

    #text(
      size: 15pt,
    )[
      *Calculation*

      $h_0 = bold(macron(B))x_0; space h_1 = bold(macron(A))bold(macron(B))x_0 + bold(macron(B))x_1; space \
      h_2 = bold(macron(B))x_2 = bold(macron(A))^2bold(macron(B))x_0 + bold(macron(A))bold(macron(B))x_1 + bold(macron(B))x_2 \
      ...$

      $y_0 &= bold(macron(C))bold(macron(B))x_0 + bold(macron(D))x_0\
      y_1 &= bold(macron(C))bold(macron(A))bold(macron(B))x_0 + bold(macron(C))bold(macron(B))x_1 + bold(macron(D))x_1\
      y_2 &= bold(macron(C))bold(macron(A))^2bold(macron(B))x_0 + bold(macron(C))bold(macron(A))bold(macron(B))x_1 + bold(macron(C))bold(macron(B))x_2 \
          &+ bold(macron(D))x_2 \
      ...$

      $bold(macron(K)) &= (bold(macron(C))bold(macron(A))^(i)bold(macron(B)))_(i in [L]) \
                      &= (bold(macron(C))bold(macron(B)), bold(macron(C))bold(macron(A))bold(macron(B)), ..., bold(macron(C))bold(macron(A))^(L-1)bold(macron(B))) \
      y               &= bold(macron(K)) * x$

    ]

  ]

]

== Structure and Dimensions

$ h'(t) &= bold(A)h(t) + bold(B)x(t) \
y(t)  &= bold(C) h'(t) + bold(D)x(t) $

#grid(
  columns: (1fr, 1fr),
)[
  *Input*: $x(t) in bb(R)^(1 times D)$ \
  *Output*: $y(t) in bb(R)^(1 times D)$ \
  *Latent State*: $h(t) in bb(R)^(N times D)$ \

][
  #text(
    22pt,
  )[
    where, $N$ is the number of dimensions in the latent state, can be any numbers
    you want. \
    $D$ is the length of the input and output.
  ]
]

#text(
  22pt,
)[
  #pause
  #v(-1em)
  $bold(A) in bb(R)^(N times N)$ \
  $bold(B), bold(D) in bb(R)^(N times 1), bold(C) in bb(R)^(1 times N)$

  #pause
  For computing efficiently, the $bold(A)$ matrix should have imposing structure
  (diagonal).
]

== SSM Architectures -- H3 Layer

*A single SSM is a standalone sequence transformation*

#text(22pt)[

  #grid(columns: (2fr, 1fr))[

    Some basic transformation for nn layers:
    - *Linear*: Matrix multiplication, $y = w x + b$
    - *CNN*: Convolutional and correlation, $y = k * x$
    - *RNN*: Recurrent Cell, $h_t = u x_t + w h_(t-1)$

    *H3 Layer*
    - Two SSM transformation with two gated connections
    - $Q dot.circle "SSM"_"diag" ("SSM"_"shift" (K) dot.circle V)$
    - $dot.circle$ is point-wise multiplication

  ][

    #align(center + horizon)[
      #image("h3.png", height: 65%)
    ]

  ]
]

= Mamba

== Mamba Three Contributions

#align(horizon)[
  + Selection Mechanism

  + Hardware-aware Algorithm

  + Simpler Architecture
]

#slide(
  composer: (1fr, 1fr), title: [Selection Mechanism -- Problem],
)[
  #text(
    20pt,
  )[
    *The fundamental problem of sequence modeling is compressing context into a
    smaller state*

    #pause

    *Transformer*: Great, but not efficient.

    #pause

    *RNN*: Efficient, but limited by the context.

    #pause

    *SSM*: Fixed the matrix $bold(A)$, $bold(B)$, $bold(C)$, $bold(D)$, also limited
    by the context.
  ]
][

  #align(center + horizon, image("ssm_problem.png", width: 120%))
]

== Selection Mechanism

#v(-0pt)
#align(center, image("select_m.png", width: 70%))

#text(
  20pt,
)[
  $s_B(x) = "Linear"_N (x)$; $s_C(x) = "Linear"_N (x)$; $s_Delta(x) = "Broadcast"_D ("Linear"_1 (x))$
]

== Hardware-aware Algorithm

#text(
  size: 22pt,
)[
  Since $bold(B C D)$ are dynamic, the kernel of SSM convolution is dynamic. Thus,
  the parallel convolution during the training is not possible.

  #pause

  Mamba proposed *parallel scan* to solve this problem.

  $ "scan"("func", "init", "list") -> & "list" \
  "scan"("add", 0, [1..4]) = [   &1, \
    &"add"(1, 2), \
    &"add"("add"(1, 2), 3), \
    &"add"("add"("add"(1, 2), 3), 4)] $

]

== Mamba Parallel Scan I

$ h_t = macron(bold(A)) h_(t-1) + macron(bold(B)) x_t $

$ h_1 &= macron(bold(A)) h_0 + macron(bold(B)) x_1 \
h_2 &= macron(bold(A)) h_1 + macron(bold(B)) x_2 = macron(bold(A))(macron(bold(A)) h_0 + macron(bold(B)) x_1) + macron(bold(B)) x_2 \
h_3 &= macron(bold(A)) h_2 + macron(bold(B)) x_3 = macron(bold(A))(macron(bold(A))(macron(bold(A)) h_0 + macron(bold(B)) x_1) + macron(bold(B)) x_2) + macron(bold(B)) x_3 $

#text(
  22pt,
)[

  E.g.,

  when calculating $macron(bold(A))h_0 + macron(bold(B))x_1$, we can parallel
  calculate the $macron(bold(B)) x_2 + macron(bold(B)) x_3$.

]

#align(center)[
  #image("pscan.png")
]

== Hardware-aware Scan

#align(center)[
  #image("hscan.png", width: 100%)
]

== Simpler Architecture

#grid(columns: (3fr, 1fr))[

  #align(center + horizon)[
    #image("mamba_p.png", width: 85%)
  ]

  Combined the H3 layer with Gated MLP to form the Mamba model.

][

  #v(-2.8em)
  #align(center + horizon)[
    #image("mamba.png", fit: "cover", width: 150%)
  ]

]

= Experiments

== Datasets

*Training*

- The Pile: An 800GB Dataset of Diverse Text for Language Modeling (Leo Gao et al.
  2020)
- SlimPajama: A 627B token, cleaned and deduplicated version of RedPajama (Daria
  Soboleva et al, 2023)

*Evaluation*

- lm-eval-harness (Leo Gao et al, 2021)

== Results I

#align(center, image("r1.png"))

== Results II

#align(center, image("r2.png", height: 80%))

== Results III

#align(center, image("r3.png"))

== More Results And Ablation Studies

For the complete results and ablation studies,\
please see the Paper: https://arxiv.org/abs/2312.00752

= Linear Attention

== Transformer

#slide(composer: (2fr, 1fr))[
  #figure(image("nasy-transformer.png", fit: "contain", height: 75%))
][
  #align(horizon)[
    $a_(1,i) &= op("softmax")((q_1 dot.c k_i) / sqrt(d)) \

    y_1     &= sum_i a_(1,i)v_i$

    $Q K           &-> O(n^2 d)\
    op("softmax") &-> O(n^2)\
    sum           &-> O(n^2 d) $
  ]
]

== KV Cache

#align(horizon + center)[
  #image("kvcache.png", height: 75%)
]

== Convolutions

#grid(columns: (2fr, 1fr))[

  *SSM*
  - What is SSM.
  - Why? Continuous, recurrent, convolutional.
  - Common SSM Layers -- S4 and H3

  *Mamba*
  - Selection Mechanism
  - Hardware-aware Algorithm
  - Simpler Architecture
][
  *Linear Attention*
  - KV Cache
]
