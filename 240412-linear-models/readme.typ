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
  primary: rgb("#526069"), background: rgb("#F2ECDE"),
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
  = Linear Transformer And Linear Attention
  #v(30pt)

  Nasy

  #datetime.today().display("[month repr:short] [day], [year]")
]

#slide[
  == Table of Contents
  #touying-outline()
]

= Introduction

== RNN And Transformer Comparison

- RNN:
  - Train: *slow*
  - Inference: *fast*
- Transformer:
  - Train: *fast*
  - Inference: *slow*

== RNN And Transformer

#slide(composer: (2fr, 1fr))[
  #v(-30pt)
  #align(horizon + center)[
    #figure(image("rnn.svg", fit: "contain", height: 70%))
    #text(size: 16pt)[RNN: A fully recurrent network]
  ]
][
  #v(-30pt)
  #align(horizon + center)[
    #figure(image("transformer.png", fit: "contain", height: 70%))
    #text(size: 16pt)[Transformer model archiecture.]
  ]
]

== Why Is Transformer Inference Slow?

#slide(
  composer: (2fr, 1fr),
)[
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

= Linear Transformer

== Papers in Linear Transformer

/ Linear Transformer: #cite(<2006.16236>, form: "full")
/ AFT: #cite(<2105.14103>, form: "full")
/ RWKV: #cite(<2305.13048>, form: "full")

== Transformers

#slide[

  #v(-10pt)

  $ Y = op("softmax")((Q K^T) / sqrt(d))V $

  #v(-10pt)

  $ Y_i = (sum_(j=1)^N exp(q_i dot.c k_j)V_j) / (sum_(j=1)^N exp(q_i dot.c k_j)) $ <att>

  #v(-10pt)

  $ Y_i = (sum_(j=1)^N op("sim")(q_i, k_j)V_j) / (sum_(j=1)^N op("sim")(q_i, k_j)) $ <satt>

][
  #align(
    horizon,
  )[
    The "$op("sim")$" function is a similarity function.

    If $op("sim")(q, k) = op("softmax")((q^T dot.c k)/sqrt(d))$, @att is equivalent
    to @satt.
  ]

]

== Linearized Transformer

If we have a kernel function $phi$, and let

$ op("sim")(q_i, k_i) = phi(q_i) phi(k_i) $

then,

#text(
  size: 24pt,
)[

  $ Y_i &= (sum_(j=1)^N phi(q_i) phi(k_j)^T V_j) / (sum_(j=1)^N phi(q_i) phi(k_j)^T)
  = (phi(q_i) sum_(j=1)^N phi(k_j)^T V_j) / (phi(q_i) sum_(j=1)^N phi(k_j)^T) $ <latt>

]

#v(1em)

We can write it in vectorized,

#v(3.5em)

$ #pin(1) (phi(Q)#pin(2)phi(K)^T) V #pin(3) = #pin(4) phi(Q) (phi(K)^T V) #pin(5) $

#pinit-highlight-equation-from((1, 2, 3), 2, height: 3.5em, pos: bottom, fill: rgb("#519a73"))[
  Traditional Transformer ($O(n^2 d)$)
]

#pinit-highlight-equation-from((4, 5), 5, height: 2.5em, pos: top, fill: rgb("#CCA4E3"), dx: -12.5em)[
  Linearized Transformer ($O(n d^2)$)
]

== Inference

We can expend the sum part in @latt, and it is a recurrent formula:

$ S_i &= sum_(j=1)^i phi(k_j)^T V_j= phi(k_i)^T + sum_(j = 1)^(i - 1) phi(k_j)^T V_j &= phi(k_i)^T + S_(i - 1)\
Z_i &= sum_(j=1)^i phi(k_j)^T = phi(k_i)^T + sum_(j = 1)^(i - 1)phi(k_j)^T         &= phi(k_j)^T + Z_(i - 1) $

Here, we can regard $S_i$ and $Z_i$ as a state, thus, we can reuse them.

== Attention Free Transformer

#align(
  horizon,
)[

  $ Y_i = sigma_q (Q_i) dot.circle (sum_(j = 1)^N exp(K_j + w_(i,j)) dot.circle V_j )/ (sum_(j = 1)^N exp(K_j + w_(i,j))) $

  where, #math.sigma is #math.op("sigmoid") function.

  It is a multi-head attention w/ heads equal to the dimension of embedding. The
  time complexity is #pause $O(n^2 d)$.

]

== Linear Time Attention Free Transformer

#v(-1em)

=== AFT-local

#text(size: 20pt)[
  $ w_(i,j) = cases(w_(i, j) &"if" |i - j| < s, 0 &"otherwise") $
]
#v(-0.5em)
where $s < n$ is a local window size. The time complexity is $O(n s d), s < n$.

#pause

#v(-0.7em)

=== AFT-simple

Drop the $w_(i,j)$ term in AFT. The time complexity is $O(n d)$.

#text(
  size: 20pt,
)[
  $ Y_i = sigma_q (Q_i) dot.circle (sum_(j = 1)^N exp(K_j) dot.circle V_j )/ (sum_(j = 1)^N exp(K_j)) = sigma_q (Q_i) dot.circle sum_(j = 1)^N (op("softmax")(K) dot.circle V)_j $
]

== RWKV

#align(
  horizon,
)[
  - R: The *Receptance* vector acts as the receiver of past information.
  #v(0.5em)
  - W: The *Weight* signifies the positional weight decay vector, a trainable
    parameter within the model.
  #v(0.5em)
  - K: The *Key* vector performs a role analogous to K in traditional attention
    mechanisms.
  #v(0.5em)
  - V: The *Value* vector functions similarly to V in conventional attention
    processes.
]

== RWKV

#slide(
  composer: (3fr, 5fr),
)[
  #v(-0.5em)
  #align(horizon)[
    #figure(image("rwkv.png", fit: "contain", height: 80%))
  ]

][
  #align(
    horizon,
  )[
    #text(
      size: 20pt,
    )[

      $ r_t &= W_r dot.c (mu_r dot.circle x_t + (1 - mu_r) dot.circle x_(t - 1))\
      k_t &= W_k dot.c (mu_k dot.circle x_t + (1 - mu_k) dot.circle x_(t - 1))\
      v_t &= W_v dot.c (mu_v dot.circle x_t + (1 - mu_v) dot.circle x_(t - 1)) $

      #text(
        size: 18pt,
      )[
        $ w k v_t =
        (sum_(i=1)^(t-1)e^(-(t-1-i)w + k_i) dot.circle v_i + e^(u + k_t) dot.circle v_t) /
        (e^(-(t-1-i)w + k_i) + e^(u + k_t)) $
      ]

      $ o_t = W_o dot.c (sigma (r_t) dot.circle w k v_t) $

    ]
  ]
]

= Mamba

== Papers

/ SSM: #cite(<2111.00396>, form: "full")
/ mamba: #cite(<2312.00752>, form: "full")

== State Spaces Model (SSM) Framework

#align(horizon)[
  #figure(image("ssm.png", fit: "contain", height: 75%))
]

== State Spaces Model (SSM)

$ x'(t) &= bold(A) x(t) + bold(B) u(t) \
y(t)  &= bold(C) x(t) + bold(D) u(t) $

/ $bold(u(t))$: the $1-D$ input.
/ $bold(x(t))$: the $N-D$ latent state.
/ $bold(y(t))$: the $1-D$ output.
/ $bold(A comma B comma C comma D)$: the model parameters.

== Discretization

#text(
  size: 18pt,
)[

  $ x'(t) &= bold(macron(A)) x_(t-1) + bold(macron(B)) u_t \
  y(t)  &= bold(macron(C)) x_(t) + bold(macron(D)) u_t $

  where in the _S4_ model,

  $ bold(macron(A)) &= (bold(I) - Delta / 2 dot.c bold(A))^(-1)(bold(I) + Delta / 2 dot.c bold(A)) \
  bold(macron(B)) &= (bold(I) - Delta / 2 dot.c bold(B))^(-1)Delta bold(B) \
  bold(macron(C)) &= C \
  bold(macron(D)) &= D $
]

== Discretization in Mamba

#align(horizon)[
  $ bold(macron(A)) = exp(Delta A) \
  bold(macron(B)) = (Delta A)^(-1)(exp(Delta A) - I) dot.c Delta B $
]

== Recurrent and Convolution

#slide[
  #align(horizon)[
    #figure(image("ssm1.png", fit: "contain", height: 75%))
  ]
][
  #text(
    size: 15pt,
  )[
    $x_0 = bold(macron(B))u_0; space x_1 = bold(macron(A))bold(macron(B))u_0 + bold(macron(B))u_1; space \
    x_2 = bold(macron(B))u_2 = bold(macron(A))^2bold(macron(B))u_0 + bold(macron(A))bold(macron(B))u_1 + bold(macron(B))u_2 \
    ...$

    $y_0 &= bold(macron(C))bold(macron(B))u_0 + bold(macron(D))u_0\
    y_1 &= bold(macron(C))bold(macron(A))bold(macron(B))u_0 + bold(macron(C))bold(macron(B))u_1 + bold(macron(D))u_1\
    y_2 &= bold(macron(C))bold(macron(A))^2bold(macron(B))u_0 + bold(macron(C))bold(macron(A))bold(macron(B))u_1 + bold(macron(C))bold(macron(B))u_2 \
        &+ bold(macron(D))u_2 \
    ...$

    $bold(macron(K)) &= (bold(macron(C))bold(macron(A))^(i)bold(macron(B)))_(i in [L]) \
                    &= (bold(macron(C))bold(macron(B)), bold(macron(C))bold(macron(A))bold(macron(B)), ..., bold(macron(C))bold(macron(A))^(L-1)bold(macron(B))) \
    y               &= bold(macron(K)) * u$

  ]
]

== Selection

#align(horizon)[
  #figure(image("select.png", fit: "contain", height: 75%))
]

== SSM And SSM w/ Selection

#align(horizon)[
  #figure(image("s6.png", fit: "contain", height: 75%))
]

== Mamba

#align(horizon)[
  #figure(image("mamba.png", fit: "contain", height: 75%))
]

== Mamba Results -- Selection

#align(horizon)[
  #figure(image("r1.png", fit: "contain", height: 75%))
]

== Mamba Results -- Scaling Laws

#align(horizon)[
  #figure(image("r2.png", fit: "contain", height: 75%))
]

== Mamba Results -- Zero shot

#v(-2em)

#align(horizon)[
  #figure(image("r3.png", fit: "contain", height: 90%))
]

== Mamba Results -- Efficency Benchmarks

#align(horizon)[
  #figure(image("r4.png", fit: "contain", height: 75%))
]

== More Results And Ablation Studies

For the complete results and ablation studies,\
please see the Paper: https://arxiv.org/abs/2312.00752

= Linear Attention

== KV Cache

#v(-2em)

#align(horizon)[
  #figure(image("kvcache.png", height: 75%))
]

#v(-1em)
#text(size: 16pt)[#cite(<pope2022efficiently>, form: "full")]

= Conclusion

== Summary

- RNN And Transformer
- Linear Transformer
  - Linearized Transformers $Q dot.c (K V)$
  - Attention Free Transformer
  - RWKV
  - SSM And Mamba
- KV Cache

#bibliography("ref.bib", style: "modern-humanities-research-association")
