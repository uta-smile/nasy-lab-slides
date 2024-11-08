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
  = KAN -- Kolmogorov-Arnold Networks
  #v(30pt)

  Nasy

  #datetime.today().display("[month repr:short] [day], [year]")
]

#slide[
  == Table of Contents
  #touying-outline()
]

= Introduction

== Motivation

#v(1em)

- Design for interpretable AI for Science, Physics

== Kolmogorov-Arnold Representation Theorem (KART)

[From wikipedia]

If $f$ is a multivariate continuous function, then $f$ can be written as a
finite composition of continuous functions of a single variable and the binary
operation of addition.

$ f(X) = f(x_1, ..., x_n) = sum_(q=1)^(2n+1) Phi_q (sum_(p=1)^n phi_(q,p(x_p))) $

where $phi_q,p:[0,1] -> RR$ and $Phi_q : RR -> RR$

== Intuitive Picture of KART

#figure(image("./p1.png", height: 80%))

== KART to KAN

#figure(image("./p2.png", height: 80%))

== An example of KAN

#grid(columns: (1.5fr, 1fr))[

  #align(horizon)[

    Fit function

    $ exp(sin(x_1^2 + x_2^2) + sin(x_3^2 + x_4^2)) $

    Which may need three layers of KAN

  ]
][

  #figure(image("./p3.png", height: 80%))
]

== MLP vs KAN

#v(-28pt)

#figure(image("./p4.png", height: 70%))

#v(-20pt)

#text(size: 16pt)[

  `MLP = einsum("ij,j->i", w1, sigma(input))`

  `KAN = einsum("ijk,jk->i", w2, phi(input))`

  `input = (d,); w1 = (out, d); w2 = (out, d, 1 + m); phi(input) = (d, 1 + m)`

]

== Functions Represented by KAN

#figure(
  image("./p5.png", height: 80%),
)

= Application

KAN for scientific discoveries

= Theory

#v(1em)

- Proof and algorithm detail please see the section 2.2 in the paper
- $phi$ is Basic Spline


= When to Use It?

#figure(
  image("./p6.png", height: 80%),
)

== Conclusion

#v(1em)

- Kolmogorov-Arnold Representation Theorem (KART)
- KART to KAN
- MLP vs KAN
- Examples

// #grid(columns: (2fr, 1fr))[

//   *SSM*
//   - What is SSM.
//   - Why? Continuous, recurrent, convolutional.
//   - Common SSM Layers -- S4 and H3

//   *Mamba*
//   - Selection Mechanism
//   - Hardware-aware Algorithm
//   - Simpler Architecture
// ][
//   *Linear Attention*
//   - KV Cache
// ]
