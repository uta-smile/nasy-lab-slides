#import "@preview/touying:0.5.3": *
#import themes.simple: *
#import "@preview/pinit:0.2.2": *
#import "@preview/shadowed:0.1.2": shadowed

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: context {
    set text(size: 12pt, fill: rgb("#2e59a7"), weight: "semibold")
    v(1.5em)
    grid(
      columns: (2fr, 3fr, 2fr),
      align: (left, center, right),
      [Dept. CSE, UT Arlington],
      [Scalable Modeling & Imaging & Learning Lab (SMILE)],
      utils.slide-counter.display() + " / " + utils.last-slide-number,
    )
  },
  primary: rgb("#526069"),
  config-methods(
    init: (self: none, body) => {
      show footnote.entry: set text(size: .6em)

      set text(
        fill: self.colors.neutral-darkest,
        size: 25pt,
        font: "EB Garamond",
        fallback: true,
      )
      show heading.where(level: 2): it => {
        set text(fill: red)
        text(bottom-edge: "bounds", weight: "light")[
          *#it.body*
          #v(-30pt)
          #line(length: 100%, stroke: 2pt + gradient.linear(..color.map.mako))
          #v(-20pt)
        ]
      }
      show link: underline

      set math.equation(numbering: "(1)")
      set quote(block: true, quotes: false)
      show quote: set align(center)
      show quote: set text(size: 22pt)


      body
    },
    alert: utils.alert-with-primary-color,
  ),
  config-common(new-section-slide-fn: body => centered-slide([
    #text(2em, weight: "bold", utils.display-current-heading(level: 1))

    #body
  ])),
  config-colors(
    // neutral-lightest: rgb("#ffffff"),
    neutral-lightest: rgb("#F2ECDE"),
    neutral-darkest: rgb("#526069"),
  ),
  footer-right: none,
  foreground: rgb("#526069"),
  subslide-preamble: context {
    block(
      below: 0.4em,
      text(1.4em, weight: "semibold", utils.display-current-heading(level: 2)),
    )
    line(length: 100%, stroke: 2pt + gradient.linear(..color.map.mako))
    v(-0.5em)
  },
)

#let fcite(key) = footnote(cite(key, form: "full"), numbering: x => "")

#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 {
      center
    } else {
      left
    }
  ),
)

#title-slide[
  = *Optimizer and LR Schedule*
]

== Outline <touying:hidden>

#show outline.entry: it => {
  text(it.body)
}
#components.adaptive-columns(
  outline(fill: none, title: none, depth: 1, indent: 1em),
)

= Introduction

== Why?

- *Core component* of deep learning
  - Drives the entire training process
  - Determines how models learn from data
  - Critical for model convergence

- *Impact*
  - Training speed
  - Model performance
  - Final accuracy
  - Generalization

== Challenges

- Complex loss landscapes
  - Non-convex optimization
  - Multiple local minima and Saddle points

- Training
  - Vanishing/exploding gradients
  - Slow convergence
  - Unstable
  - Overfitting

== Optimizer vs Schedulers

=== Optimizer

- How parameters update
- adapt learning based on gradients

=== Learninig rate schedule

- Manage learning rate dynamics
- Balance exploration vs exploitation
- Convergence speed
- Final model perormance

== Evolution

- Traditional
  - Basic Gradient Descent
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - mini-batch

- Modern
  - Momentum
  - Adaptive learning rates
  - Combined strategies

= Optimizer

== Common Optimizers Family

- First Generation
  - SGD
  - SGD with momentum
  - Nesterov accelerated gradient

#grid(
  columns: (1fr, 1fr),
  [
    - Adaptive methods
      - AdaGrad
      - RMSprop
      - AdaDelta
      - Adam
  ],
  [
    - Modern
      - AdamW
      - Lion
      - Lamb
      - ...
  ],
)

== Gradient Descent

$ theta = theta - eta * nabla J(theta) $

Where,

- $theta$: model parameters
- $eta$: learning rate
- $nabla J(theta)$: gradient of loss function

== Batch Gradient Descent

- Uses entire dataset for each update
- Slow
- High memory
- Deterministic updates

For a dataset with $n$ sample, $i in [1, n]$:

$
  L &= (1 / n) * sum L(theta; x_i, y_i) \
  theta &= theta - eta * (1 / n) * sum nabla L(theta; x_i, y_i)
$

== Batch Gradient Descent Pseudo code

```python
for epoch in epochs:
  grads = grad(loss_fn(weights, all_n_samples))
  weights = weights - learning_rate * grads
```

== SGD and Mini-batch

- Update parameters for each sample or a $m$-size batch

$
  L &= (1 / m) * L(theta; x_i, y_i) \
  theta &= theta - eta * (1 / m) * nabla L(theta; x_i, y_i)
$

```python
for epoch in epochs:
  for batch in get_batches(dataset, size=m):
    grads = grad(loss_fn(weights, batch))
    weights = weights - learning_rate * grads
```

== AdaGrad (Adaptive Gradient)

- Adaptive learning rate for each parameter
- Larger updates for infrequent parameters
- Smaller updates for frequent parameters

$
  r_t &= r_(t-1) + nabla J(theta_(t-1))^2 \
  theta_t &= theta_(t-1) - (eta / sqrt(r_t + epsilon)) * nabla J(theta_(t-1))
$

#pause

- Learning rate will decrease over time

== Momentum

- Inpire by physics momentum
- Accumulate gradients history
- Helps overcome local minima
- Reduce training oscillations

$
  v_t &= beta v_(t-1) + (1 - beta)nabla J(theta_(t-1)) \
  theta_t &= theta_(t-1) - eta v
$

- $v$: velocity (momentum)
- $beta$: momentum coefficient (0.9 in practice)


```
Without momentum (oscillating):

  /\    /\/\/\
 /  \/\/
/

With momentum (smooth):

  /^^^^^
 /
/
```

== Weight Decay

- Prevents model overfitting
- Penalize large weights
- Encourage simpler models
- Reduces model's dependency on single features

== Weight Decay

=== L2 norm in the loss function

$ L = L + (lambda / 2) ||theta||^2 $

=== Standard Weight Decay in optimizer

$ theta_t = theta_(t-1) - eta nabla J(theta_(t-1)) - lambda theta_(t-1) $

== Weight Decay pseudo code

```python
for params, grad in zip(params, grads):
  param = param - lr * (grad + wd * param)
  # or
  param = param - lr * grad - lr * wd * param
```

== Some Typical values for $lambda$

#table(
  columns: (1fr, 1fr),
  table.header([Dataset Size], [$lambda$ Value]),
  [Small ($<$ 10k sample)], [1e-3],
  [Medium (10k - 1M)], [1e-4],
  [Large ($>$ 1M)], [1e-5],
)

#table(
  columns: (1fr, 1fr),
  table.header([Architecture Type], [$lambda$ Adjustment]),
  [CNN], [Base $lambda$],
  [Transformer], [0.1 × Base $lambda$],
  [ResNet], [0.5 × Base $lambda$],
)

---

#table(
  columns: (1fr, 1fr),
  table.header([Data Type], [$lambda$ Adjustment]),
  [Simple/Linear], [2 × Base $lambda$],
  [Complex/Nonlinear], [0.5 × Base $lambda$],
)

#table(
  columns: (1fr, 1fr),
  table.header([Training Length], [$lambda$ Adjustment]),
  [Short ($<$ 20 epochs)], [0.5 × Base $lambda$],
  [Long ($>$ 100 epochs)], [2 × Base $lambda$],
)

== SGD with Momentum

Add momentum term to vanilla SGD

$
  v_t &= beta * v_(t-1) + nabla J(theta_(t-1)) \
  theta_t &= theta_(t-1) - eta * v_t
$

#text(size: 20pt)[

  === Nesterov Accelerated Gradient

  - Momentum: Current $->$ Gradient $->$ Momentum $->$ Update
  - Nesterov: Current $->$ Look-ahead $->$ Gradient $->$ Momentum $->$ Update

  $
    v_t &= beta * v_(t-1) + nabla J(theta_(t-1) + beta * v_(t-1)) \
    theta_t &= theta_(t-1) - eta * v_t
  $
]

== RMSprop

- Improves AdaGrads' declining learning rate with exponential moving average (EMA)

#text(size: 18pt)[

  === Before

  $
    r_t &= r_(t-1) + nabla J(theta_(t-1))^2 \
    theta_t &= theta_(t-1) - (eta / sqrt(r_t + epsilon)) * nabla J(theta_(t-1))
  $

  === After

  $
    r_t &= beta * r_(t-1) + (1 - beta) * nabla J(theta_(t-1))^2 \
    theta_t &= theta_(t-1) - (eta / sqrt(r_t + epsilon)) * nabla J(theta_(t-1))
  $
]

== Adam

#text(size: 20pt)[

  - Adds momentum to RMSprop
  - Maintains moving averages of gradients (momentum) and squared gradients (LR)

    $
      m_t &= beta_1 * m_(t-1) + (1 - beta_1) * nabla J(theta_(t-1)) \
      v_t &= beta_2 * v_(t-1) + (1 - beta_2) * nabla J(theta_(t-1))^2 \
      m_t &= m_t / (1 - beta_1^t) \
      v_t &= v_t / (1 - beta_2^t) \
      theta_t &= theta_(t-1) - (eta / sqrt(v_t + epsilon)) * m_t
    $
]

== AdamW

- Add weight decay to Adam

$
  theta_t = theta_(t-1) - (
    eta / sqrt(v_t + epsilon)
  ) * m_t - lambda * theta_(t-1)
$

== Lion (Google in 2023)

#fcite(<chenSymbolicDiscoveryOptimization2023>)

- Use sign gradient instead of raw gradient
- Reduce memory usage

$
  m_t &= beta_1 * m_(t-1) + (1 - beta_1) * op("sign")(nabla J(theta_(t-1))) \
  theta_t &= theta_(t-1) - eta * op("sign")(m_t)
$

== SOTA Optimizers

=== SOAP#super[2]#fcite(<vyasSOAPImprovingStabilizing2024>)

#v(-2em)

#figure(image("p1.png", width: 80%))

=== Muon#super[3]#fcite(<jordanKellerJordanModdednanogpt2024>)

#v(-2em)

#figure(image("p2.jpeg", height: 90%))


= Schedule

== Why?

Problems with fixed LR

- Large: training unstable
- Small: slow convergence

== Common Schedules

- Step Decay
- Cosine Annealing
- Warm-up
- One Cycle
- Reduce on Plateau
- ...

== Step Decay

$ eta = eta_("init") gamma^floor("epoch" / "step_size") $

Where $gamma$ is the decay factor and *$"step_size"$* is the number of epochs to decay.

```python
lr = [
  0.1,  # epoch 0-30
  0.01,  # epoch 30-60
  0.001,  # epoch 60-90
  ...
]
```

== Cosine Annealing

#fcite(<loshchilovSGDRStochasticGradient2017>)

$
  eta_t = eta_min + (1 / 2) * (eta_max - eta_min) * (
    1 + cos( T_"cur" / T_max pi)
  )
$

Where $t$ is the current epoch, $T_"cur"$ is the current step, and $T_max$ is the total number of steps.

== Warm-up Strategy

- Stabilizes early training
- Prevents early divergence

```python
if step < warmup_steps:
  lr = warmup_schedule(step)
else:
  lr = normal_schedule(step)
```

== One Cycle

+ Linearly increase LR to max lr (warmup)
+ Linearly decrease LR to min lr
  - Optionally, use other annealing (like cosine) to decrease LR further

#text(size: 20pt)[
  ```python
  if step < warmup_steps:
    lr = linear_increase(step, lr)
  else:
    lr = cosine(step - warmup_steps, total_steps - warmup_steps)
  ```
]

== Reduce on Plateau

#text(size: 16pt)[
  Adaptize Learning Rate Strategy

  ```python
  def update(current_metric):
    # Check if we should change the learning rate or not by comparing the metrics
    is_better = compare(current_metric, best_metric)

    # Change the best metric
    if is_better:
      best_metric = current_metric
      num_bad_epochs = 0
    else:
      num_bad_epochs += 1

    # Change the learning rate if necessary
    if num_bad_epochs >= patience:
      lr = max(lr * factor, min_lr)
      num_bad_epochs = 0
  ```
]

== Hyperparameter suggestion in LR Schedules

=== Step Decay

#table(
  columns: (1fr, 1fr),
  table.header([Parameter], [Common Values]),
  [step_size], [2000-4000 steps],
  [gamma], [0.1-0.5],
)

=== Warm-up

#table(
  columns: (1fr, 1fr),
  table.header([Model Type], [Warmup Steps]),
  [CNN], [5% steps],
  [Transformer], [10% steps],
  [Fine-tuning], [6% steps],
  [Large batch size $>$ 1000], [15% - 20%],
  [Transfer learning], [3-5%],
  [Small datasets], [3-5%],
  [Unstable training], [Increase 5%],
)

== Learning Rate Range Suggestions

=== LR Range

#table(
  columns: (1fr, 1fr),
  table.header([Optimizer], [Learning Rate Range]),
  [SGD (no momentum)], [0.1 - 1.0],
  [SGD with Momentum], [0.01 - 0.1],
  [Adam/AdamW], [1e-4 - 1e-3],
  [RMSprop], [1e-4 - 1e-3],
  [AdaGrad], [0.01 - 0.1],
  [Lion], [1e-4 - 3e-4],
)

---

=== Adjustments

#table(
  columns: (1fr, 1fr),
  table.header([Scenario], [LR Adjustment]),
  [Batch size doubled], [LR × $sqrt(2)$],
  [Deeper network], [LR × 0.5],
  [Fine-tuning], [LR × 0.1],
  [Unstable training], [LR × 0.1],
)

== SOTA Schedule The Road Less Scheduled#super[5]

#fcite(<defazioRoadLessScheduled2024>)

#v(-1.5em)

#figure(image("p3.png", height: 70%))

---
#fcite(<defazioRoadLessScheduled2024>)

#v(-1.5em)

#figure(image("p4.png", width: 80%))

== SOAP & Schedule Free

#v(-0.5em)

#figure(image("p5.jpeg", height: 89%))

= Reference

#slide[
  #bibliography("tricks.bib", title: none, style: "nature")
]
