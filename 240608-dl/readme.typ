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
#set quote(block: true)

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
  = The fast train loops
  #v(30pt)

  Nasy

  #datetime.today().display("[month repr:short] [day], [year]")
]

#slide[
  == Table of Contents
  #touying-outline()
]

= Introduction

== Question

Which of the following is the fastest loop for iterating over 1000 epochs and
300,000 samples with batch size 300?

+ Manual loop with pure python
+ PyTorch Dataset and Dataloader w/ multiple workers
+ Tensorflow Dataset
+ Manual loop with numpy

== Manual loop with pure python

#text(size: 22pt)[

```python
from random import sample
from tqdm import tqdm

dataset = list(range(300_000))
for i in tqdm(range(1000)):
  rd = sample(dataset, k=len(dataset))
  for ii in range(0, len(rd), 300):
    rd[ii:ii+300]
    continue
```

]

== PyTorch version

#text(size: 22pt)[

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = torch.arange(300_000)
dl = DataLoader(dataset, batch_size=300, shuffle=True,
                num_workers=10, prefetch_factor=16,
                pin_memory=True, persistent_workers=True)

for i in tqdm(range(1000)):
  for ii in dl:
    continue
```
]

== Tensorflow version

#text(size: 22pt)[

```python
import tensorflow as tf
from tqdm import tqdm

dataset = tf.data.Dataset.range(300_000)
dl = (dataset.shuffle(buffer_size=dataset.cardinality(),
                      reshuffle_each_iteration=True)
             .batch(300)
             .prefetch(tf.data.experimental.AUTOTUNE))

for i in tqdm(range(1000)):
  for _,ii in zip(range(1000), dl):
    continue
```
]

== Manual loop with numpy

#text(size: 22pt)[

```python
import numpy as np
from tqdm import tqdm

dataset = np.arange(300_000)
rng = np.random.default_rng()

for i in tqdm(range(1000)):
  rd = rng.permutation(dataset).reshape(-1, 300)
  for ii in rd:
    continue
```

]

= Why Slow? Why Fast?

== Inside PyTorch Dataset and DataLoader

The basic torch dataset:

#grid(columns: (1fr, 1fr))[

#text(size: 20pt)[

```python
class D(Dataset):

  def __init__(self, data):
    self.data = data

  def __getitem__(self, idx):
    return self.data[idx]

  def __len__(self):
    return len(self.data)
```

]][

Every batch, it will use call batch size times `__getitem__`
method. which is supper slow.

]

== Example

#figure(image("p1.png"))

100 times `__getitem__` call will use around 122 $mu$s.

= How to solve?

== Base idea

PyTorch support one time get multiple items. Thus, we only need to run one time `__getitem__` call.
Down the time from 122 $mu$s to 3.42 $mu$s.

#figure(image("p2.png"))

== Inside Dataset

#align(
  center + horizon,
)[
#quote[Subclasses could also optionally implement :meth:`__getitems__`, for speedup
batched samples loading. This method accepts list of indices of samples of batch
and returns list of samples.]
]

== Custom Dataset

#text(
  size: 20pt,
)[

```python
class FD(D):

  def __getitems__(self, idxs):
    return self[idxs]

d = D(torch.arange(300_000))
dl = DataLoader(d, batch_size=300, shuffle=True)
fd = FD(torch.arange(300_000))
fdl = DataLoader(fd, batch_size=300, shuffle=True, collate_fn=lambda x: x)

for i in tqdm(range(1000), ncols=40):
  for ii in dl:
    pass
```

]

== Or A Custom DataLoader

#v(-10pt)

#text(
  size: 15pt,
)[

```python
class FastDataloader:

  def __init__(self, dataset, batch_size, shuffle=True):
    self.data = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle

  def __iter__(self):
    self.idx = torch.randperm(len(self.data)) if self.shuffle else torch.arange(len(self.data))
    self.i = 0
    return self

  def __next__(self):
    if self.i >= len(self.data):
      raise StopIteration
    batch = self.idx[self.i:self.i+self.batch_size]
    self.i += self.batch_size
    return self.data[batch]
```

]

= Other Tips

== Random number generator

It is slow to generate random number in each epoch for big array.

#v(2em)

#figure(image("p4.png", width: 90%))

== Fast Random Number Generator

We can split the indexes into groups, and shuffle the groups and indexes inside
the groups.

#figure(image("p5.png", width: 85%))

== Fuse operator

#text(
  size: 18pt,
)[

  Pointwise operations such as elementwise addition, multiplication, and math
  functions like sin(), cos(), sigmoid(), etc., can be combined into a single
  kernel. This fusion helps reduce memory access and kernel launch times.

]

#v(-1.6em)

#figure(image("p6.png", width: 70%))

== To Devices Problem

=== How to move a dict of tuple/list/dict of tensors to devices?

```python
xs = {
  "im": {
    "label": torch.rand(1, 224, 224),
    "image": torch.rand(3, 224, 224),
  },
  "cim": torch.rand(3, 224, 224),
  "pim": [torch.rand(3, 224, 224), torch.rand(3, 224, 224)],
}
```

== Automatic Solution

#v(1em)

- Use DataParallel
- Use DistributedDataParallel

== General Solution

#text(size: 18pt)[

```python
def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device):
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")
```

]

== PyTree Structure

There are three common structures for pytorch:

- Torch PyTree:
  https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py
- tensordict & tensorclass: https://pytorch.org/tensordict/stable/index.html
- deepmind tree: https://github.com/google-deepmind/tree.git
- optree: https://github.com/metaopt/optree.git

== PyTree

This a simple version of PyTree:

```python
from torch.utils._pytree import tree_map

xs = tree_map(lambda x: x.to(device), xs)
```

== Tensordict

If you only need to store Tensors, you can use Tensordict

```python
from tensordict import TensorDict

txs = TensorDict.from_dict(xs)
txs = txs.to(device)
```

== optree

Full support for PyTree structure.

```python
import optree

txs = optree.tree_map(lambda x: x.to(device), xs)
```

== IO

PyTorch can load saved model to a specific device.

```python
import io

temp = io.BytesIO()
torch.save(xs, temp)
temp.seek(0)
lxs = torch.load(temp, map_location=device)
```

== JAX

You don't need to manually move the data to devices. However, you can still use `jax.device_put` to
put any structure to devices.

= Conclusion

== Conclusion

#v(1em)

#grid(columns: (1fr, 1fr))[
- Dataloader Loops
  - Idea: less `__getitem__`
  - Custom Dataset
  - Custom Dataloader
][
  - Other tips
    - Random number generator
    - Fuse operator
    - To device
      - DataParallel & DistributedDataParallel
      - Tree Structure
      - IO
]
