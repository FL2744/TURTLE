# TURTLE: Trajectory-based Understanding and Rendering of Transformations in Latent Embeddings

TURTLE compares three ways of turning latent representations into visible 3D
paths, including a bundled demonstration made from real BERT hidden states.

## Three modes

### Glyph mode — encoding

The original artistic and pedagogical mapping:

- `abs(value)` controls segment length.
- Positive values turn right.
- Negative values turn left.
- A selectable fixed angle separates consecutive segments.
- An index-driven roll changes the local turning plane to create 3D structure.

This is a transformation from a vector into a turtle signature. The displayed
z-structure is partly introduced by the encoding algorithm. Decorative tools
such as twist, expand, smooth, and click-to-deform are available in this mode.

### Geometric mode — projection

The turtle walks literally through the original coordinate space. For a vector
`v` with `n` coordinates, point `i` contains the first `i` coordinates of `v`
and zeros for the remaining coordinates. The 769th point is therefore the
actual embedding vector.

The complete n-dimensional trajectory is centered and projected to three
principal-component score dimensions with PCA. No artificial turns, roll, or
manual deformation are applied. Decorative geometry controls are disabled so
the visible form remains data-derived.

Segment color retains the sign of the source coordinate after projection:
positive dimensions are cyan and negative dimensions are magenta by default.
Click a geometric point or segment to see its original dimension number,
signed value, sign, and magnitude. When the path is resampled, the readout
reports the closest source dimension along the path.

The interface labels these modes as **Encoding** and **Projection** to prevent
the two interpretations from being confused.

### Trajectory mode — real model states

The bundled research demonstration uses `google-bert/bert-base-uncased` to
extract the representation of **jaguar** from the embedding output and every
one of BERT-base's 12 transformer blocks. It compares:

- “The jaguar rested quietly in the jungle.”
- “The Jaguar accelerated quickly out of the garage.”

Each context produces a real `13 × 768` hidden-state matrix. TURTLE concatenates
both matrices, fits one shared PCA basis, and then separates the projected
paths for display. Because the PCA basis is shared, their relative positions
and directions can be compared. Click any trajectory layer for its context,
layer number, and source sentence.

## Research trajectory support

`representation_trajectory_to_3d(states)` accepts a matrix of actual model
representations—such as one token's hidden state after each transformer
layer—and projects that genuine representational trajectory to 3D with PCA.
This provides a foundation for a later layer, translation, or contextual-state
trajectory interface.

`project_trajectories_shared(*trajectories)` is the appropriate function for
comparative work: it fits one PCA basis across every supplied trajectory.

The complete matrices and extraction provenance are stored in:

```text
data/bert_jaguar_hidden_states.npz
data/bert_jaguar_metadata.json
```

To reproduce them, install the extraction dependencies and rerun:

```bash
./.venv/bin/python -m pip install -r requirements-demo.txt
./.venv/bin/python extract_bert_demo.py
```

## Run locally

```bash
cd /Users/will/VIBES/TURTLE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python TURTLE.py
```

The browser visualization is written to `turtle_visualization.html` and opened
automatically.

## Run in Google Colab

Upload `TURTLE.py`, then run:

```python
%run /content/TURTLE.py
```

## Controls

- **Mode** switches among glyph encoding, coordinate projection, and the real
  BERT trajectory comparison.
- **Random reset** creates a new normally distributed 768-value vector and
  rebuilds both modes.
- **Path points** resamples the displayed path for presentation.
- **Consecutive angle** applies only to glyph mode.
- Decorative shape and manual deformation controls apply only to glyph mode.
- Geometric-mode sign colors are independently adjustable; clicking its path
  reports the corresponding source coordinate and value.
- Trajectory colors identify the two contexts; clicking a layer reports its
  model stage and sentence.
- Rotation, orbit, zoom, colors, line width, and point size are presentation
  controls available in both modes.

## GitHub Pages

The standalone site is `docs/index.html`. In repository settings, configure
GitHub Pages to deploy the `/docs` folder from the default branch.

## Files

```text
TURTLE.py                 Python generator and notebook application
extract_bert_demo.py      Reproducible BERT hidden-state extractor
data/                     Real 13 x 768 example matrices and provenance
docs/index.html           Standalone GitHub Pages build
requirements.txt          Local runtime dependency
requirements-demo.txt     Optional BERT extraction dependencies
requirements-notebook.txt Optional notebook dependencies
```

## Scientific cautions

- The glyph is an encoding, not a projection of the original n-dimensional
  geometry.
- The geometric path is a PCA projection, so 3D distances do not preserve all
  information from 768 dimensions.
- Resampling reduces the number of visible trajectory points.
- A screenshot cannot be used for exact vector reconstruction.
- Similar-looking glyphs or projections should not be assumed to imply similar
  embeddings without empirical validation.
- The BERT comparison uses a shared PCA basis, but a 3D projection still omits
  most of the variance present in the original 768-dimensional states.

## License

MIT License. See `LICENSE`.
