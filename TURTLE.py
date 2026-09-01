"""
HOW TO RUN TURTLE
=============================

First-time local setup (run in the VS Code terminal):
    cd TURTLE
    python3 -m venv .venv
    ./.venv/bin/python -m pip install --upgrade pip
    ./.venv/bin/python -m pip install numpy ipython ipykernel

VS Code interpreter:
    1. Press Cmd+Shift+P.
    2. Choose "Python: Select Interpreter".
    3. Select:
       <repository-path>/.venv/bin/python

Run locally from the terminal:
    cd TURTLE
    ./.venv/bin/python TURTLE.py

You can also use the VS Code Run/Debug button after selecting the interpreter.
When run locally, the script writes the interactive visualization to:
    <repository-path>/turtle_visualization.html

The HTML visualization should open automatically in the default browser. If it
does not, open turtle_visualization.html manually.

Run in Google Colab:
    1. Upload TURTLE.py to the Colab Files panel.
    2. Run this in a notebook cell:
       %run /content/TURTLE.py

In Colab/Jupyter/VS Code notebooks the visualization renders inside the notebook.
In a normal Python terminal or VS Code debugger it opens as a browser page.
"""

import html as html_module
import json
import uuid
import webbrowser
from pathlib import Path

import numpy as np

try:
    from IPython.display import HTML, display
except ImportError:
    # IPython is only required for inline notebook output. Local terminal and
    # VS Code Run/Debug execution writes a standalone HTML file instead.
    HTML = None
    display = None


def example_embedding(seed=42, spread_scale=5.0):
    """Return one deterministic example 768-dimensional embedding."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(768) * spread_scale


def embedding_to_3d_path(vector, turn_angle_degrees=45.0):
    """Create a 3-D path with sign-controlled right/left fixed-angle turns."""
    vector = np.asarray(vector, dtype=float)
    alpha = np.deg2rad(turn_angle_degrees)
    points = [np.zeros(3)]
    direction = np.array([1.0, 0.0, 0.0])

    world_up = np.array([0.0, 0.0, 1.0])
    for i, value in enumerate(vector):
        # For a heading along +x, cross(heading, up) points toward -y: right.
        local_right = np.cross(direction, world_up)
        if np.linalg.norm(local_right) < 1e-10:
            local_right = np.cross(direction, [0.0, 1.0, 0.0])
        local_right /= np.linalg.norm(local_right)
        local_up = np.cross(local_right, direction)
        local_up /= np.linalg.norm(local_up)

        # Slowly roll the local turning plane to make the path genuinely 3-D.
        # Roll is independent of the embedding sign.
        roll = 0.72 * np.sin(i * np.pi * (3.0 - np.sqrt(5.0)))
        turn_side = np.cos(roll) * local_right + np.sin(roll) * local_up
        turn_side *= 1.0 if value >= 0 else -1.0

        new_direction = (
            np.cos(alpha) * direction + np.sin(alpha) * turn_side
        )
        new_direction /= np.linalg.norm(new_direction)
        points.append(points[-1] + abs(value) * new_direction)
        direction = new_direction

    return np.asarray(points)


def embedding_to_nd_walk(vector):
    """Walk one coordinate axis at a time and finish at the actual vector."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    dimensions = vector.size
    visited = np.arange(dimensions + 1)[:, None] > np.arange(dimensions)[None, :]
    return visited * vector[None, :]


def project_nd_path_to_3d(points_nd):
    """Project an n-dimensional trajectory to three PCA score dimensions."""
    points_nd = np.asarray(points_nd, dtype=float)
    if points_nd.ndim != 2 or points_nd.shape[0] < 2:
        raise ValueError("points_nd must contain at least two trajectory points")
    centered = points_nd - points_nd.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    component_count = min(3, singular_values.size)
    projected = u[:, :component_count] * singular_values[:component_count]
    if component_count < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - component_count)))
    return projected


def embedding_to_geometric_path(vector):
    """Return the PCA shadow of a literal walk through embedding space."""
    return project_nd_path_to_3d(embedding_to_nd_walk(vector))


def representation_trajectory_to_3d(states):
    """Project actual layer/context representation states for research use."""
    return project_nd_path_to_3d(states)


def project_trajectories_shared(*trajectories):
    """Project several trajectories in one PCA basis so they are comparable."""
    arrays = [np.asarray(states, dtype=float) for states in trajectories]
    if not arrays or any(states.ndim != 2 for states in arrays):
        raise ValueError("each trajectory must be a two-dimensional matrix")
    if len({states.shape[1] for states in arrays}) != 1:
        raise ValueError("all trajectories must have the same hidden-state width")
    lengths = [len(states) for states in arrays]
    projected = project_nd_path_to_3d(np.concatenate(arrays, axis=0))
    boundaries = np.cumsum([0, *lengths])
    return [projected[boundaries[i]:boundaries[i + 1]] for i in range(len(arrays))]


def load_bert_demo():
    """Load the bundled contextual Jaguar hidden-state trajectories."""
    data_directory = Path(__file__).with_name("data")
    archive_path = data_directory / "bert_jaguar_hidden_states.npz"
    metadata_path = data_directory / "bert_jaguar_metadata.json"
    if not archive_path.exists() or not metadata_path.exists():
        return [], []
    with np.load(archive_path) as archive:
        states = [archive["animal"], archive["car"]]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["examples"]
    return states, metadata


def normalize_points(points, radius=42.0):
    points = np.asarray(points, dtype=float)
    centered = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = float(np.max(np.ptp(centered, axis=0)))
    return centered if span == 0 else centered * (2.0 * radius / span)


def show_turtle(embedding=None):
    """Display glyph, coordinate projection, and model-trajectory modes."""
    if embedding is None:
        embedding = example_embedding()

    glyph_points = normalize_points(embedding_to_3d_path(embedding))
    geometric_points = normalize_points(embedding_to_geometric_path(embedding))
    trajectory_states, trajectory_metadata = load_bert_demo()
    trajectory_points = project_trajectories_shared(*trajectory_states) if trajectory_states else []
    if trajectory_points:
        combined = np.concatenate(trajectory_points, axis=0)
        normalized_combined = normalize_points(combined)
        split = len(trajectory_points[0])
        trajectory_points = [normalized_combined[:split], normalized_combined[split:]]
    element_id = "embedding_" + uuid.uuid4().hex

    body_html = f"""
    <div id="{element_id}" style="position:relative;width:100%;height:720px;
         background:#060914;border-radius:14px;overflow:hidden;">
      <canvas style="display:block;width:100%;height:100%;touch-action:none"></canvas>
      <div style="position:absolute;left:14px;top:12px;color:#e8edff;
           font:13px system-ui;pointer-events:none">
        <b>TURTLE · 768-dimensional vector</b><br>
        <span data-readout="mode-description">Encoding: magnitude → length · sign → right/left</span><br>
        <span data-readout="angle-row">Consecutive angle = <span data-readout="angle">45</span>° · index-driven 3D roll</span><br>
        <span data-readout="interaction">Click to deform · drag to orbit · wheel to zoom</span>
        <div data-readout="selection" style="margin-top:4px;color:#ffd98a"></div>
        <div data-readout="status" style="margin-top:4px;color:#8ee6a8">Loading canvas…</div>
      </div>
      <div class="embedding-controls" style="position:absolute;right:12px;top:12px;width:210px;
        padding:10px;background:rgba(9,14,32,.88);color:#e8edff;border:1px solid #26345c;
        border-radius:10px;font:12px system-ui;box-shadow:0 6px 22px #0008">
        <label style="display:block;margin:0 0 8px">Mode
          <select data-control="mode" style="box-sizing:border-box;width:100%;margin-top:3px;background:#0d1730;
            color:#fff;border:1px solid #40527e;border-radius:5px;padding:5px">
            <option value="glyph">Glyph · turtle encoding</option>
            <option value="geometric">Geometric · n-D walk + PCA</option>
            <option value="trajectory">Trajectory · real BERT hidden states</option>
          </select>
        </label>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
          <button data-action="pause">Pause</button><button data-action="reset">Random reset</button>
          <button data-action="twist-left">Twist −</button><button data-action="twist-right">Twist +</button>
          <button data-action="contract">Contract</button><button data-action="expand">Expand</button>
          <button data-action="smooth">Smooth</button><button data-action="jitter">Deform</button>
        </div>
        <div data-section="angle" style="display:grid;grid-template-columns:1fr 68px;gap:6px;align-items:end;
          padding-bottom:5px;border-bottom:1px solid #26345c;margin-bottom:6px">
          <label style="display:block;margin:0">Path points
            <input data-control="count" type="number" min="2" max="769" step="1" value="768"
              style="box-sizing:border-box;width:100%;margin-top:2px;background:#0d1730;color:#fff;
              border:1px solid #40527e;border-radius:5px;padding:4px">
          </label>
          <button data-action="apply-count">Apply</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 68px;gap:6px;align-items:end;
          padding-bottom:5px;border-bottom:1px solid #26345c;margin-bottom:6px">
          <label style="display:block;margin:0">Consecutive angle (°)
            <input data-control="angle" type="number" min="1" max="179" step="1" value="45"
              style="box-sizing:border-box;width:100%;margin-top:2px;background:#0d1730;color:#fff;
              border:1px solid #40527e;border-radius:5px;padding:4px">
          </label>
          <button data-action="apply-angle">Apply</button>
        </div>
        <label style="display:block;margin:5px 0">Rotation speed <input data-control="speed" style="width:100%" type="range" min="-3" max="3" step=".05" value="1"></label>
        <label style="display:block;margin:5px 0">Deform strength <input data-control="strength" style="width:100%" type="range" min=".5" max="12" step=".5" value="4.5"></label>
        <label style="display:block;margin:5px 0">Deform radius <input data-control="radius" style="width:100%" type="range" min="2" max="40" step="1" value="12"></label>
        <label style="display:block;margin:5px 0">Line width <input data-control="line" style="width:100%" type="range" min=".25" max="5" step=".25" value="1.25"></label>
        <label style="display:block;margin:5px 0">Point size <input data-control="point" style="width:100%" type="range" min="0" max="6" step=".25" value="1.15"></label>
        <label style="display:block;margin:5px 0">Zoom <input data-control="zoom" style="width:100%" type="range" min="1.5" max="15" step=".1" value="5.8"></label>
        <div data-section="glyph-colors" style="display:flex;justify-content:space-between;margin-top:7px">
          <label>Line <input data-control="line-color" type="color" value="#65d9ff"></label>
          <label>Points <input data-control="point-color" type="color" value="#ffd166"></label>
        </div>
        <div data-section="sign-colors" style="display:none;justify-content:space-between;margin-top:7px">
          <label>Positive <input data-control="positive-color" type="color" value="#65d9ff"></label>
          <label>Negative <input data-control="negative-color" type="color" value="#ff5fa2"></label>
        </div>
        <div data-section="trajectory-colors" style="display:none;justify-content:space-between;margin-top:7px">
          <label>Animal <input data-control="animal-color" type="color" value="#65d9ff"></label>
          <label>Automobile <input data-control="car-color" type="color" value="#ff8a65"></label>
        </div>
      </div>
    </div>
    """

    # Colab executes Javascript output, whereas scripts inside HTML output are sanitized.
    js = r"""
    (() => {
      try {
      const host = document.getElementById(ELEMENT_ID);
      if (!host) throw new Error('Embedding container was not created.');
      const canvas = host.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const getControl = name => host.querySelector(`[data-control="${name}"]`);
      const getAction = name => host.querySelector(`[data-action="${name}"]`);
      const status = host.querySelector('[data-readout="status"]');
      const selection = host.querySelector('[data-readout="selection"]');
      let embedding = EMBEDDING_DATA;
      let glyphBasePath = GLYPH_POINT_DATA;
      let geometricBasePath = GEOMETRIC_POINT_DATA;
      const trajectoryPaths = TRAJECTORY_POINT_DATA;
      const trajectoryMetadata = TRAJECTORY_METADATA;
      let mode = 'glyph';
      let embeddingBasePath = glyphBasePath;
      let pts = embeddingBasePath.map(p => p.slice());
      let yaw = -0.55, pitch = 0.52, spin = 0, zoom = 5.8;
      let rotationSpeed=1, deformStrength=4.5, deformRadius=12;
      let lineWidth=1.25, pointSize=1.15, lineColor='#65d9ff', pointColor='#ffd166';
      let positiveColor='#65d9ff', negativeColor='#ff5fa2';
      let trajectoryColors=['#65d9ff','#ff8a65'];
      let displayedSourceIndices=Array.from({length:embedding.length},(_,i)=>i);
      let paused=false;
      let dragging = false, moved = false, lastX = 0, lastY = 0;
      let projected = [];
      let trajectoryProjected = [];

      function randomGaussian() {
        // Box-Muller transform: independent standard-normal random values.
        let u=0, v=0;
        while (u===0) u=Math.random();
        while (v===0) v=Math.random();
        return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
      }

      function randomEmbedding(length, scale=5) {
        return Array.from({length},()=>randomGaussian()*scale);
      }

      function pcaCoordinateWalk(values) {
        // Kernel PCA of the literal coordinate walk without decorative turns.
        // Row i of the n-D walk contains values[0:i] followed by zeros.
        const n=values.length, q=n+1;
        const prefixA=new Float64Array(n+1);
        const prefixB=new Float64Array(n+1);
        const prefixC=new Float64Array(n+1);
        for (let k=0;k<n;k++) {
          const weight=values[k]*values[k], mean=(n-k)/q;
          prefixA[k+1]=prefixA[k]+weight*(1-mean)*(1-mean);
          prefixB[k+1]=prefixB[k]-weight*mean*(1-mean);
          prefixC[k+1]=prefixC[k]+weight*mean*mean;
        }
        const kernel=Array.from({length:q},()=>new Float64Array(q));
        for (let i=0;i<q;i++) for (let j=i;j<q;j++) {
          const value=prefixA[i]+(prefixB[j]-prefixB[i])+(prefixC[n]-prefixC[j]);
          kernel[i][j]=value; kernel[j][i]=value;
        }
        let basis=Array.from({length:q},(_,i)=>new Float64Array([
          Math.sin((i+1)*.731), Math.cos((i+1)*1.117), Math.sin((i+1)*1.913+.4)
        ]));
        function orthonormalize(matrix) {
          for (let c=0;c<3;c++) {
            for (let previous=0;previous<c;previous++) {
              let dot=0;
              for (let i=0;i<q;i++) dot+=matrix[i][c]*matrix[i][previous];
              for (let i=0;i<q;i++) matrix[i][c]-=dot*matrix[i][previous];
            }
            let length=0;
            for (let i=0;i<q;i++) length+=matrix[i][c]*matrix[i][c];
            length=Math.sqrt(length)||1;
            for (let i=0;i<q;i++) matrix[i][c]/=length;
          }
          return matrix;
        }
        basis=orthonormalize(basis);
        for (let iteration=0;iteration<24;iteration++) {
          const next=Array.from({length:q},()=>new Float64Array(3));
          for (let i=0;i<q;i++) for (let j=0;j<q;j++) {
            const value=kernel[i][j];
            next[i][0]+=value*basis[j][0];
            next[i][1]+=value*basis[j][1];
            next[i][2]+=value*basis[j][2];
          }
          basis=orthonormalize(next);
        }
        const scales=new Float64Array(3);
        for (let c=0;c<3;c++) {
          let eigenvalue=0;
          for (let i=0;i<q;i++) {
            let product=0;
            for (let j=0;j<q;j++) product+=kernel[i][j]*basis[j][c];
            eigenvalue+=basis[i][c]*product;
          }
          scales[c]=Math.sqrt(Math.max(0,eigenvalue));
        }
        return basis.map(row=>[row[0]*scales[0],row[1]*scales[1],row[2]*scales[2]]);
      }

      function embeddingPath(values, angleDegrees) {
        const alpha=angleDegrees*Math.PI/180, ca=Math.cos(alpha), sa=Math.sin(alpha);
        const result=[[0,0,0]]; let direction=[1,0,0];
        const norm=a=>Math.hypot(a[0],a[1],a[2]);
        const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
        for (let i=0;i<values.length;i++) {
          const sign=values[i]>=0 ? 1 : -1;
          let localRight=cross(direction,[0,0,1]);
          if (norm(localRight)<1e-10) localRight=cross(direction,[0,1,0]);
          let n=norm(localRight); localRight=localRight.map(v=>v/n);
          let localUp=cross(localRight,direction);
          n=norm(localUp); localUp=localUp.map(v=>v/n);
          const roll=.72*Math.sin(i*Math.PI*(3-Math.sqrt(5)));
          const turnSide=localRight.map((v,k)=>sign*(Math.cos(roll)*v+Math.sin(roll)*localUp[k]));
          let next=direction.map((v,k)=>ca*v+sa*turnSide[k]);
          n=norm(next); next=next.map(v=>v/n);
          const previous=result[result.length-1], length=Math.abs(values[i]);
          result.push(previous.map((v,k)=>v+length*next[k])); direction=next;
        }
        return result;
      }

      function normalizePath(source, radius=42) {
        const mins=[Infinity,Infinity,Infinity], maxs=[-Infinity,-Infinity,-Infinity];
        for (const p of source) for (let k=0;k<3;k++) {
          mins[k]=Math.min(mins[k],p[k]); maxs[k]=Math.max(maxs[k],p[k]);
        }
        const center=mins.map((v,k)=>(v+maxs[k])/2);
        const span=Math.max(...mins.map((v,k)=>maxs[k]-v)) || 1;
        return source.map(p=>p.map((v,k)=>(v-center[k])*(2*radius/span)));
      }

      function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const rect = host.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function rotatePoint(p) {
        // Animated z rotation, followed by user-controlled x/y orbit.
        let c = Math.cos(spin), s = Math.sin(spin);
        let x = c*p[0] - s*p[1], y = s*p[0] + c*p[1], z = p[2];
        c = Math.cos(yaw); s = Math.sin(yaw);
        let x2 = c*x + s*z, z2 = -s*x + c*z;
        c = Math.cos(pitch); s = Math.sin(pitch);
        return [x2, c*y - s*z2, s*y + c*z2];
      }

      function project(p, w, h) {
        const q = rotatePoint(p);
        const cameraDistance = 150;
        const perspective = cameraDistance / Math.max(35, cameraDistance - q[2]);
        return [w/2 + q[0]*zoom*perspective,
                h/2 - q[1]*zoom*perspective, q[2], perspective];
      }

      function draw() {
        const w = host.clientWidth, h = host.clientHeight;
        ctx.clearRect(0, 0, w, h);
        projected = pts.map(p => project(p, w, h));

        if (mode==='trajectory') {
          trajectoryProjected=trajectoryPaths.map(path=>path.map(p=>project(p,w,h)));
          ctx.lineWidth=lineWidth;
          for (let pathIndex=0;pathIndex<trajectoryProjected.length;pathIndex++) {
            const path=trajectoryProjected[pathIndex], color=trajectoryColors[pathIndex];
            ctx.strokeStyle=color; ctx.shadowColor=color; ctx.shadowBlur=5;
            ctx.beginPath(); ctx.moveTo(path[0][0],path[0][1]);
            for (let i=1;i<path.length;i++) ctx.lineTo(path[i][0],path[i][1]);
            ctx.stroke(); ctx.fillStyle=color;
            for (let i=0;i<path.length;i++) {
              const radius=Math.max(2,Math.min(8,(pointSize+1)*path[i][3]));
              ctx.beginPath(); ctx.arc(path[i][0],path[i][1],radius,0,2*Math.PI); ctx.fill();
            }
            ctx.shadowBlur=0; ctx.font='12px system-ui';
            ctx.fillText(trajectoryMetadata[pathIndex].label,path[path.length-1][0]+8,path[path.length-1][1]-8);
          }
          return;
        }

        const glow = ctx.createLinearGradient(0, 0, w, h);
        glow.addColorStop(0, '#7de8ff'); glow.addColorStop(1, '#716bff');
        ctx.lineWidth = 1.25; ctx.strokeStyle = glow;
        ctx.shadowColor = '#34cfff'; ctx.shadowBlur = 4;
        ctx.lineWidth = lineWidth; ctx.strokeStyle = lineColor;
        ctx.shadowColor = lineColor; ctx.shadowBlur = 4;
        if (mode==='geometric') {
          for (let i=1; i<projected.length; i++) {
            const sourceIndex=displayedSourceIndices[Math.min(i-1,displayedSourceIndices.length-1)] ?? 0;
            const color=embedding[sourceIndex]>=0 ? positiveColor : negativeColor;
            ctx.strokeStyle=color; ctx.shadowColor=color;
            ctx.beginPath(); ctx.moveTo(projected[i-1][0],projected[i-1][1]);
            ctx.lineTo(projected[i][0],projected[i][1]); ctx.stroke();
          }
        } else {
          ctx.beginPath();
          ctx.moveTo(projected[0][0], projected[0][1]);
          for (let i=1; i<projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
          ctx.stroke();
        }

        ctx.shadowBlur = 3;
        for (let i=0;i<projected.length;i++) {
          if (pointSize <= 0) break;
          const p=projected[i];
          if (mode==='geometric' && i>0) {
            const sourceIndex=displayedSourceIndices[Math.min(i-1,displayedSourceIndices.length-1)] ?? 0;
            ctx.fillStyle=embedding[sourceIndex]>=0 ? positiveColor : negativeColor;
          } else ctx.fillStyle=pointColor;
          const r = Math.max(.4, Math.min(8, pointSize*p[3]));
          ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2*Math.PI); ctx.fill();
        }
      }

      function pointSegmentDistance(px, py, a, b) {
        const vx=b[0]-a[0], vy=b[1]-a[1], wx=px-a[0], wy=py-a[1];
        const vv=vx*vx+vy*vy;
        const t=vv ? Math.max(0, Math.min(1, (wx*vx+wy*vy)/vv)) : 0;
        return Math.hypot(px-(a[0]+t*vx), py-(a[1]+t*vy));
      }

      function deform(index) {
        const radius=deformRadius, strength=deformStrength;
        const p=pts[Math.max(0, Math.min(index, pts.length-1))];
        let push=[-p[1], p[0], 24];
        const n=Math.hypot(...push) || 1; push=push.map(v => v/n);
        for (let j=Math.max(1,index-radius); j<=Math.min(pts.length-1,index+radius); j++) {
          const w=0.5+0.5*Math.cos(Math.PI*(j-index)/radius);
          for (let k=0;k<3;k++) pts[j][k] += push[k]*strength*w;
        }
      }

      function transformShape(kind) {
        const center=[0,0,0];
        for (const p of pts) for (let k=0;k<3;k++) center[k]+=p[k]/pts.length;
        if (kind==='expand' || kind==='contract') {
          const scale=kind==='expand' ? 1.12 : 0.89;
          for (const p of pts) for (let k=0;k<3;k++) p[k]=center[k]+(p[k]-center[k])*scale;
        } else if (kind==='twist-left' || kind==='twist-right') {
          const sign=kind==='twist-right' ? 1 : -1;
          for (let i=0;i<pts.length;i++) {
            const a=sign*(i/(pts.length-1)-.5)*.65, c=Math.cos(a), s=Math.sin(a);
            const x=pts[i][0]-center[0], y=pts[i][1]-center[1];
            pts[i][0]=center[0]+c*x-s*y; pts[i][1]=center[1]+s*x+c*y;
          }
        } else if (kind==='smooth') {
          const next=pts.map(p=>p.slice());
          for (let i=1;i<pts.length-1;i++) for (let k=0;k<3;k++)
            next[i][k]=.25*pts[i-1][k]+.5*pts[i][k]+.25*pts[i+1][k];
          pts=next;
        } else if (kind==='jitter') {
          const index=1+Math.floor(Math.random()*(pts.length-2)); deform(index);
        }
      }

      function resamplePath(source, count) {
        // Evenly resample by cumulative arc length, preserving the overall path.
        count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(count)));
        const cumulative=[0];
        for (let i=1;i<source.length;i++)
          cumulative.push(cumulative[i-1]+Math.hypot(
            source[i][0]-source[i-1][0], source[i][1]-source[i-1][1], source[i][2]-source[i-1][2]));
        const total=cumulative[cumulative.length-1];
        if (!total) return Array.from({length:count},()=>source[0].slice());
        const result=[]; let segment=1;
        for (let n=0;n<count;n++) {
          const target=total*n/(count-1);
          while (segment<cumulative.length-1 && cumulative[segment]<target) segment++;
          const a=source[segment-1], b=source[segment];
          const span=cumulative[segment]-cumulative[segment-1];
          const t=span ? (target-cumulative[segment-1])/span : 0;
          result.push(a.map((v,k)=>v+(b[k]-v)*t));
        }
        return result;
      }

      function sourceIndicesForDisplay(source, count) {
        if (count===source.length)
          return Array.from({length:Math.max(0,source.length-1)},(_,i)=>Math.min(i,embedding.length-1));
        const cumulative=[0];
        for (let i=1;i<source.length;i++)
          cumulative.push(cumulative[i-1]+Math.hypot(
            source[i][0]-source[i-1][0],source[i][1]-source[i-1][1],source[i][2]-source[i-1][2]));
        const total=cumulative[cumulative.length-1];
        if (!total) return Array.from({length:count-1},(_,i)=>Math.min(i,embedding.length-1));
        const result=[]; let segment=1;
        for (let i=0;i<count-1;i++) {
          const midpoint=total*(i+.5)/(count-1);
          while (segment<cumulative.length-1 && cumulative[segment]<midpoint) segment++;
          result.push(Math.max(0,Math.min(embedding.length-1,segment-1)));
        }
        return result;
      }

      const decorativeActions=['twist-left','twist-right','contract','expand','smooth','jitter'];

      function selectedPointCount() {
        return Math.max(2,Math.min(embeddingBasePath.length,
          Math.round(+getControl('count').value||embeddingBasePath.length)));
      }

      function showBasePath() {
        const count=selectedPointCount();
        getControl('count').value=count;
        displayedSourceIndices=sourceIndicesForDisplay(embeddingBasePath,count);
        pts=count===embeddingBasePath.length
          ? embeddingBasePath.map(p=>p.slice())
          : resamplePath(embeddingBasePath,count);
      }

      function updateModeInterface() {
        const geometric=mode==='geometric';
        const trajectory=mode==='trajectory', dataMode=geometric || trajectory;
        host.querySelector('[data-section="angle"]').style.opacity=dataMode ? '.4' : '1';
        getControl('angle').disabled=dataMode;
        getAction('apply-angle').disabled=dataMode;
        getControl('count').disabled=trajectory;
        getAction('apply-count').disabled=trajectory;
        getAction('reset').disabled=trajectory;
        for (const action of decorativeActions) getAction(action).disabled=dataMode;
        host.querySelector('[data-readout="mode-description"]').textContent=trajectory
          ? 'Trajectory: real BERT states for “jaguar” · shared PCA → 3D'
          : geometric ? 'Projection: literal n-D coordinate walk → PCA → 3D'
          : 'Encoding: magnitude → length · sign → right/left';
        host.querySelector('[data-readout="angle-row"]').style.display=dataMode ? 'none' : '';
        host.querySelector('[data-readout="interaction"]').textContent=trajectory
          ? 'Click a layer point or segment for details · drag to orbit · wheel to zoom'
          : geometric ? 'Click a point or segment for its value · drag to orbit · wheel to zoom'
          : 'Click to deform · drag to orbit · wheel to zoom';
        host.querySelector('[data-section="glyph-colors"]').style.display=dataMode ? 'none' : 'flex';
        host.querySelector('[data-section="sign-colors"]').style.display=geometric ? 'flex' : 'none';
        host.querySelector('[data-section="trajectory-colors"]').style.display=trajectory ? 'flex' : 'none';
        selection.textContent=trajectory
          ? 'Blue = animal context · Orange = automobile context'
          : geometric ? 'Positive = cyan · Negative = magenta' : '';
      }

      function selectMode(nextMode) {
        mode=nextMode;
        embeddingBasePath=mode==='geometric' ? geometricBasePath
          : mode==='trajectory' ? trajectoryPaths[0] : glyphBasePath;
        getControl('count').max=embeddingBasePath.length;
        getControl('count').value=mode==='trajectory'
          ? embeddingBasePath.length
          : Math.min(768,embeddingBasePath.length);
        showBasePath(); updateModeInterface();
      }

      canvas.addEventListener('pointerdown', e => {
        dragging=true; moved=false; lastX=e.clientX; lastY=e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', e => {
        if (!dragging) return;
        const dx=e.clientX-lastX, dy=e.clientY-lastY;
        if (Math.hypot(dx,dy)>1) moved=true;
        yaw += dx*0.007; pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*0.007));
        lastX=e.clientX; lastY=e.clientY;
      });
      canvas.addEventListener('pointerup', e => {
        dragging=false;
        if (moved) return;
        const r=canvas.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
        if (mode==='trajectory') {
          let closest=null, distance=12;
          for (let pathIndex=0;pathIndex<trajectoryProjected.length;pathIndex++) {
            const path=trajectoryProjected[pathIndex];
            for (let layer=0;layer<path.length;layer++) {
              const d=Math.hypot(x-path[layer][0],y-path[layer][1]);
              if (d<distance) { distance=d; closest={pathIndex,layer}; }
            }
            for (let layer=0;layer<path.length-1;layer++) {
              const d=pointSegmentDistance(x,y,path[layer],path[layer+1]);
              if (d<distance) { distance=d; closest={pathIndex,layer:layer+1}; }
            }
          }
          if (closest) {
            const meta=trajectoryMetadata[closest.pathIndex];
            const layerLabel=closest.layer===0 ? 'Embedding output' : `Transformer layer ${closest.layer}`;
            selection.textContent=`${meta.label} · ${layerLabel} · ${meta.sentence}`;
          }
          return;
        }
        let best=-1, distance=10;
        for (let i=0;i<projected.length;i++) {
          const d=Math.hypot(x-projected[i][0], y-projected[i][1]);
          if (d<distance) { distance=d; best=i; }
        }
        if (best<0) for (let i=0;i<projected.length-1;i++) {
          const d=pointSegmentDistance(x,y,projected[i],projected[i+1]);
          if (d<distance) { distance=d; best=i+1; }
        }
        if (mode==='glyph' && best>=0) deform(best);
        else if (mode==='geometric' && best>=0) {
          const displayedSegment=Math.max(0,Math.min(displayedSourceIndices.length-1,best-1));
          const sourceIndex=displayedSourceIndices[displayedSegment] ?? 0;
          const value=embedding[sourceIndex];
          const sign=value>=0 ? 'positive' : 'negative';
          selection.textContent=`Dimension ${sourceIndex+1}: ${value.toFixed(4)} · ${sign} · |v| = ${Math.abs(value).toFixed(4)}`;
        }
      });
      canvas.addEventListener('wheel', e => {
        e.preventDefault(); zoom=Math.max(1.5,Math.min(15,zoom*Math.exp(-e.deltaY*0.001)));
      }, {passive:false});
      getAction('pause').onclick=() => {
        paused=!paused; getAction('pause').textContent=paused ? 'Play' : 'Pause';
      };
      getAction('reset').onclick=() => {
        const angle=Math.max(1,Math.min(179,+getControl('angle').value||45));
        const count=Math.max(2,Math.min(embedding.length+1,Math.round(+getControl('count').value||embedding.length+1)));
        embedding=randomEmbedding(embedding.length);
        selection.textContent='Computing a new random embedding…';
        status.textContent='Computing new glyph and PCA projection…';
        requestAnimationFrame(()=>setTimeout(()=>{
          glyphBasePath=normalizePath(embeddingPath(embedding,angle));
          geometricBasePath=normalizePath(pcaCoordinateWalk(embedding));
          embeddingBasePath=mode==='geometric' ? geometricBasePath : glyphBasePath;
          getControl('count').max=embeddingBasePath.length;
          getControl('count').value=Math.min(count,embeddingBasePath.length);
          showBasePath();
          yaw=-0.55; pitch=0.52; spin=0; zoom=5.8;
          getControl('zoom').value=zoom;
          host.querySelector('[data-readout="angle"]').textContent=Number(angle.toFixed(2));
          status.textContent='Running';
        },0));
      };
      getControl('count').max=embeddingBasePath.length;
      getControl('count').value=Math.min(768,embeddingBasePath.length);
      getAction('apply-count').onclick=() => {
        const count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(+getControl('count').value||2)));
        getControl('count').value=count;
        showBasePath();
      };
      getAction('apply-angle').onclick=() => {
        if (mode!=='glyph') return;
        const angle=Math.max(1,Math.min(179,+getControl('angle').value||45));
        const count=Math.max(2,Math.min(embeddingBasePath.length,Math.round(+getControl('count').value||embeddingBasePath.length)));
        getControl('angle').value=angle;
        host.querySelector('[data-readout="angle"]').textContent=Number(angle.toFixed(2));
        glyphBasePath=normalizePath(embeddingPath(embedding,angle));
        embeddingBasePath=glyphBasePath;
        displayedSourceIndices=sourceIndicesForDisplay(embeddingBasePath,count);
        pts=count===embeddingBasePath.length ? embeddingBasePath.map(p=>p.slice()) : resamplePath(embeddingBasePath,count);
      };
      for (const action of decorativeActions)
        getAction(action).onclick=() => { if (mode==='glyph') transformShape(action); };
      getControl('mode').onchange=e=>selectMode(e.target.value);
      getControl('speed').oninput=e=>rotationSpeed=+e.target.value;
      getControl('strength').oninput=e=>deformStrength=+e.target.value;
      getControl('radius').oninput=e=>deformRadius=+e.target.value;
      getControl('line').oninput=e=>lineWidth=+e.target.value;
      getControl('point').oninput=e=>pointSize=+e.target.value;
      getControl('zoom').oninput=e=>zoom=+e.target.value;
      getControl('line-color').oninput=e=>lineColor=e.target.value;
      getControl('point-color').oninput=e=>pointColor=e.target.value;
      getControl('positive-color').oninput=e=>positiveColor=e.target.value;
      getControl('negative-color').oninput=e=>negativeColor=e.target.value;
      getControl('animal-color').oninput=e=>trajectoryColors[0]=e.target.value;
      getControl('car-color').oninput=e=>trajectoryColors[1]=e.target.value;

      if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe(host);
      else window.addEventListener('resize', resize);
      updateModeInterface(); resize(); draw();
      status.textContent='Running';
      let previous=performance.now();
      function animate(now) {
        if (!paused) spin += (now-previous)*0.00018*rotationSpeed;
        previous=now; draw(); requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
      } catch (error) {
        const host=document.getElementById(ELEMENT_ID);
        const status=host && host.querySelector('[data-readout="status"]');
        if (status) { status.textContent='Error: '+error.message; status.style.color='#ff8b8b'; }
        throw error;
      }
    })();
    """
    js = js.replace("ELEMENT_ID", json.dumps(element_id))
    js = js.replace("GLYPH_POINT_DATA", json.dumps(glyph_points.round(7).tolist()))
    js = js.replace("GEOMETRIC_POINT_DATA", json.dumps(geometric_points.round(7).tolist()))
    js = js.replace("TRAJECTORY_POINT_DATA", json.dumps([
        points.round(7).tolist() for points in trajectory_points
    ]))
    js = js.replace("TRAJECTORY_METADATA", json.dumps(trajectory_metadata))
    js = js.replace("EMBEDDING_DATA", json.dumps(np.asarray(embedding, dtype=float).round(10).tolist()))
    # Run in a self-contained iframe. This isolates JavaScript declarations from
    # Colab and avoids both HTML script sanitization and eval_js bridge errors.
    iframe_document = f"""<!doctype html>
    <html><head><meta charset="utf-8"></head>
    <body style="margin:0;background:#060914;overflow:hidden">
      {body_html}
      <script>{js}</script>
    </body></html>"""
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        shell_name = ""

    if shell_name == "ZMQInteractiveShell" and HTML is not None and display is not None:
        # Google Colab, JupyterLab, and VS Code notebooks.
        escaped_document = html_module.escape(iframe_document, quote=True)
        display(HTML(
            f'<iframe srcdoc="{escaped_document}" '
            'style="width:100%;height:720px;border:0;border-radius:14px" '
            'sandbox="allow-scripts"></iframe>'
        ))
    else:
        # Normal Python execution, including VS Code Run/Debug.
        output_path = Path(__file__).with_name("turtle_visualization.html").resolve()
        output_path.write_text(iframe_document, encoding="utf-8")
        print(f"Visualization written to: {output_path}")
        opened = webbrowser.open(output_path.as_uri())
        if not opened:
            print("Open that HTML file in a browser to view the visualization.")


# Running this file/cell displays the example immediately.
show_turtle()
