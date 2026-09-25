# 4dRefGS: Code-Level Method and Contribution Summary

> **Purpose.** This document is a technical handoff for a paper-writing agent. It records what the current code actually implements, which parts descend from prior projects, what 4dRefGS changes, and which claims should be stated cautiously.
>
> **Audit target.** Branch `4drefgs_abuzabi`, commit `fb2d5543d` (`experiment base 0727`), inspected on 2026-07-27. Code links and line numbers refer to this revision.
>
> **Scope.** The description below prioritizes active code paths and the latest representative recipes, especially [`configs/dynerf/abuzabi100f_sumexp_shdc.yaml`](configs/dynerf/abuzabi100f_sumexp_shdc.yaml). The repository also contains many historical experiments; disabled alternatives are labeled explicitly.

## 1. Executive summary

4dRefGS reconstructs a dynamic scene containing reflective objects by combining four families of ideas:

1. a native 4D Gaussian Splatting codebase, but with an **explicit FreeTimeGS-style motion representation** in the active renderer;
2. **PGSR-style planar geometry**, plane depth, and rendered-normal/depth-normal consistency;
3. a **Ref-GS-style global reflection branch** based on deferred shading, roughness-aware Sph-Mip direction encoding, and spatial–directional outer-product factorization;
4. a **Ref-DGS-style auxiliary local-reflection Gaussian set** whose feature splats model near-field and geometry-dependent reflections without directly splatting into the main surface G-buffer.

The code-supported 4dRefGS extensions are more specific than simply “combining” these methods:

- a learned **flat-top temporal opacity window** with a constant central plateau and Gaussian shoulders;
- a **trajectory-preserving temporal bifurcation** that replaces one flat-window Gaussian with two children covering half of its plateau each;
- **sine/cosine Fourier modulation of the global reflective Gaussian feature**, rather than direct temporal modulation of RGB or SH color;
- a global/local reflection design with **separate neural outputs**, making the local contribution measurable and directly penalizable;
- a **local-output usage loss** that makes the flexible local branch expensive, encouraging ordinary glossy reflection to remain in the global branch while reserving local Gaussians for effects the global environment model cannot explain;
- a staged **coarse-to-fine optimization schedule** that establishes geometry and base appearance before enabling global reflection, local reflection, temporal feature modulation, and late temporal refinement.

The most defensible one-sentence contribution statement is:

> **4dRefGS extends explicit-motion dynamic Gaussian splatting with stable flat-window temporal support and motion-consistent temporal subdivision, then adapts global/local reflective Gaussian decomposition to time-varying scenes using Fourier-conditioned global reflection features and an explicitly regularized local reflection output.**

## 2. Attribution and implementation lineage

The repository README is still the upstream native 4DGS README, and the early history contains the official 4DGS commits. The active representation, however, is not the original coherent 4D covariance formulation. It advances an independently parameterized 3D Gaussian with an explicit linear velocity and independently controls its temporal support. Therefore, the paper should distinguish **code lineage** from **active mathematical representation**.

| Source project | What the paper introduces | What is active in 4dRefGS | What 4dRefGS changes |
|---|---|---|---|
| [Native 4DGS](https://arxiv.org/abs/2310.10642) | A coherent Gaussian in \((x,y,z,t)\), 4D rotation/covariance, temporal conditioning, and time-aware appearance | Repository scaffold, data/rasterization ancestry, temporal parameters, and parts of density control | The active renderer uses explicit position, center time, duration, and velocity instead of deriving motion from a full 4D covariance |
| [FreeTimeGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.html) | Independent position, time, duration, velocity, scale, rotation, opacity, and SH; explicit linear motion; Gaussian temporal opacity | The active dynamic parameterization and multi-frame point/velocity initialization are FreeTimeGS-like | Flat-top support replaces the Gaussian-only lifetime; deterministic temporal subdivision is added; the full FreeTimeGS relocation method is not reproduced |
| [PGSR](https://arxiv.org/abs/2406.06521) | Flattened planar Gaussians, plane normal/distance, unbiased depth, single- and multi-view geometry, and exposure compensation | Smallest-axis normal, plane-distance/depth rasterization, depth-derived normal, single-view normal consistency, scale flattening, and optional train-camera affine compensation | The constraints operate on time-conditioned dynamic Gaussians; the code also adds external monocular depth/normal priors |
| [Ref-GS](https://arxiv.org/abs/2412.00905) | Deferred reflection shading, roughness-aware Sph-Mip, and directional/spatial outer-product factorization | Global Sph-Mip environment feature, a splatted Gaussian feature, roughness, reflected-direction query, and factorized global light MLP | The splatted global reflection feature becomes time-dependent through per-Gaussian Fourier coefficients |
| [Ref-DGS](https://arxiv.org/abs/2603.07664) | Separate geometry and local-reflection Gaussian sets; global far-field and local near-field reflection features; adaptive neural fusion | An isolated auxiliary Gaussian rasterization produces a local feature buffer without writing to the main normal, depth, diffuse, or geometry buffers; gradients through shared shader inputs may still update main parameters | Both Gaussian sets are temporally supported; global and local neural outputs are separated; their fusion is explicit; the local output receives a usage penalty |

### 2.1 Important ancestry distinction

Despite names such as `rot_4d`, the active training path evaluates

\[
\mathbf{x}_i(t)=\mathbf{x}_i+\mathbf{v}_i(t-t_i),
\]

because `get_sigma_t_fixed` currently returns one for every Gaussian. This is implemented in the training forward pass and again in the renderer ([`train.py`, motion evaluation](train.py#L959), [`gaussian_renderer/__init__.py`, active motion](gaussian_renderer/__init__.py#L568), [`scene/gaussian_model.py`, fixed normalization](scene/gaussian_model.py#L1822)).

The correct wording is therefore:

> “4dRefGS is implemented on a native 4DGS-derived codebase and uses a FreeTimeGS-style explicit motion parameterization.”

Do not describe the current method as a direct implementation of either prior work. Several inherited fields remain in the model for historical experiments, while the active equations use only a subset.

## 3. End-to-end architecture

```mermaid
flowchart LR
    I[Multi-view RGB video<br/>per-frame point clouds] --> G[Global dynamic Gaussians]
    I --> P[Precomputed monocular<br/>depth and normal priors]

    G -->|x_i(t), alpha_i(t)| GR[Main planar rasterization]
    L[Local dynamic<br/>reflection Gaussians] -->|x_j(t), alpha_j(t), f_j| LR[Isolated local-feature rasterization]

    GR --> GB[G-buffer:<br/>diffuse, normal, plane depth,<br/>roughness, global feature]
    LR --> LF[Local feature map<br/>and local alpha]

    GB --> R[Per-pixel reflection direction]
    R --> S[Roughness-aware Sph-Mip query]
    S --> O[Directional feature]
    GB --> GF[Time-conditioned<br/>global Gaussian feature]
    O --> GM[Global factorized light MLP]
    GF --> GM
    O --> LM[Local light MLP]
    LF --> LM
    GB --> LM

    GM --> F[Explicit global/local fusion]
    LM --> F
    GB --> C[Diffuse radiance]
    F --> C
    C --> OUT[Linear-to-sRGB image]

    OUT --> PL[Photometric/perceptual losses]
    P --> GL[Geometry-prior losses]
    GB --> GL
    LM --> UL[Local-output usage loss]
    PL --> OPT[Joint optimization and density control]
    GL --> OPT
    UL --> OPT
```

At a given timestamp, the main Gaussian set is responsible for surface geometry, diffuse appearance, roughness, normals, and the global reflection feature. The local Gaussian set is rasterized separately into a feature map and cannot overwrite the main normal, depth, diffuse color, or surface geometry ([isolated local pass](gaussian_renderer/__init__.py#L756)). Its decoder still consumes shared/main geometry-dependent inputs, so local-shading gradients can indirectly update those parameters unless detached.

## 4. Active Gaussian representation

### 4.1 Per-Gaussian parameters

The main model allocates the following fields in [`scene/gaussian_model.py`](scene/gaussian_model.py#L617):

| Code field | Shape per Gaussian | Active role |
|---|---:|---|
| `_xyz` | 3 | Reference position \(\mathbf{x}_i\) |
| `_t` | 1 | Reference/center time \(t_i\) |
| `_velocity` | 3 | Linear velocity \(\mathbf{v}_i\) |
| `_scaling` | 3 | Anisotropic spatial scale |
| `_rotation` | 4 | Spatial quaternion |
| `_opacity` | 1 | Raw base opacity, passed through a sigmoid |
| `_scaling_t` | 1 | Temporal edge parameter |
| `_velocity2[...,0]` | 1 | Learned flat-window half-width \(r_i\) in the active flat-window mode |
| `_features_dc`, `_features_rest` | SH-dependent | Base/diffuse SH appearance |
| `_specular` | 4 | Static global or local reflection feature after `tanh` |
| `_specular2` | 44 | Storage for temporal global-feature coefficients; the active renderer uses the final 40 values as \(4\times10\) coefficients |
| `_albedo` | 3 | Positive learned diffuse albedo option |
| `_roughness` | 1 | Sigmoid-activated roughness |
| `_delta_normal` | 3 | Learned normal correction field, currently regularized but not used by the active reflection-direction branch |

The code retains `_velocity2[...,1:3]`, `_velocity3`, and `_rot_velocity` from earlier experiments. They should not be presented as active higher-order or rotational motion in the current method.

### 4.2 Time-conditioned position and opacity

For view time \(t\), the global and local positions are

\[
\mathbf{x}_i(t)=\mathbf{x}_i+\mathbf{v}_i(t-t_i).
\]

The effective opacity is

\[
\alpha_i(t)=\sigma(o_i)\,m_i(t),
\]

where \(o_i\) is the raw opacity and \(m_i(t)\) is the selected temporal opacity factor. Gaussians with \(m_i(t)\leq0.05\) are masked out of the active render ([global and local time gating](train.py#L978)).

### 4.3 Multi-frame initialization

`create_from_multi_pcd` loads per-frame point clouds from `pcds_j10` and a static point cloud from `points3d.ply` ([initialization entry point](scene/gaussian_model.py#L2257)).

For dynamic points:

- the point timestamp initializes \(t_i\), and points in the first loaded frame receive zero initial velocity;
- for later frames, velocity is initialized from the displacement relative to the mean of the five nearest points in the preceding loaded point cloud, multiplied by FPS;
- the flat half-width starts at half a frame, \(r_i=0.5/\mathrm{FPS}\);
- the Gaussian shoulder is initialized so that the temporal factor reaches approximately \(0.05\) half a frame beyond the flat region ([dynamic initialization](scene/gaussian_model.py#L2360)).

For static points:

- \(t_i\) is initialized to the center of the configured clip;
- velocity is zero;
- the flat half-width covers the whole clip plus half a frame of margin ([static initialization](scene/gaussian_model.py#L2450)).

This initialization supports a useful structural interpretation: dynamic points begin with narrow temporal support, while points from the static `points3d.ply` cloud begin as long-lived Gaussians.

## 5. PGSR-derived geometry in a dynamic setting

### 5.1 Planar normal and plane depth

The Gaussian normal is the rotated axis associated with the smallest spatial scale, flipped to face the camera ([`get_smallest_axis` and `get_normal`](gaussian_renderer/__init__.py#L1237)). The renderer then computes:

- a camera-space normal;
- the signed camera-to-Gaussian-plane distance;
- a plane depth returned by the planar rasterizer;
- a normal reconstructed from the plane-depth image ([depth-to-normal conversion](gaussian_renderer/__init__.py#L1222)).

The main rasterization writes normal, alpha, plane distance, roughness, world normal, reflection feature, albedo, and diffuse color into a 32-channel buffer ([G-buffer packing](gaussian_renderer/__init__.py#L686)).

### 5.2 Geometry losses actually present

The active training code includes:

1. **Planarization:** after iteration 1,000, the smallest visible spatial scale is penalized with weight \(100\), flattening Gaussians toward local planes ([`train.py`](train.py#L1399)).
2. **Single-view rendered normal/depth-normal consistency:** after iteration 3,000,

   \[
   \mathcal{L}_{\mathrm{nd}}
   =
   0.03\,\mathbb{E}_{p}
   \left[1-\hat{\mathbf n}_{\mathrm{render}}(p)
   \cdot\hat{\mathbf n}_{\mathrm{depth}}(p)\right]
   \]

   ([`train.py`](train.py#L1420)).
3. **Alpha completeness:** \(0.1\,\mathbb{E}[1-\alpha]\) after the same phase.
4. **Normal-delta regularization:** \(0.05\,\operatorname{mean}_{i,c}(\Delta n_{i,c}^2)\), equivalently \(0.05/3\,\mathbb E_i\lVert\Delta\mathbf n_i\rVert_2^2\).
5. **Direct supplied-normal path:** an `n_gt` L1 term exists, but the current `CameraDataset` always returns `None` for this item, so it is inactive in representative runs.

This is a **subset and adaptation** of PGSR, not the full method. The current training loop does not implement PGSR’s multi-view homography, patch-NCC photometric consistency, or multi-view geometric consistency.

### 5.3 Additional monocular priors

4dRefGS also loads precomputed `.npy` priors from sibling `sgt_depth` and `sgt_normal` directories ([`utils/data_utils.py`](utils/data_utils.py#L47)). The provided preprocessing script uses Depth Anything V2 for depth and a Marigold-based normal pipeline ([`create_depth_normal_abuzabi.py`](create_depth_normal_abuzabi.py#L32)).

The supplied script is not plug-compatible with the loader path as written: it outputs model-side `predicted_depth_npy` and `predicted_normal_npy`, whereas training expects dataset-side `sgt_depth` and `sgt_normal`. A copy/rename staging step is required but is not implemented in that script.

The depth prior is median-centered and divided by its mean absolute deviation. The rendered plane depth is normalized independently, and the loss uses their sum; this implies an assumed opposite-polarity convention between the stored prior and rendered plane depth, but the loader does not validate that convention:

\[
\mathcal{L}_{\mathrm{mono-depth}}
=
0.01\,\mathbb{E}\left[
\left|
\widetilde D_{\mathrm{prior}}
+
\widetilde D_{\mathrm{render}}
\right|
\right],
\]

active after iteration 1,500 until the configured cutoff ([`train.py`](train.py#L1154)).

The predicted normal is sign-flipped, normalized, and compared by cosine distance to the rendered normal. Its weight is \(0.02\) from iterations 500–1,500 and \(0.05\) thereafter until its cutoff ([`train.py`](train.py#L1187)).

These monocular priors are **additional to PGSR**. PGSR’s core single-view depth-normal constraint is self-derived and does not require a pretrained depth or normal model.

### 5.4 PGSR-style camera compensation

When enabled, each training camera receives six learnable affine color parameters:

\[
\mathbf I'_c=\exp(\mathbf a_c)\odot\mathbf I+\mathbf b_c.
\]

Only the L1 term sees the adjusted image; SSIM and test rendering use the unadjusted result. The affine transform begins after a configured iteration and is used only if raw-image SSIM error passes the configured gate ([setup](train.py#L647), [loss application](train.py#L1125)). The latest representative recipe enables this from iteration 1,000 with an SSIM-loss gate of \(0.5\).

## 6. Reflective appearance model

### 6.1 Global branch: Ref-GS-style directional factorization

The global branch follows the Ref-GS pattern:

1. splat a four-dimensional per-Gaussian feature into the G-buffer;
2. compute a per-pixel reflection direction from the rendered surface normal;
3. query a learnable spherical feature map using the reflection direction and roughness-dependent mip level;
4. form the outer product between the 16-D directional feature and the 4-D splatted global feature;
5. decode the directional feature and \(16\times4\) outer product with a small MLP.

The Sph-Mip encoding uses a \(512\times1024\) base map and nine mip levels in the current implementation ([light initialization](scene/gaussian_model.py#L716)). The global MLP has two 256-unit hidden layers and outputs three log-radiance channels.

Let \(\mathbf e(p)\in\mathbb R^{16}\) be the roughness-aware Sph-Mip feature and \(\mathbf f_g(p,t)\in\mathbb R^4\) the normalized splatted Gaussian feature. Its active input is

\[
\left[\mathbf e(p),\;
\operatorname{vec}\!\left(\mathbf e(p)\otimes\mathbf f_g(p,t)\right)\right].
\]

The construction occurs in [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py#L954).

### 6.2 Local branch: Ref-DGS-style auxiliary feature geometry

The local branch is a second time-conditioned Gaussian set. At initialization it clones the global Gaussian tensors, then receives its own optimizer. It can be created at the start or lazily at `local_feature_start_iter` ([local model cloning](train.py#L165), [lazy spawn](train.py#L863)).

To avoid a normalized nonzero feature covering the entire scene at birth, the current spawn path can:

- zero the local `_specular` feature;
- reset local opacity to a small value, typically \(0.1\).

The renderer uses the local set only for:

- time-conditioned position;
- temporal opacity;
- spatial scale and rotation;
- base opacity;
- the four-dimensional `tanh(_specular)` feature.

It performs a black-background, isolated rasterization and extracts a local feature map and alpha map ([local pass](gaussian_renderer/__init__.py#L756)). The local Gaussians do not directly write main geometry, normal, plane depth, roughness, diffuse color, or the global feature. Their shading loss can nevertheless update shared or main geometry-dependent inputs through roughness and the normal/reflection cosine unless those paths are detached.

The local MLP receives:

\[
\left[\mathbf e(p),\;\mathbf f_l(p,t),\;\rho(p),\;c(p)\right],
\]

where \(\mathbf f_l\) is the normalized local feature, \(\rho\) is global roughness, and the active code computes

\[
c(p)=\operatorname{clip}\!\left(\mathbf n(p)\cdot\mathbf r(p),-1,1\right).
\]

The MLP has three 64-unit hidden layers and three outputs ([definition](scene/gaussian_model.py#L839), [input construction](gaussian_renderer/__init__.py#L1021)).

The local feature map is only one input to this decoder. Direction encoding, main roughness, the normal/reflection cosine, and decoder biases can produce a nonzero local-head output even when the auxiliary feature is zero; consequently, the usage loss regularizes the complete local decoder head, not exclusively the auxiliary Gaussian feature field.

Conceptually, this branch captures reflection effects whose apparent geometry differs from the physical surface—near-field inter-reflection, self-reflection, and mirror-induced virtual images. The paper should prefer this standard terminology over “middle-range reflection.”

### 6.3 Explicit global/local fusion

Unlike Ref-DGS’s single joint adaptive shader, the active 4dRefGS path exposes a global log-light output \(\mathbf g\) and a local log-light output \(\boldsymbol\ell\). It supports two fusion modes ([fusion code](gaussian_renderer/__init__.py#L1085)):

**Additive positive radiance (`sum_exp`):**

\[
\mathbf L_{\mathrm{spec}}
=
\exp(\min(\mathbf g,5))+\exp(\min(\boldsymbol\ell,5)).
\]

**Log-space local correction (`exp_sum`):**

\[
\mathbf L_{\mathrm{spec}}
=
\exp(\min(\mathbf g+\boldsymbol\ell,5)).
\]

The final image is

\[
\mathbf I_{\mathrm{sRGB}}
=
\operatorname{linear2sRGB}
\left(\mathbf L_{\mathrm{diffuse}}+\mathbf L_{\mathrm{spec}}\right).
\]

The latest `sum_exp` recipes zero-initialize the local MLP’s final weights and set its output bias to \(-5\), so the additive local radiance begins near \(\exp(-5)\). In `exp_sum`, zero bias instead gives a neutral multiplicative factor of one ([initialization](scene/gaussian_model.py#L850)).

## 7. Project-specific modification 1: flat-top temporal opacity

### 7.1 Definition

Native 4DGS and FreeTimeGS use a Gaussian-shaped temporal marginal. 4dRefGS adds the `flat_window` mode:

\[
m_i(t)=
\exp\!\left[
-\frac12
\left(
\frac{\max(|t-t_i|-r_i,0)}
{\sigma_{e,i}}
\right)^2
\right],
\]

where:

- \(t_i\) is the learned temporal center;
- \(r_i=\max(\texttt{\_velocity2}_{i,0},0)\) is a learned flat half-width;
- \(\sigma_{e,i}=\exp(2\,\texttt{\_scaling\_t}_i)\) is the Gaussian shoulder width used by the code.

This is implemented in [`get_temporal_opacity_factor`](scene/gaussian_model.py#L1958).

For \(|t-t_i|\le r_i\), the numerator is zero and \(m_i(t)=1\). Outside the plateau, only the excess distance beyond the plateau is attenuated.

### 7.2 Intended optimization effect

The code guarantee is narrower than complete screen-space stability. It gives each Gaussian:

- a central interval in which temporal opacity weight is exactly one, removing the temporal gate itself as a source of feature attenuation;
- smooth differentiable entry and exit through Gaussian shoulders;
- a temporal interval that can be subdivided deterministically.

This is particularly useful for deferred reflective shading, because the global or local feature is not continuously suppressed near the center of its lifetime before screen-space normalization.

### 7.3 Temporal-window regularization

The active getter learns \(r_i\) directly from raw `_velocity2[...,0]`. Retained keys such as `temporal_flat_radius_mult`, `temporal_flat_edge_sigma_mult`, the flat-radius ramp fields, and `temporal_schedule_iteration` are legacy compatibility/experiment fields and do not alter the current flat-window equation. The effective active constraints are the support minimum, radius cap, shoulder penalty, and center-time bounds below.

The training loop computes the support radius at temporal factor \(0.05\):

\[
R_i(0.05)
=
r_i+\sigma_{e,i}\sqrt{-2\log 0.05}.
\]

It then:

- penalizes supports shorter than a configured fraction of a frame;
- caps the high-opacity flat radius by a configured number of frames;
- penalizes overly broad shoulders;
- clamps learned temporal centers to the scene’s configured time interval;
- applies the same rules to the local set when it also uses flat windows.

See [`train.py`](train.py#L1317) and [`get_temporal_range_for_opacity`](scene/gaussian_model.py#L2040).

### 7.4 Precise novelty wording

Safe:

> “We replace the Gaussian-only temporal opacity profile with a learned flat-top window that maintains unit temporal weight over a central interval and decays through differentiable Gaussian shoulders.”

Unsafe:

> “We introduce temporal opacity for Gaussian splatting.”

Temporal opacity already exists in native 4DGS and FreeTimeGS.

## 8. Project-specific modification 2: motion-consistent temporal bifurcation

### 8.1 Candidate selection

Within the configured temporal-split window, a Gaussian may be selected by the union of:

- the default top 1% of positive accumulated \(|\partial\mathcal L/\partial t_i|\), using a 0.99 population quantile optionally bounded by an absolute floor; this path runs only when more than 1,000 positive values exist;
- a thresholded, maximum-accumulated reflection-gradient proxy formed from `_specular` and `_specular2[:,4:]`; this is a parameter-gradient statistic, not a direct temporal derivative.

Candidates are additionally required to:

- have at least half a frame of flat radius;
- have more than half a frame of effective temporal support at factor \(0.05\);
- lie inside the configured environment sphere.

The selection and gates are in [`densify_and_split_time`](scene/gaussian_model.py#L3083). Absolute per-view time gradients are accumulated to avoid cancellation across different timestamps ([gradient accumulation](scene/gaussian_model.py#L3643), [batched accumulation](train.py#L1544)).

This dual criterion matters for reflective dynamic scenes: temporal subdivision can be triggered either by mis-modeled motion/diffuse content or by temporally under-resolved reflective appearance.

### 8.2 Two-child split

For the standard \(N=2\) flat-window split, parent \((t_i,r_i)\) becomes:

\[
t_{i,1}=t_i-\frac{r_i}{2},
\qquad
t_{i,2}=t_i+\frac{r_i}{2},
\]

\[
r_{i,1}=r_{i,2}=\frac{r_i}{2}.
\]

The shoulder width \(\sigma_{e,i}\) is copied unchanged. Therefore, the two child plateaus tile the parent plateau:

\[
[t_i-r_i,t_i]
\cup
[t_i,t_i+r_i].
\]

The implementation is in [`get_temporal_split_params`](scene/gaussian_model.py#L2052) and [`densify_and_split_time`](scene/gaussian_model.py#L3173).

### 8.3 Trajectory preservation

Changing the reference time without changing the reference position would shift the motion trajectory. The child anchors are therefore moved along the parent trajectory:

\[
\mathbf x_{i,k}
=
\mathbf x_i+\mathbf v_i(t_{i,k}-t_i).
\]

Each child then keeps the parent velocity, so for any evaluation time \(t\):

\[
\mathbf x_{i,k}+\mathbf v_i(t-t_{i,k})
=
\mathbf x_i+\mathbf v_i(t-t_i).
\]

Thus the split initially changes temporal capacity without changing the represented linear motion. This equality preserves the motion centerline, not the complete rendered signal: both children copy the full parent opacity and their temporal shoulders overlap, so post-split composited alpha is not analytically constrained to equal the parent alpha.

### 8.4 State inheritance and later specialization

Children copy spatial scale, rotation, opacity, SH appearance, global/local reflection feature, Fourier coefficients, albedo, roughness, and normal parameters. They also inherit the parent’s Adam first and second moments to avoid a large optimizer transient ([optimizer-state inheritance](scene/gaussian_model.py#L2909), [split append/prune](scene/gaussian_model.py#L3221)). The parent is then removed.

Because the child parameters are independent after the split, they can specialize their appearance and reflection over their shorter lifetimes. This is the code mechanism behind the intended use of time splitting for larger or longer-term lighting changes.

### 8.5 Precise novelty wording

Safe:

> “We propose deterministic, trajectory-preserving temporal bifurcation: an under-resolved flat-window Gaussian is replaced by two half-plateau children whose anchors remain on the parent motion path.”

Avoid claiming that temporal densification itself is unprecedented. Native 4DGS already performs joint spatial-and-temporal sampling during Gaussian splitting; the distinctive element here is the deterministic flat-window, half-plateau, trajectory-preserving construction.

Also note that “half lifespan” refers exactly to the **flat plateau**. Since each child retains the parent shoulder width, its complete low-opacity support is not exactly half of the parent’s complete support.

## 9. Project-specific modification 3: Fourier-conditioned global reflection feature

### 9.1 Definition

Each global Gaussian carries a base feature \(\mathbf f_i^0\in\mathbb R^4\) and 40 active coefficient values arranged as \(\mathbf C_i\in\mathbb R^{4\times10}\). The ten temporal basis values are:

\[
\boldsymbol\gamma(t)=
[\cos(4\pi2^0t),\ldots,\cos(4\pi2^4t),\sin(4\pi2^0t),\ldots,\sin(4\pi2^4t)]^{\!\top}\in\mathbb R^{10}.
\]

The per-Gaussian feature rasterized by the global branch is:

\[
\mathbf f_i(t)
=
\tanh(\mathbf z_i)
+
2\,\tanh(\mathbf Z_i)\,\boldsymbol\gamma(t),
\]

where \(\mathbf z_i\) is `_specular` and \(\mathbf Z_i\) is the final 40 values of `_specular2`. The active code is in [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py#L701).

The basis uses the absolute scene timestamp, not \(t-t_i\) or clip-normalized time. With timestamps measured in seconds, the five bands correspond to 2, 4, 8, 16, and 32 cycles per second; changing timestamp units changes their physical frequencies.

After alpha compositing, the four-dimensional screen-space feature is L2-normalized before it enters the global factorization ([feature normalization](gaussian_renderer/__init__.py#L901)). The modulation therefore primarily changes the feature’s direction/material code rather than merely scaling radiance.

### 9.2 Why this is not native 4DGS 4DSH

Native 4DGS already uses temporal Fourier bases in its time-varying appearance formulation. The 4dRefGS distinction is the target of the modulation:

| Native 4DGS appearance | 4dRefGS reflection feature |
|---|---|
| Temporal basis participates in direct view-dependent color/4D SH modeling | Temporal sine/cosine coefficients modulate a four-dimensional Gaussian reflection/material feature |
| Appearance is decoded as radiance/color | The modulated feature is splatted, normalized, outer-product-factorized with Sph-Mip direction features, then decoded as global specular light |
| Paper formulation emphasizes cosine temporal bases | Active 4dRefGS code uses paired cosine and sine bands |

The paper should call this a **time-varying global reflection feature**, not a time-varying global environment map.

### 9.3 Coarse-to-fine activation

The code can activate the five frequency bands low-to-high with a raised-cosine window:

\[
w_k(\alpha)
=
\frac12
\left[
1-\cos\left(
\pi\,\operatorname{clip}(\alpha-k,0,1)
\right)
\right].
\]

This option is used only when `fourier_c2f_end_iter > fourier_c2f_start_iter` ([band annealing](gaussian_renderer/__init__.py#L714)). In the latest representative recipe, `fourier_c2f_start_iter=30000` and `fourier_c2f_end_iter=-1`; consequently, all bands are enabled together at iteration 30,000. A paper claiming low-to-high frequency annealing must use and report a configuration with a valid end iteration.

### 9.4 Short- and long-timescale division of labor

The intended interpretation is:

- Fourier coefficients let an existing Gaussian’s global reflection code vary smoothly over short time intervals without duplicating geometry;
- temporal split gives separately optimizable, temporally localized children, supporting larger, piecewise, or non-periodic changes.

This is a useful design rationale, not a theoretical guarantee. It should be validated by the Fourier and split ablations.

The analogous local-feature Fourier update is commented out in the current renderer. Only the global reflection feature receives this temporal basis.

## 10. Project-specific modification 4: regularized global/local reflection

### 10.1 What is inherited and what is new

The existence of complementary global and local reflection features is central to Ref-DGS and should be credited accordingly. The 4dRefGS-specific extension is:

- both Gaussian sets participate in the dynamic temporal model;
- the global feature is Fourier-conditioned;
- the global and local neural outputs remain separate until the radiance-combination step;
- the local decoder head receives a mode-appropriate output-usage penalty.

### 10.2 Local-output usage loss

For `sum_exp`, the measurable local contribution is its actual positive additive radiance:

\[
\mathbf U_l(p)=\exp(\min(\boldsymbol\ell(p),5)),
\qquad
\mathcal L_{\mathrm{local-use}}
=
\lambda_l\,\mathbb E_p[\mathbf U_l(p)].
\]

For `exp_sum`, the local branch is a log-space correction and zero is neutral:

\[
\mathbf U_l(p)=|\boldsymbol\ell(p)|,
\qquad
\mathcal L_{\mathrm{local-use}}
=
\lambda_l\,\mathbb E_p[\mathbf U_l(p)].
\]

The renderer exposes `local_light_usage` according to the selected fusion mode ([usage definition](gaussian_renderer/__init__.py#L1087)), and training adds its mean after the configured start iteration ([loss](train.py#L1498)).

This differs from penalizing local Gaussian opacity. Opacity sparsity can reduce the number or weight of local splats, but the normalized local feature may still drive the shader. Under `sum_exp`, the loss prices actual additive local radiance. Under `exp_sum`, it prices the magnitude of the local log-space modulation, a proxy for deviation from the neutral multiplier rather than a separately observable radiance term.

### 10.3 Optimization interpretation

The local branch has enough spatial freedom to fit both:

- desired mirror-like virtual images and geometry-dependent near-field reflection;
- undesired glossy effects that the smoother global environment branch could explain.

The local-output penalty introduces an economy principle:

> use global reflection by default; pay for local reflection only when its photometric improvement offsets the usage cost.

This motivates the empirical hypothesis that the loss reduces local-branch overuse on glossy surfaces; the zero-weight ablation must demonstrate that effect.

The latest representative `sum_exp` recipe uses:

- `local_light_mlp_zero_init: True`;
- `local_mlp_output_loss_weight: 0.05`;
- `local_mlp_output_loss_from_iter: 15000`;
- `local_opacity_loss_weight: 0.0`.

Thus its active preference is enforced at the **neural output**, not through a local-opacity penalty.

## 11. Project-specific modification 5: staged coarse-to-fine training

The code exposes a collection of phase boundaries rather than one monolithic “coarse-to-fine loss.” In the latest 60k-iteration recipe, the sequence is:

| Iteration range / event | Newly active behavior | Purpose |
|---:|---|---|
| 0 onward | Dynamic geometry, SH/base color, photometric reconstruction | Establish coarse scene occupancy and motion |
| 500 onward | Monocular normal prior | Begin geometry guidance after a short warm-up |
| 1,000 onward | Planar scale penalty; optional camera affine compensation | Flatten geometry and absorb train-camera exposure variation |
| 1,500 onward | Monocular depth prior | Add scale/shift-invariant depth structure |
| 3,000 onward | Rendered normal vs. depth-normal consistency; alpha completeness; effective-opacity entropy | PGSR-style surface consolidation |
| through 5,000 | LPIPS term | Early perceptual guidance |
| 9,000 | Global Sph-Mip reflection enabled; albedo optionally initialized from SH | Add global reflective appearance after coarse geometry/color |
| 12,000 | Local reflection pass and local MLP enabled | Add the more flexible near-field branch later |
| 15,000 | Local-output usage penalty enabled | Let the global branch establish itself, then constrain local usage |
| 20,000 | Monocular normal prior stops | Remove the external orientation prior after geometry formation |
| 30,000 | The early spatial-only phase ends; Fourier modulation and alternating spatial/temporal refinement begin | Add time-varying reflection and temporally localized capacity after the global/local solution is established |
| 30,000–50,000 | Every second density-control event performs temporal bifurcation; intervening events retain spatial clone/split | Alternate temporal specialization with continued spatial refinement |
| 40,000 | Monocular depth prior stops | Remove the external depth prior during late appearance/refinement optimization |
| 55,000 | Batch size changes from 1 to 2 | Reduce late gradient noise across timestamps |
| 60,000 | End | Final model |

The relevant phase controls are declared in [`arguments/__init__.py`](arguments/__init__.py#L193), applied in [`train.py`](train.py#L831), and instantiated by the representative config ([phase settings](configs/dynerf/abuzabi100f_sumexp_shdc.yaml#L94)).

This schedule is best presented as a practical optimization contribution for the **combined dynamic-reflective problem**. Avoid a broad “first coarse-to-fine Gaussian training” claim.

## 12. Full training objective

The implementation does not optimize one named objective. It assembles a phase-dependent collection of terms:

\[
\mathcal{L}=
\mathcal{L}_{\mathrm{image}}
+\lambda_{\mathrm{LPIPS}}\mathcal{L}_{\mathrm{LPIPS}}
+\mathcal{L}_{\mathrm{mono-depth}}
+\mathcal{L}_{\mathrm{mono-normal}}
+\mathcal{L}_{\mathrm{normal-depth}}
+\mathcal{L}_{\mathrm{temporal}}
+\mathcal{L}_{\mathrm{planar}}
+\mathcal{L}_{\mathrm{branch}}
+\mathcal{L}_{\mathrm{opacity-entropy}}
+\mathcal{L}_{\mathrm{camera}}.
\]

The weights and even the presence of these terms depend on the iteration and configuration. The active implementation is summarized below.

| Term | Active implementation | Function in the combined method |
|---|---|---|
| Image reconstruction | \((1-0.2)L_1+0.2(1-\mathrm{SSIM})\). The L1 term can use a train-camera affine-corrected render, while SSIM always uses the uncorrected render. | Main reconstruction signal without allowing exposure compensation to inflate the reported/test image |
| LPIPS | Weight 0.01 through iteration 5,000 | Early perceptual stabilization |
| Monocular depth | Predicted and rendered depth are each median-centered and divided by their mean absolute deviation; the loss is \(0.01\lvert d_{\mathrm{pred}}+d_{\mathrm{render}}\rvert\) after iteration 1,500 and before the configured cutoff | Scale/shift-insensitive coarse geometry guidance; the plus sign implies an assumed opposite-polarity convention that the loader does not validate |
| Monocular normal | Mean \(1-\hat{\mathbf n}_{\mathrm{render}}\cdot\hat{\mathbf n}_{\mathrm{prior}}\), weighted 0.02 from 500–1,500 and 0.05 afterward until the configured cutoff | External orientation guidance |
| Render-normal/depth-normal consistency | \(0.03\,\mathbb E[1-\hat{\mathbf n}_{\mathrm{render}}\cdot\hat{\mathbf n}_{\mathrm{depth}}]\) after iteration 3,000 | PGSR-style single-view surface consistency |
| Alpha completeness | \(0.1\,\mathbb E[1-A]\) after iteration 3,000 | Encourages a coherent visible surface |
| Effective-opacity entropy | \(0.01\,\operatorname{mean}[-\alpha\log\alpha-(1-\alpha)\log(1-\alpha)]\) over visible temporally weighted global opacities after iteration 3,000 | Encourages effective opacity toward binary values |
| Normal-residual magnitude | \(0.05\,\operatorname{mean}_{i,c}(\Delta n_{i,c}^2)\), equivalently \(0.05/3\,\mathbb E_i\lVert\Delta\mathbf n_i\rVert_2^2\), after iteration 3,000 | Stops the optional residual normal from becoming unconstrained |
| Minimum temporal support | \(0.01\max(R_{\min}-R_i,0)\) | Prevents dynamic primitives from collapsing to negligibly short support |
| Flat-radius cap | \(\max(r_i-R_{\max},0)\) | Prevents one dynamic Gaussian from remaining fully opaque over too much of the clip |
| Shoulder-width penalty | \(0.05\max(R_{i,\mathrm{edge}}-R_{\min},0)\) | Keeps Gaussian temporal transitions localized |
| Time-center bounds | Unit-weight hinge losses outside the configured clip interval | Keeps primitive centers assigned to observed time |
| Planarity | 100 times the mean smallest spatial scale after iteration 1,000 | Pushes ellipsoids toward planar surfels |
| Global opacity | Mean visible global opacity, normally weight 0.02 during the pre-30k densification phase | Counterbalances excessive opaque primitives during geometry formation |
| Local opacity | Mean visible temporally weighted local opacity | Optional local sparsity control; weight 0 in the representative recipe |
| Local neural output | Mean clamped positive local-head radiance for `sum_exp`, or mean absolute local log modulation for `exp_sum` | Regularizes the complete local decoder output, not only the auxiliary feature map |
| Camera affine | Per-training-camera diagonal gain \(\exp(\mathbf a)\) and RGB bias, activated after iteration 1,000 when SSIM error is below 0.5 | Absorbs train-camera photometric mismatch as a nuisance parameter |

The image and prior losses are assembled in [`train.py`](train.py#L1125), temporal support losses in [`train.py`](train.py#L1317), planarity in [`train.py`](train.py#L1399), PGSR-style single-view consistency in [`train.py`](train.py#L1416), branch/entropy regularization in [`train.py`](train.py#L1489).

Two data-path details should be documented in an eventual paper implementation section:

- monocular priors are precomputed rather than predicted during optimization; [`utils/data_utils.py`](utils/data_utils.py) loads sibling `sgt_depth` and `sgt_normal` directories, while [`create_depth_normal_abuzabi.py`](create_depth_normal_abuzabi.py) writes differently named model-side directories, so an undocumented copy/rename staging step is currently required;
- the affine correction is used only in the training L1 path. Test rendering and the SSIM component remain uncorrected, so the affine variables are not part of the deployable scene appearance.

## 13. Density control and parameter refinement

Density control is shared by the spatial, temporal, global-reflection, and local-reflection parts of the system. The representative run uses the following policy:

1. **Statistics.** Each backward pass accumulates screen-space position gradients. In multi-view batches, visibility-normalized absolute time gradients are accumulated as well. During the temporal-split phase, reflection-feature time-gradient statistics are collected on selected intervals.
2. **Early spatial growth.** From iteration 500 through 30,000, the global set uses the inherited clone/split/prune machinery with planar rasterization gradients. The densification interval is itself scheduled rather than fixed.
3. **Late temporal refinement.** From 30,000 through 50,000, every second densification event performs temporal bifurcation. The intervening event retains spatial clone/split behavior; this measured alternation is deliberate rather than accidental.
4. **Reflection-aware selection.** A temporal candidate can arise from large loss gradient with respect to center time or from the global reflection feature’s temporal gradient proxy. It must also pass the plateau-width, temporal-support, and environment-volume gates described in Section 8.
5. **Local-set refinement.** Local Gaussians start participating only after the local-feature phase begins. They use half the global spatial gradient threshold, share the temporal-split cadence in flat-window mode, and use a gentler randomized prune rule so a normalized local feature buffer does not disappear abruptly.
6. **Opacity management.** The representative recipe resets global opacity to 0.5 at 3k, 6k, and 9k, continues high-opacity reset behavior through 25k, and stops ordinary spatial densification at 30k.
7. **Optimizer surgery.** Pruning preserves retained Adam state. Ordinary spatial children receive zero-initialized moments, whereas temporal-bifurcation children explicitly inherit their parent’s first and second moments. The main optimizer (including lighting groups) and local-Gaussian optimizer are stepped once per iteration.

The orchestration is in [`train.py`](train.py#L1722); the Gaussian-side clone, split, pruning, and optimizer-state operations are in [`scene/gaussian_model.py`](scene/gaussian_model.py#L2909). A paper should describe this as **alternating spatial and temporal capacity refinement**, not merely “densification.”

## 14. Exact forward-pass trace

For one training view at timestamp \(t\), the active path is:

1. Load RGB, timestamp, external depth, and external normal for the selected camera.
2. Set the current timestamp on both Gaussian models.
3. Move every main and local primitive to \(\mathbf x_i(t)=\mathbf x_i+\mathbf v_i(t-t_i)\).
4. Evaluate the flat-top temporal factor and discard primitives below the 0.05 temporal threshold.
5. Derive each main Gaussian normal from the axis with the smallest spatial scale and orient it toward the camera.
6. Construct the main G-buffer channels: diffuse/base feature, plane normal and distance, opacity/depth quantities, roughness, global reflection feature, and optional normal residual.
7. If the Fourier phase is active, augment the static four-dimensional global feature with its sine/cosine-conditioned term before rasterization.
8. Rasterize the main Gaussian set once with the planar anti-aliased PGSR-derived rasterizer.
9. Rasterize the local set in a separate black-background pass containing only local feature and local alpha. It does not directly write main depth, normal, diffuse color, or visibility, although local-shading gradients can update shared/main geometry-dependent inputs.
10. L2-normalize the alpha-composited global and local feature vectors; the active code does not divide them by rendered alpha.
11. Form a per-pixel reflected view direction from the main surface normal and view direction.
12. Query the roughness-aware 16-dimensional Sph-Mip encoding at that reflected direction.
13. Form the global 80-dimensional factorized code by flattening the outer product of the 16-dimensional direction code and four-dimensional global Gaussian feature, then concatenating the direction code itself.
14. Decode global log-radiance with the 80→256→256→3 MLP.
15. Concatenate the direction code, four-dimensional local feature, roughness, and clamped normal/reflection cosine into the 22-dimensional local input, then decode local log-radiance with the 22→64→64→64→3 MLP.
16. Combine global and local outputs using `sum_exp` or `exp_sum`, add diffuse linear radiance, and convert the result to sRGB.
17. Accumulate the image, geometry, temporal-support, branch-usage, and optional camera-affine losses.
18. Backpropagate, aggregate multi-view statistics if needed, run the scheduled density-control event, and step the main optimizer (which owns the active lighting-module parameter groups), the local-Gaussian optimizer, and the optional camera-affine optimizer.

The complete renderer is [`render_3d_pgsr_anti`](gaussian_renderer/__init__.py#L474). The local Gaussian model supplies the splatted local feature, while both `light_mlp` and `light_mlp_2` are separate decoder MLPs owned and checkpointed by the **main** Gaussian model. The architecture is therefore two Gaussian sets and two decoder MLPs with shared main-model ownership, not two independent neural renderers.

## 15. Representative recipe and existing ablations

The most representative configuration is [`configs/dynerf/abuzabi100f_sumexp_shdc.yaml`](configs/dynerf/abuzabi100f_sumexp_shdc.yaml). Despite its filename, the header and frame bounds specify **60 frames, 20–79, at 30 fps**. Its main settings are:

| Setting | Value |
|---|---:|
| Total iterations | 60,000 |
| Batch schedule | 1 view, then 2 views from 55,000 |
| Temporal opacity | `flat_window` |
| Spatial refinement | spatial-only from 500–30,000; alternating spatial events continue through 50,000 |
| Temporal split | 30,000–50,000 |
| Temporal split reflection threshold | \(10^{-5}\) |
| Global lighting start | 9,000 |
| Local feature start | 12,000 |
| Local-output penalty start / weight | 15,000 / 0.05 |
| Fourier activation | all five bands at 30,000 |
| Reflection fusion | `sum_exp` |
| Diffuse source | SH DC |
| Dynamic Sph-Mip residual/hierarchical/parity/sliding bands | disabled |
| Camera affine compensation | enabled |

The repository already contains three focused configs suitable for a minimum component study:

| Config | Disabled component | Mechanism |
|---|---|---|
| [`abuzabi_abl_flat.yaml`](configs/dynerf/abuzabi_abl_flat.yaml) | Flat-top temporal opacity | switches to `normalized_sigmoid` temporal opacity |
| [`abuzabi_abl_fourier.yaml`](configs/dynerf/abuzabi_abl_fourier.yaml) | Fourier global feature | moves Fourier start to iteration 10,000,000 |
| [`abuzabi_abl_split.yaml`](configs/dynerf/abuzabi_abl_split.yaml) | Temporal bifurcation | moves split start to iteration 10,000,000 |

Matching `abuzabi_2_abl_*` files provide the same three interventions for the second-scene recipe and should be treated as replication configs rather than direct ablations of the representative run.

These filenames encode interventions, but a paper should still verify that every other effective option is identical before quoting them as controlled ablations. The current repository does not contain an equally explicit local-output-loss ablation; that comparison should be added by changing only `local_mlp_output_loss_weight` from 0.05 to 0.

## 16. Checkpoints and inference contract

A complete reconstruction is more than one PLY or one Gaussian tensor. [`Scene.save`](scene/__init__.py#L219) writes:

- a main Gaussian checkpoint containing dynamic geometry, temporal windows, material features, and optimizer-capturable model state;
- a separate local Gaussian checkpoint;
- the global BRDF/environment base;
- the global and local lighting MLP objects;
- the directional Sph-Mip encoder;
- optional camera-affine state in the training-side checkpoint path.

[`render.py`](render.py#L437) reconstructs both Gaussian models, restores the main lighting modules, and restores the local checkpoint when present. Its `--zero_local_gaussians` option replaces the saved local set with a zero-feature clone. This is a **zero-local-feature diagnostic**, not a true global-only render: `light_mlp_2` remains active and can emit radiance from direction, roughness, normal/reflection cosine, and bias inputs. A true global-only render must explicitly bypass that decoder or its fusion term.

For reproducibility, a release must include both Gaussian files, the `light`, `cubemap`, and direction-encoding assets, and the exact effective YAML/arguments. The checkpoint assets do not encode the fusion rule, diffuse source, temporal-opacity mode, environment sphere, Fourier gate, or branch phase settings; shipping only model tensors can silently select incompatible renderer defaults.

## 17. Code navigation map

| File / directory | Responsibility | Most relevant entry points |
|---|---|---|
| [`train.py`](train.py) | End-to-end optimization, phase gates, data loss assembly, batch gradient aggregation, global/local density-control scheduling, evaluation | `training`, `initialize_local_gaussian_model`, `evaluate_test_views` |
| [`scene/gaussian_model.py`](scene/gaussian_model.py) | Parameter storage, temporal opacity, point-cloud initialization, Sph-Mip and MLP construction, optimizer groups, clone/split/prune | `get_temporal_opacity_factor`, `get_temporal_split_params`, `densify_and_split_time`, `init_light_env` |
| [`gaussian_renderer/__init__.py`](gaussian_renderer/__init__.py) | Time-conditioned planar rasterization, G-buffer construction, Fourier feature, isolated local pass, deferred reflection shading and fusion | `render_3d_pgsr_anti`, `get_normal`, `render_normal`, imported `normal_from_depth_image` |
| [`diff-plane-rasterization-anti/`](diff-plane-rasterization-anti/) | CUDA planar Gaussian rasterizer and backward pass used by the active renderer | Python binding plus `cuda_rasterizer/forward.cu` and `backward.cu` |
| [`scene/__init__.py`](scene/__init__.py) | Scene creation, checkpoint discovery, restore, and multi-artifact save | `Scene.__init__`, `Scene.save` |
| [`scene/dataset_readers.py`](scene/dataset_readers.py) and [`utils/data_utils.py`](utils/data_utils.py) | Camera/video/point-cloud ingestion and precomputed-prior loading | scene loaders and camera item construction |
| [`create_depth_normal_abuzabi.py`](create_depth_normal_abuzabi.py) | Offline monocular depth/normal preprocessing | script entry point |
| [`render.py`](render.py) | Checkpoint restoration, timestamped evaluation, and zero-local-feature diagnostics | `render_set`, `render_sets` |
| [`arguments/__init__.py`](arguments/__init__.py) | Defaults for model, renderer, losses, phase starts, and density control | `ModelParams`, `PipelineParams`, `OptimizationParams` |
| [`configs/dynerf/`](configs/dynerf/) | Experiment recipes and ablations | latest `abuzabi100f_sumexp_shdc.yaml` and three `_abl_` configs |

Linear motion, flat-window opacity, temporal masking, and Fourier feature evaluation are PyTorch-side operations completed before the active PGSR plane rasterizer. The committed `diff-plane-rasterization-anti` CUDA extension alpha-composites the generic 32-channel buffer and contains no timestamp, flat-window, or temporal-split kernel. The inherited `diff-gaussian-rasterization` tree retains native-4DGS temporal CUDA code, but the active renderer import is commented out.

The renderer also auto-selects `diff_plane_rasterization_anti_fast` when installed. Its source directory is untracked in this audited worktree, so it cannot support a reproducibility or speed claim until the exact source/build is committed and documented.

Several similarly named backup or historical files remain in the tree, including `gaussian_model_4d.py`, `gaussian_model_bak.py`, and temporal-hierarchy modules. They should not be used to describe the active `render_3d_pgsr_anti` path unless a specific experiment selects them.

## 18. Implementation chronology from Git history

The commit history provides useful provenance for how the current framework was assembled. It is implementation evidence, not a substitute for a literature novelty search.

| Phase | Representative commits | What entered the code line |
|---|---|---|
| Dynamic base | `b596c06ee` “initial modification for 4d gaussian”; `22a129323` “basic tgh implementation”; `fa1b4eb48` “multiple frame initialization”; `a50448dab` “feature 3dgs and speed” | Native 4DGS-derived training scaffold, multi-frame initialization, explicit dynamic features, and the PGSR-style plane rasterization path introduced at `a50448dab` |
| Motion and geometry | `0c28e5f1e` “add rotation velocity”; `5b368a7b4` “add polynomial trajectory and depth” | Motion variables and depth/geometry integration |
| Global reflection | `af4ce91be` “add env map”; `c3e8c1492` “base specular”; `5e6ce5b7d` “sph encoding”; `494264eca` “specular static” | Environment/light path, specular feature, Sph-Mip-style direction encoding, static global reflection |
| Temporal reflection | `a63739fbd` “sph fourier”; `da4632aac` “sph fourier time split”; `51706d0c4` “flat top fourier time split” | Fourier-conditioned reflection feature, temporal splitting, and their first flat-top combination |
| Dual reflection | `a038d3750` “dgs base”; `5a694213e` “local geo hybrid base”; `61c9d02ce` “local ref abuzabi base” | Auxiliary local Gaussian branch and global/local reflective shading |
| Stable flat-window branch | `29426b2dc` “flat window base”; `4eb5235d1` “flat window time split base” | Current learned plateau/shoulder model and current temporal-bifurcation base |
| Training and memory refinement | `32abd1095` “training optimization”; `ae44f6758` “gpu memory optimization”; `cdffbe3f5` “batch support”; `19dac26d9` “local gaussian fixed”; `fb2d5543d` “experiment base 0727” | Scheduling, memory lifetime fixes, multi-view batch statistics, checkpoint/local-model corrections, latest experiment state |

This chronology supports the paper narrative that 4dRefGS was built by first establishing dynamic geometry, then global reflection, then temporal appearance/refinement, and finally a controlled local branch. It does **not** establish that an idea was first in the literature.

## 19. Claim-safety matrix

| Candidate statement | Status | Recommended wording |
|---|---|---|
| “We introduce temporal opacity in Gaussian splatting.” | Too broad | Native 4DGS and FreeTimeGS already use temporal support. Say: “we replace Gaussian-only temporal opacity with a learned flat-top window.” |
| “We are the first to split Gaussians in time.” | Too broad | Native 4DGS already performs joint spatial-and-temporal sampling during Gaussian splitting. Claim only the exact **trajectory-preserving half-plateau bifurcation**. |
| “We introduce Fourier features for dynamic Gaussians.” | Too broad | Native 4DGS already has temporal harmonic appearance. Say: “we apply paired sine/cosine coefficients to the splatted global reflection feature before directional factorization.” |
| “We propose global and local reflection Gaussians.” | Incorrect attribution | Ref-DGS introduces the dual global/local reflective representation. Credit it and claim the dynamic extension plus explicit local-output regularization. |
| “The local branch handles middle-range reflection.” | Ambiguous | Use “near-field, geometry-dependent specular reflection and mirror-induced virtual imagery,” matching the Ref-DGS vocabulary and the actual local spatial feature mechanism. |
| “The local branch cannot affect geometry.” | Needs qualification | Its raster pass cannot overwrite the main G-buffer, but its shader inputs include main roughness and normal/reflection cosine and can backpropagate through them unless detached. |
| “Fourier handles short changes and splitting handles long changes.” | Design interpretation | Say the two mechanisms are intended to provide complementary smooth within-lifetime variation and piecewise temporal capacity; validate with timescale-specific ablations. There is no explicit router between them. |
| “Our PGSR module recovers correct geometry.” | Too strong | The code uses a subset of PGSR plus monocular priors. Report measured geometry improvement and say “nearer to a coherent planar surface” unless ground-truth geometry proves accuracy. |
| “Our coarse-to-fine pipeline improves efficiency.” | Requires evidence | Report wall-clock time, iterations to a target metric, peak memory, and final primitive count against an all-at-once schedule. |
| “The local-output loss prevents glossy overfitting.” | Empirical hypothesis | Say it discourages unnecessary local radiance; demonstrate global/local decompositions and a zero-weight ablation on glossy versus mirror-like regions. |
| “The model uses FreeTimeGS.” | Imprecise implementation claim | Say “FreeTimeGS-style explicit center-time, duration, and velocity parameters on a native 4DGS-derived codebase.” The complete FreeTimeGS relocation pipeline is not present. |
| “The environment light changes over time.” | Misstates the default path | In the representative config the Sph-Mip itself is static; the per-Gaussian global reflection/material feature changes with time. Optional dynamic Sph-Mip experiments exist but are disabled. |

## 20. Implementation caveats and reproducibility risks

These points should be resolved or disclosed before turning the handoff into a camera-ready method description.

1. **The repository README is stale.** It still presents the upstream native 4DGS project and does not explain 4dRefGS, the required prior directories, the dual checkpoint, or the representative command line.
2. **The active motion is not a full 4D covariance trajectory.** `get_sigma_t_fixed` returns one and the forward path is linear translation. Inherited 4D rotation/trajectory fields do not imply that they are active.
3. **The normal residual is not used for reflection direction.** A threshold of 50,000,000 makes every ordinary run use the unadjusted global normal. The residual is rasterized/regularized, but it does not steer the active reflected ray.
4. **Local Fourier modulation is inactive.** The corresponding update is commented out. The temporal Fourier contribution applies only to the global Gaussian feature.
5. **The active shader is learned radiance, not the retained BRDF call.** NVDIFFREC/BRDF objects remain in the model and checkpoint path, but the direct BRDF shading calls in the renderer are commented; Sph-Mip plus two MLP heads produces the active specular radiance.
6. **Optional dynamic Sph-Mip variants are off in the representative result.** Residual, hierarchical, parity, and sliding keyframe fields exist as experiments but their config strings/counts are empty or zero.
7. **Temporal split has a hidden enable condition.** `densify_and_split_time` immediately returns when `grad_spec_t_threshold` is `None`; therefore even pure photometric time-gradient candidates require a positive configured specular-time threshold to reach the split code.
8. **The split implementation is concretely binary.** The default and training call use `N=2`, and the reflection features are constructed by concatenating exactly two copies. General `N` is not safely supported by all fields.
9. **Fourier storage includes four inactive values.** `_specular2` has 44 values; the active renderer slices off the first `gsdim=4` and interprets only the remaining 40 as a 4×10 coefficient matrix.
10. **The configured `specular_lr` is not wired to the specular feature optimizer group.** `_specular` uses `feature_lr*5` and `_specular2` uses `feature_lr/2`. Reproducing a run requires following the optimizer code, not assuming every YAML key is active.
11. **The local raster is geometrically isolated, but the local shader is not fully gradient-isolated.** With the representative `local_light_mlp_geo_inputs=True` and cosine detachment disabled, local photometric error can still update main roughness/normals through shader inputs.
12. **Prior loading assumes valid files.** When monocular losses are active, missing `sgt_depth`/`sgt_normal` arrays will fail. Depth normalization also has no epsilon around its mean-absolute-deviation denominator, which is unsafe for a constant prior or render.
13. **The representative filename and generic duration are misleading.** `abuzabi100f_sumexp_shdc.yaml` trains frames 20–79, and the effective clip/Sph-Mip interval is approximately 0.667–2.633 seconds even though the top-level generic `time_duration` is `[0,10]`.
14. **Local zero initialization is near-zero, not exactly zero, for `sum_exp`.** Its final bias is -5, so the initial additive radiance is approximately \(e^{-5}\), while the other fusion mode uses a different neutral convention.
15. **Several flat-window YAML knobs are inactive legacy fields.** The active radius/edge getters ignore `temporal_flat_radius_mult`, `temporal_flat_edge_sigma_mult`, the flat-radius ramp/loss fields, and `temporal_schedule_iteration`; `temporal_opacity_k_*` applies only outside `flat_window` mode. The effective flat-window weights in `train.py` are the values that must be reported.
16. **Raw flat radius can enter a dead zone.** The active parameterization is `clamp_min(_velocity2[...,0],0)` with identity inverse activation. A negative raw radius receives zero gradient through the clamp, and the upper clamp exposed by `get_velocity2` is bypassed because the flat-window getter reads `_velocity2` directly.
17. **The optional fast rasterizer is not repository-reproducible in this worktree.** It is imported automatically when installed, but `diff-plane-rasterization-anti-fast/` is untracked. Commit and version it before attributing the documented speed comment to a reproducible release.
18. **Paper-level efficiency and generalization are not established by code structure.** The staged schedule is plausible and extensively tuned in comments, but claims over multiple scenes require clean benchmark tables, seeds, timing protocol, and comparable baselines.

## 21. Evidence and ablation plan for the paper-writing agent

The current code supports the proposed mechanisms, but the paper still needs experiments that isolate them. A defensible study would use the same initialization, total iterations, data, seed set, and final evaluation path for each row.

| Comparison | Required intervention | Primary evidence |
|---|---|---|
| Explicit-motion global baseline | Flat window off, split off, Fourier off, local branch off | Establish the difficulty of dynamic reflective reconstruction |
| + flat-top opacity | Use the supplied flat-window intervention only | Framewise PSNR/SSIM/LPIPS, temporal flicker, visible-feature variance, plateau/support histograms |
| + temporal bifurcation | Enable split with all other settings fixed | Fast-motion/lighting-change crops, per-frame error around split events, Gaussian count, transient error immediately after a split |
| + Fourier reflection feature | Enable Fourier only | Short-timescale reflection/shadow changes, global-feature temporal spectra, comparison with static global features |
| + local branch | Enable the local feature/radiance head without its usage penalty | Mirror/near-field region quality and global/local decomposition |
| + local-output loss | Change only weight 0→0.05 | Glossy-region error, mirror-region error, mean/percentile local usage, local-usage heatmaps, local Gaussian count |
| + staged schedule | Compare with all reflection/refinement modules active from iteration 0 | Time-to-quality curve, wall-clock, peak memory, convergence failures, final quality |
| PGSR-style geometry | Remove planarity and rendered-normal/depth-normal consistency; separately remove monocular priors | Normal angular error, depth error, Chamfer/F-score if ground truth exists, and reflection quality caused by changed normals |
| Global/local fusion mode | Compare `sum_exp` and `exp_sum` with mode-appropriate initialization/penalty | Reconstruction, branch attribution stability, training dynamics |

Recommended reporting details:

- report metrics on the full frame and on reflective/mirror, glossy, and non-reflective masks;
- plot per-frame metrics rather than only a sequence average, because the contribution targets time-localized failures;
- include a true global-only render with the local decoder bypassed, the zero-local-feature diagnostic, the combined render, local-usage, roughness, normal, and depth visualizations for the same views;
- report final main/local primitive counts, peak GPU memory, training time, and render FPS;
- repeat key ablations with multiple seeds, since clone/split/prune decisions are history-dependent;
- for the “short Fourier / long split” interpretation, create or annotate intervals by temporal frequency and test each mechanism on both groups rather than inferring timescale behavior from one aggregate score;
- compare against native 4DGS/FreeTimeGS-style dynamic appearance, PGSR geometry constraints, Ref-GS global reflection, and Ref-DGS dual reflection under clearly stated static/dynamic applicability. If a prior method cannot process the dynamic sequence directly, state the adaptation rather than presenting it as an unmodified baseline.

## 22. Manuscript-ready method synopsis

### 22.1 Compact method paragraph

> We represent a dynamic reflective scene with a primary set of explicitly moving, planarized 3D Gaussians and an auxiliary set of local-reflection Gaussians. Each primitive follows a linear trajectory and is gated by a learned flat-top temporal opacity window that keeps its temporal-opacity multiplier constant within the plateau, avoiding opacity-induced attenuation of splatted attributes. A reflection- and time-gradient-driven bifurcation operator replaces under-resolved primitives with two trajectory-aligned children of half plateau width, enabling piecewise temporal specialization without changing the underlying linear trajectory. The primary rasterization produces diffuse appearance, plane geometry, roughness, and a global reflection feature; a separate rasterization produces only a local reflection feature. A roughness-aware spherical Mip encoding of the reflected direction is factorized with the global Gaussian feature and decoded as global radiance, while a smaller local head predicts a positive additive local-reflection radiance term in the representative `sum_exp` mode. Paired Fourier coefficients make the global reflection feature vary smoothly over time, and an output-space local-usage penalty encourages the optimizer to use the local branch only when its photometric benefit justifies its cost. Optimization first establishes geometry and base appearance, then activates global and local reflection; Fourier variation and alternating temporal/spatial refinement begin jointly after that reflective solution is established.

### 22.2 Defensible contribution bullets

1. **Stable temporal support.** A learned flat-top opacity window decouples high-opacity lifetime from boundary smoothness, keeps the temporal-opacity multiplier constant within the plateau, and avoids opacity-induced attenuation of splatted attributes while retaining differentiable temporal entry and exit.
2. **Motion-consistent temporal refinement.** A binary temporal bifurcation preserves the parent trajectory at both child center times, halves plateau coverage, inherits optimizer state, and lets the children subsequently specialize.
3. **Time-varying global reflective features.** Per-Gaussian sine/cosine coefficients modulate the compact global reflection feature before Ref-GS-style spherical directional factorization, adding temporal appearance capacity without duplicating every primitive.
4. **Regularized dynamic dual reflection.** The Ref-DGS-inspired global/local decomposition is extended to temporally supported Gaussians with separated output heads and an output-space usage loss that discourages unnecessary local radiance.
5. **Staged geometry-to-reflection optimization.** A practical phase schedule establishes planar dynamic geometry first, then global lighting and the local reflection branch, followed jointly at 30k by Fourier variation and alternating spatial/temporal refinement, targeting the instability and cost of optimizing every capacity from the beginning.

The first sentence of every contribution should name the exact mechanism. Do not let the paper collapse these into the generic claim “we combine four existing methods”; the code contribution lies in how temporal support/refinement and reflective decomposition interact.

## 23. Primary references reviewed

1. Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. [“Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting.”](https://arxiv.org/abs/2310.10642) ICLR 2024.
2. Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. [“FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction.”](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.html) CVPR 2025, pp. 21750–21760.
3. Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. [“PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction.”](https://arxiv.org/abs/2406.06521) IEEE TVCG 31(9):6100–6111, September 2025; DOI 10.1109/TVCG.2024.3494046 (early access 2024).
4. Youjia Zhang, Anpei Chen, Yumin Wan, Zikai Song, Junqing Yu, Yawei Luo, and Wei Yang. [“Ref-GS: Directional Factorization for 2D Gaussian Splatting.”](https://arxiv.org/abs/2412.00905) CVPR 2025.
5. Ningjing Fan, Yiqun Wang, Dong-Ming Yan, and Peter Wonka. [“Ref-DGS: Reflective Dual Gaussian Splatting.”](https://doi.org/10.1145/3799902.3811163) ACM SIGGRAPH 2026 Conference Papers; DOI 10.1145/3799902.3811163.

The comparison in this document uses the primary papers/project formulations rather than repository names alone. This matters particularly for PGSR: the full paper includes unbiased planar depth, single-view and multi-view constraints, and exposure compensation, whereas the current project activates only a subset plus its own external priors.

## 24. Final attribution summary

The cleanest way to communicate the method to a paper-writing agent is to separate inheritance, adaptation, and project-specific additions:

| Category | Components |
|---|---|
| **Inherited foundation** | Native 4DGS-derived project/training scaffold; Gaussian rasterization and time-aware machinery |
| **Adapted prior representations** | FreeTimeGS-style explicit position/time/duration/velocity; PGSR-style planar normal/depth G-buffer and single-view consistency; Ref-GS-style Sph-Mip directional factorization; Ref-DGS-style complementary local reflection Gaussians |
| **4dRefGS code-level additions** | Learned flat-top temporal opacity; trajectory-preserving half-plateau temporal bifurcation; sine/cosine modulation of the global reflection feature; temporally supported global/local integration; separate measurable radiance heads; local-output usage loss; tuned staged training and alternating spatial/temporal refinement |
| **Optional experiments, not representative-method defaults** | Dynamic residual/hierarchical/parity/sliding Sph-Mip encodings, local Gaussian temporal-opacity override, alternative fusion modes and several commented BRDF/normal variants |

In short, the strongest contribution is not any one inherited renderer component. It is a **flat-window-gated and temporally refinable dynamic Gaussian representation coupled to a deliberately capacity-controlled global/local reflection decomposition**. That framing matches the active code, acknowledges the prior methods, and gives the future paper a concrete experimental burden for every claim.
