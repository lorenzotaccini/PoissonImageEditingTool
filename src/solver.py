import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, cg
from scipy.ndimage import sobel


def poisson_edit(source, destination, mask, offset=(0, 0), mix_gradients=False):
    """
    Seamlessly blends a source region into a destination image using
    Poisson image editing (Perez et al., SIGGRAPH 2003).

    Parameters
    ----------
    source       : (H_s, W_s, 3) uint8 source image
    destination  : (H_d, W_d, 3) uint8 destination image
    mask         : (H_s, W_s)    binary mask; values > 128 are foreground
    offset       : (y, x)        position of source top-left in destination
    mix_gradients: bool          use mixed-gradient guidance (eq. 12-13)

    Returns
    -------
    result : (H_d, W_d, 3) uint8
    """
    guidance = _build_import_guidance(source, destination, mask, offset,
                                      mix_gradients)
    return _solve(source, destination, mask, offset, guidance)


def poisson_edit_flatten(source, destination, mask, offset=(0, 0),
                         edge_threshold=0.1):
    """
    Section 4 — Texture flattening (eq. 14-15).

    Passes the source gradient through a sparse edge sieve: only gradients
    at detected edges are retained; everything else is zeroed out.
    The result looks flat/cartoon-like inside the selection while blending
    seamlessly into the destination.

    Parameters
    ----------
    edge_threshold : float  fraction of max gradient magnitude kept as edges
    """
    guidance = _build_flatten_guidance(source, mask, edge_threshold)
    return _solve(source, destination, mask, offset, guidance)


def poisson_edit_illumination(source, destination, mask, offset=(0, 0),
                               alpha_factor=0.2, beta=0.2):
    """
    Section 4 — Local illumination change (eq. 16).

    Applies a non-linear compression to the gradient field of the source
    to locally correct exposure or reduce specular highlights.

    Parameters
    ----------
    alpha_factor : float  fraction of mean gradient norm used as alpha (paper: 0.2)
    beta         : float  compression exponent (paper: 0.2)
    """
    guidance = _build_illumination_guidance(source, mask, alpha_factor, beta)
    return _solve(source, destination, mask, offset, guidance)


def poisson_edit_color(source, destination, mask, offset=(0, 0),
                       red_scale=1.0, green_scale=1.0, blue_scale=1.0):
    """
    Section 4 — Local color change.
    Scales the gradients of each channel independently to change tint
    while preserving texture and shading.
    """
    scales = [red_scale, green_scale, blue_scale]
    
    def guidance(c, y_idx, x_idx, y_d, x_d, mask_ids):
        s_c = source[:, :, c].astype(np.float64)
        vpq_grid = {}
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny_s = y_idx + dy
            nx_s = x_idx + dx
            in_src = (ny_s >= 0) & (ny_s < s_c.shape[0]) & (nx_s >= 0) & (nx_s < s_c.shape[1])
            ny_s_c = np.clip(ny_s, 0, s_c.shape[0] - 1)
            nx_s_c = np.clip(nx_s, 0, s_c.shape[1] - 1)
            
            raw_vpq = in_src.astype(np.float64) * (s_c[y_idx, x_idx] - s_c[ny_s_c, nx_s_c])
            # Apply color scale to the gradient
            vpq_grid[(dy, dx)] = scales[c] * raw_vpq
        return vpq_grid

    return _solve(source, destination, mask, offset, guidance)

def _prepare_mask(mask, source_shape, destination_shape, offset):
    """
    Return (y_idx, x_idx, y_d, x_d, mask_ids) for the valid mask pixels.
    Also clips away mask pixels whose destination coordinates are out of bounds.
    """
    h_s, w_s = source_shape[:2]
    h_d, w_d = destination_shape[:2]
    y_off, x_off = offset

    m = (mask > 128).astype(np.uint8)
    y_raw, x_raw = np.where(m > 0)

    valid = (
        (y_raw + y_off >= 0) & (y_raw + y_off < h_d) &
        (x_raw + x_off >= 0) & (x_raw + x_off < w_d)
    )
    y_idx, x_idx = y_raw[valid], x_raw[valid]
    y_d = y_idx + y_off
    x_d = x_idx + x_off

    mask_ids = np.full((h_s, w_s), -1, dtype=np.intp)
    mask_ids[y_idx, x_idx] = np.arange(len(y_idx))

    return y_idx, x_idx, y_d, x_d, mask_ids


def _build_laplacian(y_idx, x_idx, mask_ids, source_shape):
    """
    Build the sparse Laplacian matrix A over the mask region.

    The diagonal entry for pixel p is |Np| — the number of valid
    source-image neighbours (NOT always 4: pixels at the source image
    boundary have fewer valid neighbours, per Section 2 of the paper).
    """
    h_s, w_s = source_shape[:2]
    N = len(y_idx)

    all_rows, all_cols, all_data = [], [], []
    diag_count = np.zeros(N, dtype=np.int32)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = y_idx + dy
        nx = x_idx + dx
        in_src = (ny >= 0) & (ny < h_s) & (nx >= 0) & (nx < w_s)
        diag_count += in_src        # |Np|: count only valid source neighbours

        ny_c = np.clip(ny, 0, h_s - 1)
        nx_c = np.clip(nx, 0, w_s - 1)
        n_idx = np.where(in_src, mask_ids[ny_c, nx_c], -1)
        in_mask = n_idx != -1       # neighbour is inside Ω

        pos = np.where(in_mask)[0]
        all_rows.append(pos)
        all_cols.append(n_idx[in_mask])
        all_data.append(np.full(len(pos), -1.0))

    # Diagonal
    all_rows.append(np.arange(N))
    all_cols.append(np.arange(N))
    all_data.append(diag_count.astype(np.float64))

    A = coo_matrix(
        (np.concatenate(all_data),
         (np.concatenate(all_rows), np.concatenate(all_cols))),
        shape=(N, N)
    ).tocsr()
    return A


def _accumulate_rhs(b, channel_idx, y_idx, x_idx, y_d, x_d,
                    mask_ids, source_channel, dest_channel,
                    source_shape, dest_shape, vpq_grid):
    """
    Vectorised accumulation of guidance divergence and boundary conditions
    into the RHS vector b, per colour channel.

    vpq_grid : dict mapping direction (dy,dx) → 1-D array of vpq values
               (one entry per mask pixel). These are pre-computed by the
               guidance-field builder so this function stays generic.
    """
    h_s, w_s = source_shape[:2]
    h_d, w_d = dest_shape[:2]
    N = len(y_idx)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny_s = y_idx + dy
        nx_s = x_idx + dx
        ny_d_nb = y_d + dy
        nx_d_nb = x_d + dx

        in_src = (ny_s >= 0) & (ny_s < h_s) & (nx_s >= 0) & (nx_s < w_s)
        ny_s_c = np.clip(ny_s, 0, h_s - 1)
        nx_s_c = np.clip(nx_s, 0, w_s - 1)
        ny_d_c = np.clip(ny_d_nb, 0, h_d - 1)
        nx_d_c = np.clip(nx_d_nb, 0, w_d - 1)

        # ---- Guidance divergence term ----
        b += vpq_grid[(dy, dx)]

        # ---- Boundary condition: f*_q for neighbours outside Ω ----
        # A neighbour is a boundary pixel when it is outside the source image
        # OR inside the source but not selected (mask_ids == -1).
        safe_n_idx = np.where(in_src, mask_ids[ny_s_c, nx_s_c], -1)
        is_boundary = (safe_n_idx == -1)
        b += is_boundary.astype(np.float64) * dest_channel[ny_d_c, nx_d_c]


def _build_import_guidance(source, destination, mask, offset, mix_gradients):
    """
    Returns a callable `guidance(c, y_idx, x_idx, y_d, x_d, ...)`
    that produces vpq_grid for the import-gradients / mixed-gradients case.
    """
    y_off, x_off = offset
    h_s, w_s = source.shape[:2]
    h_d, w_d = destination.shape[:2]

    def guidance(c, y_idx, x_idx, y_d, x_d, mask_ids):
        s_c = source[:, :, c].astype(np.float64)
        d_c = destination[:, :, c].astype(np.float64)
        vpq_grid = {}

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny_s = y_idx + dy
            nx_s = x_idx + dx
            ny_d_nb = y_d + dy
            nx_d_nb = x_d + dx

            in_src = (ny_s >= 0) & (ny_s < h_s) & (nx_s >= 0) & (nx_s < w_s)
            in_dst = (ny_d_nb >= 0) & (ny_d_nb < h_d) & (nx_d_nb >= 0) & (nx_d_nb < w_d)

            ny_s_c = np.clip(ny_s, 0, h_s - 1)
            nx_s_c = np.clip(nx_s, 0, w_s - 1)
            ny_d_c = np.clip(ny_d_nb, 0, h_d - 1)
            nx_d_c = np.clip(nx_d_nb, 0, w_d - 1)

            # Source gradient v_pq = g_p - g_q (zero when neighbour out-of-source)
            gq = s_c[ny_s_c, nx_s_c]
            vpq = in_src.astype(np.float64) * (s_c[y_idx, x_idx] - gq)

            if mix_gradients:
                # Eq. 12-13: retain whichever gradient has the larger magnitude
                vpq_d = in_dst.astype(np.float64) * (
                    d_c[y_d, x_d] - d_c[ny_d_c, nx_d_c]
                )
                vpq = np.where(np.abs(vpq_d) > np.abs(vpq), vpq_d, vpq)

            vpq_grid[(dy, dx)] = vpq
        return vpq_grid

    return guidance


def _build_flatten_guidance(source, mask, edge_threshold):
    """
    Section 4, eq. 14-15 — Texture flattening.

    Passes the source gradient through a sparse edge sieve M: only pixels
    where the gradient magnitude exceeds `edge_threshold * max_magnitude`
    are retained; all others are zeroed out.
    """
    h_s, w_s = source.shape[:2]

    def guidance(c, y_idx, x_idx, y_d, x_d, mask_ids):
        s_c = source[:, :, c].astype(np.float64)
        # Edge magnitude via Sobel
        gx = sobel(s_c, axis=1)
        gy = sobel(s_c, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        edge_mask = mag > (edge_threshold * mag.max())

        vpq_grid = {}
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny_s = y_idx + dy
            nx_s = x_idx + dx
            in_src = (ny_s >= 0) & (ny_s < h_s) & (nx_s >= 0) & (nx_s < w_s)
            ny_s_c = np.clip(ny_s, 0, h_s - 1)
            nx_s_c = np.clip(nx_s, 0, w_s - 1)

            raw_vpq = in_src.astype(np.float64) * (
                s_c[y_idx, x_idx] - s_c[ny_s_c, nx_s_c]
            )
            # M(x): keep gradient only where an edge lies between p and q
            edge_between = edge_mask[y_idx, x_idx] | edge_mask[ny_s_c, nx_s_c]
            vpq_grid[(dy, dx)] = edge_between.astype(np.float64) * raw_vpq

        return vpq_grid

    return guidance


def _build_illumination_guidance(source, mask, alpha_factor, beta):
    """
    Section 4, eq. 16 — Local illumination change.

    Applies v = alpha * beta * |∇f*|^(-beta) * ∇f* to compress large
    gradients and boost small ones, correcting local exposure.
    """
    h_s, w_s = source.shape[:2]

    def guidance(c, y_idx, x_idx, y_d, x_d, mask_ids):
        s_c = source[:, :, c].astype(np.float64)
        gx = sobel(s_c, axis=1)
        gy = sobel(s_c, axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        # Avoid division by zero
        eps = 1e-6
        alpha = alpha_factor * (mag[y_idx, x_idx].mean() + eps)
        scale = alpha * beta * (mag + eps) ** (-beta)   # spatially-varying

        vpq_grid = {}
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny_s = y_idx + dy
            nx_s = x_idx + dx
            in_src = (ny_s >= 0) & (ny_s < h_s) & (nx_s >= 0) & (nx_s < w_s)
            ny_s_c = np.clip(ny_s, 0, h_s - 1)
            nx_s_c = np.clip(nx_s, 0, w_s - 1)

            raw_vpq = in_src.astype(np.float64) * (
                s_c[y_idx, x_idx] - s_c[ny_s_c, nx_s_c]
            )
            # Scale by the alpha-beta factor at pixel p
            vpq_grid[(dy, dx)] = scale[y_idx, x_idx] * raw_vpq

        return vpq_grid

    return guidance


def _solve(source, destination, mask, offset, guidance_fn):
    """
    Core solve: build A, assemble b per channel, solve A x = b.
    """
    y_off, x_off = offset

    y_idx, x_idx, y_d, x_d, mask_ids = _prepare_mask(
        mask, source.shape, destination.shape, offset
    )
    N = len(y_idx)
    if N == 0:
        return destination.copy()

    A = _build_laplacian(y_idx, x_idx, mask_ids, source.shape)
    result = destination.copy().astype(np.float64)

    for c in range(3):
        print(f"  Channel {c}…", end=" ", flush=True)
        d_c = destination[:, :, c].astype(np.float64)

        vpq_grid = guidance_fn(c, y_idx, x_idx, y_d, x_d, mask_ids)

        b = np.zeros(N)
        _accumulate_rhs(b, c, y_idx, x_idx, y_d, x_d,
                        mask_ids,
                        source[:, :, c].astype(np.float64), d_c,
                        source.shape, destination.shape,
                        vpq_grid)

        # Direct solver for moderate sizes; iterative for large masks.
        # spsolve is more robust than CG for typical use; CG avoids the
        # memory cost of factorisation for very large regions.
        if N < 100_000:
            x_sol = spsolve(A, b)
        else:
            x_sol, info = cg(A, b, rtol=1e-6, maxiter=3 * N)
            if info != 0:
                print(f"\n  Warning: CG {'did not converge' if info > 0 else 'broke down'} "
                      f"(info={info}).")

        result[y_d, x_d, c] = x_sol
        print("done")

    return np.clip(result, 0, 255).astype(np.uint8)