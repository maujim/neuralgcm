"""MLX implementations of JAX indexing primitives."""

from __future__ import annotations

import mlx.core as mx


def _mode(params):
    mode = params.get('mode')
    return str(mode).split('.')[-1] if mode is not None else 'PROMISE_IN_BOUNDS'


def _shape_index(index, axis, rank):
    shape = [1] * rank
    shape[axis] = index.shape[0]
    return mx.reshape(index, tuple(shape))


def _axis_grid(shape, axis, dtype):
    index = mx.arange(shape[axis], dtype=dtype)
    return _shape_index(index, axis, len(shape))


def _indices_vector(indices, index_vector_dim, component):
    if index_vector_dim == indices.ndim:
        if component != 0:
            raise ValueError('implicit index vector dimension has size one')
        return indices
    return mx.take(indices, component, axis=index_vector_dim)


def _gather(operand, indices, params, out_aval):
    dnums = params['dimension_numbers']
    operand_batch_dims = tuple(dnums.operand_batching_dims)
    indices_batch_dims = tuple(dnums.start_indices_batching_dims)
    # JAX dimension numbers omit index_vector_dim; it is always the final axis.
    index_vector_dim = indices.ndim - 1
    index_vector_size = indices.shape[index_vector_dim]
    if index_vector_size != len(dnums.start_index_map):
        raise ValueError(
            'gather index vector size does not match start_index_map')
    slice_sizes = tuple(int(size) for size in params['slice_sizes'])
    if len(slice_sizes) != operand.ndim:
        raise ValueError('gather slice_sizes rank does not match operand')

    output_shape = tuple(out_aval.shape)
    offset_dims = tuple(dnums.offset_dims)
    window_operand_dims = tuple(axis for axis in range(operand.ndim)
                                if axis not in dnums.collapsed_slice_dims and
                                axis not in operand_batch_dims)
    index_batch_axes = tuple(
        axis for axis in range(indices.ndim) if axis != index_vector_dim)
    if (len(offset_dims) != len(window_operand_dims) or
            len(indices_batch_dims) != len(operand_batch_dims) or
            len(output_shape) != len(offset_dims) + len(index_batch_axes)):
        raise ValueError(f'invalid gather dimension numbers: {dnums}')

    output_batch_dims = tuple(
        axis for axis in range(len(output_shape)) if axis not in offset_dims)
    if len(output_batch_dims) != len(index_batch_axes):
        raise ValueError(f'invalid gather output dimensions: {dnums}')
    batch_output_axis = dict(zip(index_batch_axes, output_batch_dims))
    output_grids = [
        _axis_grid(output_shape, axis, indices.dtype)
        for axis in range(len(output_shape))
    ]

    starts = {}
    for component, operand_axis in enumerate(dnums.start_index_map):
        raw = _indices_vector(indices, index_vector_dim, component)
        batch_shape = [1] * len(output_shape)
        for source_axis in index_batch_axes:
            batch_shape[batch_output_axis[source_axis]] = raw.shape[
                index_batch_axes.index(source_axis)]
        starts[operand_axis] = mx.reshape(raw, tuple(batch_shape))

    valid = mx.array(True)
    coords = []
    mode = _mode(params)
    for operand_axis, extent in enumerate(operand.shape):
        if operand_axis in operand_batch_dims:
            batch_axis = indices_batch_dims[operand_batch_dims.index(
                operand_axis)]
            coord = output_grids[batch_output_axis[batch_axis]]
        elif operand_axis in starts:
            coord = starts[operand_axis]
            size = slice_sizes[operand_axis]
            if mode == 'CLIP':
                coord = mx.minimum(mx.maximum(coord, 0), extent - size)
            elif mode == 'FILL_OR_DROP':
                valid = valid & (coord >= 0) & (coord <= extent - size)
                coord = mx.minimum(mx.maximum(coord, 0), extent - size)
            elif mode != 'PROMISE_IN_BOUNDS':
                raise NotImplementedError(
                    f'unsupported MLX gather mode: {mode}')
        else:
            output_axis = offset_dims[window_operand_dims.index(operand_axis)]
            coord = starts.get(operand_axis, 0) + output_grids[output_axis]
        coords.append(coord)

    result = operand[tuple(coords)]
    if mode == 'FILL_OR_DROP':
        fill_value = params.get('fill_value')
        if fill_value is None:
            if operand.dtype in (mx.float16, mx.float32, mx.float64):
                fill_value = float('nan')
            elif mx.issubdtype(operand.dtype, mx.integer):
                fill_value = mx.iinfo(operand.dtype).min
            else:
                fill_value = False
        result = mx.where(valid, result,
                          mx.array(fill_value, dtype=operand.dtype))
    return result


def _scatter(operand, indices, updates, params, name):
    dnums = params['dimension_numbers']
    # JAX dimension numbers omit index_vector_dim; it is always the final axis.
    index_vector_dim = indices.ndim - 1
    index_batch_axes = tuple(
        axis for axis in range(indices.ndim) if axis != index_vector_dim)
    update_window_dims = tuple(dnums.update_window_dims)
    update_scatter_axes = tuple(
        axis for axis in range(updates.ndim) if axis not in update_window_dims)
    scatter_operand_dims = tuple(dnums.scatter_dims_to_operand_dims)
    operand_batch_dims = tuple(dnums.operand_batching_dims)
    index_batching_dims = tuple(dnums.scatter_indices_batching_dims)
    if (len(index_batch_axes) != len(update_scatter_axes) or
            len(scatter_operand_dims) != (1 if index_vector_dim == indices.ndim
                                          else indices.shape[index_vector_dim])
            or len(operand_batch_dims) != len(index_batching_dims)):
        raise ValueError(f'invalid scatter dimension numbers: {dnums}')

    update_shape = tuple(updates.shape)
    update_grids = [
        _axis_grid(update_shape, axis, indices.dtype)
        for axis in range(updates.ndim)
    ]
    index_batch_to_update_axis = dict(zip(index_batch_axes,
                                          update_scatter_axes))
    starts = {}
    for component, operand_axis in enumerate(scatter_operand_dims):
        raw = _indices_vector(indices, index_vector_dim, component)
        batch_shape = [1] * updates.ndim
        for source_axis in index_batch_axes:
            batch_shape[index_batch_to_update_axis[source_axis]] = raw.shape[
                index_batch_axes.index(source_axis)]
        starts[operand_axis] = mx.reshape(raw, tuple(batch_shape))

    window_operand_dims = tuple(axis for axis in range(operand.ndim)
                                if axis not in dnums.inserted_window_dims and
                                axis not in operand_batch_dims)
    if len(window_operand_dims) != len(update_window_dims):
        raise ValueError(f'invalid scatter window dimensions: {dnums}')
    update_axis_for_operand = dict(zip(window_operand_dims, update_window_dims))
    mode = _mode(params)
    # FILL_OR_DROP drops each indexed update window if any start index is
    # outside the range that can contain the full window.
    window_valid = mx.array(True)
    if mode == 'FILL_OR_DROP':
        for operand_axis, start in starts.items():
            window_size = (updates.shape[update_axis_for_operand[operand_axis]]
                           if operand_axis in update_axis_for_operand else 1)
            window_valid = (
                window_valid & (start >= 0) &
                (start <= operand.shape[operand_axis] - window_size))
    valid = window_valid
    coords = []
    for operand_axis, extent in enumerate(operand.shape):
        if operand_axis in operand_batch_dims:
            index_batch_axis = index_batching_dims[operand_batch_dims.index(
                operand_axis)]
            coord = update_grids[index_batch_to_update_axis[index_batch_axis]]
        elif operand_axis in starts:
            coord = starts[operand_axis]
            window_size = (updates.shape[update_axis_for_operand[operand_axis]]
                           if operand_axis in update_axis_for_operand else 1)
            if mode == 'CLIP':
                coord = mx.minimum(mx.maximum(coord, 0), extent - window_size)
            elif mode not in ('FILL_OR_DROP', 'PROMISE_IN_BOUNDS'):
                raise NotImplementedError(
                    f'unsupported MLX scatter mode: {mode}')
            if operand_axis in update_axis_for_operand:
                coord = coord + update_grids[
                    update_axis_for_operand[operand_axis]]
        else:
            coord = starts.get(operand_axis, 0)
            if operand_axis in update_axis_for_operand:
                coord = coord + update_grids[
                    update_axis_for_operand[operand_axis]]
        if mode == 'FILL_OR_DROP':
            valid = valid & (coord >= 0) & (coord < extent)
        coords.append(coord)

    if mode not in ('CLIP', 'FILL_OR_DROP', 'PROMISE_IN_BOUNDS'):
        raise NotImplementedError(f'unsupported MLX scatter mode: {mode}')
    linear_indices = mx.zeros(update_shape, dtype=indices.dtype)
    stride = 1
    for operand_axis in range(operand.ndim - 1, -1, -1):
        linear_indices = linear_indices + coords[operand_axis] * stride
        stride *= operand.shape[operand_axis]

    if name == 'scatter':
        # Use a sink slot for dropped updates; MLX does not support boolean
        # indexing, and the extra slot keeps invalid updates out of the result.
        flat_operand = operand.reshape((-1,))
        flat_indices = linear_indices.reshape((-1,))
        flat_updates = updates.reshape((-1,))
        if mode == 'FILL_OR_DROP':
            sink = flat_operand.size
            flat_operand = mx.concatenate(
                [flat_operand,
                 mx.zeros((1,), dtype=operand.dtype)])
            flat_indices = mx.where(valid.reshape((-1,)), flat_indices,
                                    mx.array(sink, dtype=indices.dtype))
            flat_operand[flat_indices] = flat_updates
            return flat_operand[:sink].reshape(operand.shape)
        flat_operand[flat_indices] = flat_updates
        return flat_operand.reshape(operand.shape)

    if mode == 'FILL_OR_DROP':
        linear_indices = mx.where(valid, linear_indices, 0)
        if name == 'scatter_add':
            neutral = 0
        elif name == 'scatter_min':
            neutral = (float('inf') if updates.dtype in (mx.float16, mx.float32,
                                                         mx.float64) else
                       mx.iinfo(updates.dtype).max)
        else:
            neutral = (float('-inf') if updates.dtype
                       in (mx.float16, mx.float32,
                           mx.float64) else mx.iinfo(updates.dtype).min)
        updates = mx.where(valid, updates, mx.array(neutral,
                                                    dtype=updates.dtype))
        indexed = operand.reshape((-1,)).at[mx.reshape(linear_indices, (-1,))]
        if name == 'scatter_add':
            return indexed.add(updates.reshape((-1,))).reshape(operand.shape)
        if name == 'scatter_min':
            return indexed.minimum(updates.reshape(
                (-1,))).reshape(operand.shape)
        if name == 'scatter_max':
            return indexed.maximum(updates.reshape(
                (-1,))).reshape(operand.shape)
    indexed = operand.at[tuple(coords)]
    if name == 'scatter_add':
        return indexed.add(updates)
    if name == 'scatter_min':
        return indexed.minimum(updates)
    if name == 'scatter_max':
        return indexed.maximum(updates)
    raise NotImplementedError(f'MLX scatter primitive unsupported: {name}')


def evaluate_indexing(name, args, params, out_avals):
    """Evaluate a supported JAX indexing primitive using MLX arrays only."""
    if name == 'gather':
        return [_gather(args[0], args[1], params, out_avals[0])]
    if name in ('scatter', 'scatter_add', 'scatter_min', 'scatter_max'):
        return [_scatter(args[0], args[1], args[2], params, name)]
    if name == 'dynamic_slice':
        operand, *starts = args
        sizes = tuple(int(size) for size in params['slice_sizes'])
        if len(sizes) != operand.ndim or len(starts) != operand.ndim:
            raise ValueError(
                'dynamic_slice requires one start per operand axis')
        result = operand
        for axis, (start, size,
                   extent) in enumerate(zip(starts, sizes, operand.shape)):
            start = mx.minimum(mx.maximum(start, 0), extent - size)
            indices = start + mx.arange(size, dtype=start.dtype)
            result = mx.take(result, indices, axis=axis)
        return [result]
    if name == 'dynamic_update_slice':
        operand, update, *starts = args
        if len(starts) != operand.ndim or update.ndim != operand.ndim:
            raise ValueError(
                'dynamic_update_slice requires one start per operand axis')
        shape = tuple(update.shape)
        coords = []
        for axis, (start, size,
                   extent) in enumerate(zip(starts, shape, operand.shape)):
            if size > extent:
                raise ValueError('dynamic_update_slice update exceeds operand')
            start = mx.minimum(mx.maximum(start, 0), extent - size)
            coords.append(start + mx.arange(size, dtype=start.dtype))
        if not starts:
            return [update]
        result = operand
        result[tuple(coords)] = update
        return [result]
    raise NotImplementedError(
        f"MLX indexing helper does not own primitive '{name}'")
