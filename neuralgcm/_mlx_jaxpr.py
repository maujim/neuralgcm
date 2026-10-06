# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Execution of traced JAX programs using Apple MLX arrays.

JAX is used only to trace and infer the result pytree. All numerical execution
is performed by MLX; primitives without an MLX implementation fail explicitly.
"""

from __future__ import annotations

import operator
import time

from jax.extend import core as jax_core

import jax
import mlx.core as mx
import numpy as np
from neuralgcm._mlx_indexing import evaluate_indexing
from neuralgcm._mlx_math import evaluate_math
from neuralgcm._mlx_random import evaluate_random


def _primitive_counts(closed):
    """Counts primitives throughout nested jaxprs."""
    counts = {}
    seen = set()

    def visit(value):
        jaxpr = _as_jaxpr(value)
        if id(jaxpr) in seen:
            return
        seen.add(id(jaxpr))
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            counts[name] = counts.get(name, 0) + 1
            for param in eqn.params.values():
                if isinstance(param, (tuple, list)):
                    for item in param:
                        if hasattr(item, 'jaxpr') or hasattr(item, 'eqns'):
                            visit(item)
                elif hasattr(param, 'jaxpr') or hasattr(param, 'eqns'):
                    visit(param)

    visit(closed)
    return counts


def _graph_constants_bytes(closed):
    """Counts captured constants, including those in nested jaxprs."""
    total = 0
    seen = set()

    def visit(value):
        nonlocal total
        jaxpr = _as_jaxpr(value)
        if id(jaxpr) in seen:
            return
        seen.add(id(jaxpr))
        total += sum(
            np.asarray(
                jax.random.key_data(const) if _is_typed_key_dtype(
                    getattr(const, 'dtype', None)) else const).nbytes
            for const in getattr(value, 'consts', ()))
        for eqn in jaxpr.eqns:
            for param in eqn.params.values():
                if isinstance(param, (tuple, list)):
                    for item in param:
                        if hasattr(item, 'jaxpr') or hasattr(item, 'eqns'):
                            visit(item)
                elif hasattr(param, 'jaxpr') or hasattr(param, 'eqns'):
                    visit(param)

    visit(closed)
    return total


def _output_bytes(out_shape):
    leaves = jax.tree_util.tree_leaves(out_shape)
    total = 0
    for leaf in leaves:
        shape = tuple(leaf.shape)
        dtype = leaf.dtype
        if _is_typed_key_dtype(dtype):
            total += int(np.prod(shape, dtype=np.int64)) * 8
        else:
            total += int(np.prod(shape,
                                 dtype=np.int64)) * np.dtype(dtype).itemsize
    return total


_MLX_DTYPE_NAMES = {
    np.dtype(np.float16): 'float16',
    np.dtype(np.float32): 'float32',
    np.dtype(np.complex64): 'complex64',
    np.dtype(np.int8): 'int8',
    np.dtype(np.uint8): 'uint8',
    np.dtype(np.int16): 'int16',
    np.dtype(np.uint16): 'uint16',
    np.dtype(np.int32): 'int32',
    np.dtype(np.uint32): 'uint32',
    np.dtype(np.bool_): 'bool_',
}
_MLX_TO_NUMPY_DTYPE = {
    getattr(mx, mlx_name): numpy_dtype
    for numpy_dtype, mlx_name in _MLX_DTYPE_NAMES.items()
}


def _numpy_dtype(dtype):
    try:
        return _MLX_TO_NUMPY_DTYPE[dtype]
    except (KeyError, TypeError):
        return dtype


def _jax_dtype(dtype):
    dtype = _numpy_dtype(dtype)
    if _is_typed_key_dtype(dtype):
        return dtype
    return jax.dtypes.canonicalize_dtype(dtype)


def _mlx_dtype(dtype):
    dtype = np.dtype(_jax_dtype(dtype))
    if dtype not in _MLX_DTYPE_NAMES:
        raise TypeError(f"MLXFunction does not support dtype {dtype}")
    return getattr(mx, _MLX_DTYPE_NAMES[dtype])


def _is_typed_key_dtype(dtype):
    dtype = _numpy_dtype(dtype)
    return dtype is not None and jax.dtypes.issubdtype(dtype,
                                                       jax.dtypes.prng_key)


def _is_typed_key_aval(aval):
    return _is_typed_key_dtype(aval.dtype)


def _physical_shape(aval):
    shape = tuple(aval.shape)
    return shape + (2,) if _is_typed_key_aval(aval) else shape


def _convert(value, expected_dtype=None):
    """Transfers host values to MLX while preserving existing MLX arrays."""
    if isinstance(value, mx.array):
        if expected_dtype is not None and not _is_typed_key_dtype(
                expected_dtype):
            dtype = _mlx_dtype(expected_dtype)
            if value.dtype != dtype:
                return value.astype(dtype)
        return value
    if _is_typed_key_dtype(getattr(value, 'dtype', None)):
        value = jax.random.key_data(value)
    source_dtype = getattr(value, 'dtype', None)
    if _is_typed_key_dtype(expected_dtype):
        expected_dtype = np.uint32
    dtype = _jax_dtype(expected_dtype if expected_dtype is not None else (
        source_dtype if source_dtype is not None else np.asarray(value).dtype))
    _mlx_dtype(dtype)
    return mx.array(np.asarray(value, dtype=dtype))


def _as_jaxpr(value):
    return value.jaxpr if hasattr(value, 'jaxpr') else value


def _var_value(var, env):
    if isinstance(var, jax_core.Literal):
        return _convert(var.val, var.aval.dtype)
    return env[var]


def _put(var, value, env):
    if not isinstance(var, jax_core.DropVar):
        env[var] = value


def _axes(axes):
    return tuple(int(axis) for axis in axes)


def _dot_general(lhs, rhs, params):
    (lhs_contract, rhs_contract), (lhs_batch,
                                   rhs_batch) = params['dimension_numbers']
    lhs_contract, rhs_contract = tuple(lhs_contract), tuple(rhs_contract)
    lhs_batch, rhs_batch = tuple(lhs_batch), tuple(rhs_batch)
    lhs_free = tuple(
        i for i in range(lhs.ndim) if i not in lhs_contract + lhs_batch)
    rhs_free = tuple(
        i for i in range(rhs.ndim) if i not in rhs_contract + rhs_batch)
    lhs_perm = lhs_batch + lhs_free + lhs_contract
    rhs_perm = rhs_batch + rhs_contract + rhs_free
    batch_shape = tuple(lhs.shape[i] for i in lhs_batch)
    lhs_free_shape = tuple(lhs.shape[i] for i in lhs_free)
    rhs_free_shape = tuple(rhs.shape[i] for i in rhs_free)
    contract_size = int(np.prod([lhs.shape[i] for i in lhs_contract]))
    lhs_matrix = mx.reshape(
        mx.transpose(lhs, lhs_perm),
        batch_shape + (int(np.prod(lhs_free_shape)), contract_size))
    rhs_matrix = mx.reshape(
        mx.transpose(rhs, rhs_perm),
        batch_shape + (contract_size, int(np.prod(rhs_free_shape))))
    result = mx.matmul(lhs_matrix, rhs_matrix)
    return mx.reshape(result, batch_shape + lhs_free_shape + rhs_free_shape)


def _broadcast_shape(input_shape, output_shape, dimensions):
    result = [1] * len(output_shape)
    for dim, size in zip(dimensions, input_shape):
        result[int(dim)] = size
    return tuple(result)


def _primitive(name, xs, p, in_avals=(), out_avals=()):
    """Executes one elementwise or array primitive."""
    if name == 'iota':
        shape = tuple(p['shape'])
        axis = int(p['dimension'])
        line = mx.arange(shape[axis], dtype=_mlx_dtype(p['dtype']))
        shaped = [1] * len(shape)
        shaped[axis] = shape[axis]
        return mx.broadcast_to(mx.reshape(line, shaped), shape)
    x = xs[0]
    unary = {
        'abs': mx.abs,
        'acos': mx.arccos,
        'asin': mx.arcsin,
        'atan': mx.arctan,
        'cos': mx.cos,
        'sin': mx.sin,
        'tan': mx.tan,
        'tanh': mx.tanh,
        'exp': mx.exp,
        'expm1': mx.expm1,
        'log': mx.log,
        'log1p': mx.log1p,
        'sqrt': mx.sqrt,
        'rsqrt': lambda a: 1 / mx.sqrt(a),
        'neg': mx.negative,
        'sign': mx.sign,
        'floor': mx.floor,
        'ceil': mx.ceil,
        'erf': mx.erf,
        'stop_gradient': lambda a: a,
    }
    if name in unary:
        return unary[name](x)
    if name in ('div', 'rem', 'lt_to', 'le_to', 'nextafter', 'erf_inv', 'erfc'):
        return evaluate_math(name, xs, p)
    binary = {
        'add': operator.add,
        'sub': operator.sub,
        'mul': operator.mul,
        'pow': operator.pow,
        'max': mx.maximum,
        'min': mx.minimum,
        'eq': operator.eq,
        'ne': operator.ne,
        'lt': operator.lt,
        'le': operator.le,
        'gt': operator.gt,
        'ge': operator.ge,
        'and': operator.and_,
        'or': operator.or_,
        'xor': operator.xor,
        'shift_left': operator.lshift,
        'shift_right_arithmetic': operator.rshift,
        'shift_right_logical': lambda a, b: mx.right_shift(a, b),
    }
    if name in binary:
        return binary[name](x, xs[1])
    if name == 'integer_pow':
        return x**p['y']
    if name == 'select_n':
        if (len(xs) == 3 and in_avals and
                np.dtype(_numpy_dtype(in_avals[0].dtype)) == np.dtype(bool)):
            return mx.where(x, xs[2], xs[1])
        selector = mx.clip(x, 0, len(xs) - 2)
        result = xs[1]
        for case_index, value in enumerate(xs[2:], start=1):
            result = mx.where(selector == case_index, value, result)
        return result
    if name == 'clamp':
        return mx.minimum(mx.maximum(xs[1], xs[0]), xs[2])
    if name == 'convert_element_type':
        return x.astype(_mlx_dtype(p['new_dtype']))
    if name == 'bitcast_convert_type':
        return mx.view(x, _mlx_dtype(p['new_dtype']))
    if name == 'reshape':
        dimensions = p.get('dimensions')
        if dimensions is not None:
            axes = tuple(dimensions)
            if in_avals and _is_typed_key_aval(in_avals[0]):
                axes += (x.ndim - 1,)
            x = mx.transpose(x, axes)
        shape = tuple(p['new_sizes'])
        if in_avals and _is_typed_key_aval(in_avals[0]):
            shape += (2,)
        return mx.reshape(x, shape)
    if name == 'transpose':
        axes = tuple(p['permutation'])
        if in_avals and _is_typed_key_aval(in_avals[0]):
            axes += (x.ndim - 1,)
        return mx.transpose(x, axes)
    if name == 'squeeze':
        return mx.squeeze(x, axis=_axes(p['dimensions']))
    if name == 'broadcast_in_dim':
        shape = tuple(p['shape'])
        dimensions = tuple(p['broadcast_dimensions'])
        if out_avals and _is_typed_key_aval(out_avals[0]):
            shape += (2,)
            dimensions += (len(shape) - 1,)
        return mx.broadcast_to(
            mx.reshape(x, _broadcast_shape(x.shape, shape, dimensions)), shape)
    if name == 'concatenate':
        return mx.concatenate(xs, axis=int(p['dimension']))
    if name == 'stack':
        return mx.stack(xs, axis=int(p['axis']))
    if name == 'unstack':
        axis = int(p['axis'])
        return tuple(
            mx.squeeze(v, axis=axis)
            for v in mx.split(x, x.shape[axis], axis=axis))
    if name == 'split':
        axis = int(p['axis'])
        cuts = np.cumsum([int(v) for v in p['sizes'][:-1]]).tolist()
        return tuple(mx.split(x, cuts, axis=axis))
    if name == 'slice':
        starts, limits = p['start_indices'], p['limit_indices']
        strides = p.get('strides') or (1,) * len(starts)
        slices = [
            slice(int(a), int(b), int(c))
            for a, b, c in zip(starts, limits, strides)
        ]
        if in_avals and _is_typed_key_aval(in_avals[0]):
            slices.append(slice(None))
        return x[tuple(slices)]
    if name == 'dynamic_slice':
        sizes = p['slice_sizes']
        starts = [
            mx.maximum(0, mx.minimum(v, n - s))
            for v, n, s in zip(xs[1:], x.shape, sizes)
        ]
        # Dynamic Python indexing is not available in MLX; gather along each axis.
        result = x
        for axis, (start, size) in enumerate(zip(starts, sizes)):
            idx = mx.arange(size) + start
            result = mx.take(result, idx, axis=axis)
        return result
    if name == 'reduce_sum':
        return mx.sum(x, axis=_axes(p['axes']))
    if name == 'reduce_prod':
        return mx.prod(x, axis=_axes(p['axes']))
    if name == 'reduce_max':
        return mx.max(x, axis=_axes(p['axes']))
    if name == 'reduce_min':
        return mx.min(x, axis=_axes(p['axes']))
    if name == 'round':
        return mx.round(x)
    if name == 'pad':
        result = x
        fill = xs[1]
        for axis, (lo, hi, interior) in enumerate(p['padding_config']):
            interior = int(interior)
            if interior and result.shape[axis]:
                pieces = []
                for i in range(result.shape[axis]):
                    index = [slice(None)] * result.ndim
                    index[axis] = slice(i, i + 1)
                    pieces.append(result[tuple(index)])
                    if i + 1 < result.shape[axis]:
                        gap_shape = list(result.shape)
                        gap_shape[axis] = interior
                        pieces.append(mx.broadcast_to(fill, gap_shape))
                result = mx.concatenate(pieces, axis=axis)
            if lo or hi:
                pads = [(0, 0)] * result.ndim
                pads[axis] = (int(lo), int(hi))
                result = mx.pad(result, pads, constant_values=fill)
        return result
    if name == 'dot_general':
        return _dot_general(x, xs[1], p)
    if name == 'conv_general_dilated':
        return _conv(x, xs[1], p)
    raise NotImplementedError(
        f"MLXFunction does not support JAX primitive '{name}'")


def _conv(lhs, rhs, p):
    """Executes JAX's 1-D convolution through MLX's NWC/O-W-I interface."""
    lhs_dilation = tuple(int(v) for v in p.get('lhs_dilation', (1,)))
    if any(dilation != 1 for dilation in lhs_dilation):
        raise NotImplementedError(
            f"MLX 1-D convolution does not support lhs_dilation={lhs_dilation}")
    dims = p['dimension_numbers']
    lhs_spec, rhs_spec, out_spec = dims.lhs_spec, dims.rhs_spec, dims.out_spec
    lhs_nwc = mx.transpose(lhs, (lhs_spec[0], lhs_spec[2], lhs_spec[1]))
    # JAX kernels are O-I-W; MLX conv1d kernels are O-W-I.
    rhs_owi = mx.transpose(rhs, (rhs_spec[0], rhs_spec[2], rhs_spec[1]))
    low, high = map(int, p['padding'][0])
    if low or high:
        lhs_nwc = mx.pad(lhs_nwc, [(0, 0), (low, high), (0, 0)])
    result = mx.conv1d(lhs_nwc,
                       rhs_owi,
                       stride=int(p['window_strides'][0]),
                       padding=0,
                       dilation=int(p.get('rhs_dilation', (1,))[0]),
                       groups=int(p.get('feature_group_count', 1)))
    return mx.transpose(result, tuple(out_spec.index(i) for i in (0, 2, 1)))


def _eval_jaxpr(closed, args, partitionable=True, constants_cache=None):
    jaxpr = _as_jaxpr(closed)
    env = {}
    consts = getattr(closed, 'consts', ())
    if constants_cache is None:
        constants_cache = {}
    const_key = id(closed)
    if const_key not in constants_cache:
        constants_cache[const_key] = tuple(
            _convert(value, var.aval.dtype)
            for var, value in zip(jaxpr.constvars, consts))
    for var, value in zip(jaxpr.constvars, constants_cache[const_key]):
        env[var] = value
    for var, value in zip(jaxpr.invars, args):
        env[var] = value
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        vals = [_var_value(v, env) for v in eqn.invars]
        p = eqn.params
        try:
            out = _call_primitive(name, vals, p, [v.aval for v in eqn.invars],
                                  [v.aval for v in eqn.outvars], partitionable,
                                  constants_cache)
        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"MLX execution failed for JAX primitive '{name}': {e}") from e
        outs = out if isinstance(out, tuple) and len(out) == len(
            eqn.outvars) else (out,)
        if len(outs) != len(eqn.outvars):
            raise RuntimeError(
                f"Primitive '{name}' returned {len(outs)} results; expected {len(eqn.outvars)}"
            )
        for var, value in zip(eqn.outvars, outs):
            _put(var, value, env)
    return tuple(_var_value(v, env) for v in jaxpr.outvars)


def _call_primitive(name,
                    xs,
                    p,
                    in_avals=(),
                    out_avals=(),
                    partitionable=True,
                    constants_cache=None):
    if name in ('jit', 'pjit', 'xla_call', 'remat2', 'custom_jvp_call',
                'custom_vjp_call'):
        inner = p.get('jaxpr', p.get('call_jaxpr'))
        if inner is None:
            raise NotImplementedError(
                f"MLXFunction cannot execute {name} without a nested jaxpr")
        return _eval_jaxpr(inner, xs, partitionable, constants_cache)
    if name == 'cond':
        idx, *operands = xs
        branch_results = [
            _eval_jaxpr(branch, operands, partitionable, constants_cache)
            for branch in p['branches']
        ]
        selector = mx.asarray(idx)
        selected = branch_results[0]
        for branch_index, result in enumerate(branch_results[1:], start=1):
            selected = tuple(
                mx.where(selector == branch_index, new, old)
                for old, new in zip(selected, result))
        return selected
    if name == 'scan':
        return _scan(xs, p, in_avals, out_avals, partitionable, constants_cache)
    if name in ('while', 'while_loop'):
        return _while(xs, p, partitionable, constants_cache)
    if name == 'platform_index':
        return mx.array(1, dtype=mx.int32)
    if name in ('gather', 'scatter', 'dynamic_slice', 'dynamic_update_slice'):
        return tuple(evaluate_indexing(name, xs, p, out_avals))
    if name in ('random_wrap', 'random_unwrap', 'random_bits', 'random_split',
                'random_fold_in', 'threefry2x32'):
        random_params = dict(p)
        random_params['_threefry_partitionable'] = partitionable
        return tuple(evaluate_random(name, xs, random_params, out_avals))
    return _primitive(name, xs, p, in_avals, out_avals)


def _scan(xs, p, in_avals, out_avals, partitionable, constants_cache):
    """Runs JAX scan with its const/carry/scanned input partition."""
    inner_graph = p.get('jaxpr', p.get('call_jaxpr'))
    if inner_graph is None:
        raise NotImplementedError(
            "MLXFunction cannot execute scan without its body jaxpr")
    length = int(p['length'])
    parts = p['ft_in'].elts

    def leaf_count(tree):
        children = getattr(tree, 'elts', None)
        return sum(leaf_count(child)
                   for child in children) if children is not None else 1

    if len(parts) == 3:
        num_consts, num_carry, num_xs = map(leaf_count, parts)
    elif len(parts) == 2:
        num_consts = 0
        num_carry, num_xs = map(leaf_count, parts)
    else:
        raise NotImplementedError(
            f"Unsupported scan input tree with {len(parts)} partitions")
    consts = list(xs[:num_consts])
    carry = list(xs[num_consts:num_consts + num_carry])
    seqs = list(xs[num_consts + num_carry:])
    outputs = []
    reverse = bool(p.get('reverse', False))
    for step_index in range(length):
        index = length - step_index - 1 if reverse else step_index
        sliced = [value[index] for value in seqs]
        result = _eval_jaxpr(inner_graph, consts + carry + sliced,
                             partitionable, constants_cache)
        carry = list(result[:num_carry])
        outputs.append(result[num_carry:])
    if outputs:
        output_sequences = []
        for output_index in range(len(outputs[0])):
            values = [step[output_index] for step in outputs]
            if reverse:
                values.reverse()
            output_sequences.append(mx.stack(values, axis=0))
        ys = tuple(output_sequences)
    else:
        empty_outputs = []
        for aval in out_avals[num_carry:]:
            shape = _physical_shape(aval)
            if _is_typed_key_aval(aval):
                impl = getattr(aval.dtype, '_impl', None)
                impl_name = getattr(impl, 'name', None)
                if impl_name != 'threefry2x32':
                    raise NotImplementedError(
                        f"MLX empty scan output does not support PRNG implementation "
                        f"{impl_name!r}")
                dtype = mx.uint32
            else:
                dtype = _mlx_dtype(aval.dtype)
            empty_outputs.append(mx.zeros(shape, dtype=dtype))
        ys = tuple(empty_outputs)
    return tuple(carry) + ys


def _while(xs, p, partitionable, constants_cache):
    """Executes scalar-controlled loops with MLX-valued loop state."""
    cond = p['cond_jaxpr']
    body = p['body_jaxpr']
    num_cond_consts = int(p.get('cond_nconsts', 0))
    num_body_consts = int(p.get('body_nconsts', 0))
    cond_consts = tuple(xs[:num_cond_consts])
    body_consts = tuple(xs[num_cond_consts:num_cond_consts + num_body_consts])
    carry = tuple(xs[num_cond_consts + num_body_consts:])
    while True:
        predicate = _eval_jaxpr(cond, cond_consts + carry, partitionable,
                                constants_cache)[0]
        if not bool(np.asarray(predicate)):
            return carry
        carry = _eval_jaxpr(body, body_consts + carry, partitionable,
                            constants_cache)


class MLXFunction:
    """Caches and executes one Python callable as an MLX-evaluated JAXPR."""

    def __init__(self, function):
        self.function = function
        self._cache = {}
        self.last_profile = None

    def __call__(self, *args, **kwargs):
        self.last_profile = None
        if kwargs:
            raise TypeError('MLXFunction accepts positional arguments only')
        leaves, treedef = jax.tree_util.tree_flatten(args)
        leaf_shapes = [
            tuple(x.shape) if hasattr(x, 'shape') else tuple(np.shape(x))
            for x in leaves
        ]
        leaf_dtypes = [
            x.dtype if hasattr(x, 'dtype') else np.asarray(x).dtype
            for x in leaves
        ]
        signature = (treedef, tuple(zip(leaf_shapes, map(str, leaf_dtypes))))
        graph = self._cache.get(signature)
        if graph is None:
            traced = True
            trace_start = time.perf_counter()
            abstract = [
                jax.ShapeDtypeStruct(shape, _jax_dtype(dtype))
                for shape, dtype in zip(leaf_shapes, leaf_dtypes)
            ]

            def flat_function(*flat):
                return self.function(
                    *jax.tree_util.tree_unflatten(treedef, flat))

            closed = jax.make_jaxpr(flat_function)(*abstract)
            out_shape = jax.eval_shape(flat_function, *abstract)
            _, out_tree = jax.tree_util.tree_flatten(out_shape)
            partitionable = bool(
                getattr(jax.config, 'jax_threefry_partitionable', True))
            graph_stats = {
                'primitive_counts': _primitive_counts(closed),
                'constants_bytes': _graph_constants_bytes(closed),
                'output_bytes': _output_bytes(out_shape),
            }
            trace_seconds = time.perf_counter() - trace_start
            graph = (closed, out_tree, partitionable, {}, graph_stats)
            self._cache[signature] = graph
        else:
            traced = False
            trace_seconds = 0.0
        closed, out_tree, partitionable, constants_cache, graph_stats = graph
        execution_start = time.perf_counter()
        mlx_args = [
            _convert(x, var.aval.dtype)
            for x, var in zip(leaves,
                              _as_jaxpr(closed).invars)
        ]
        outputs = _eval_jaxpr(closed, mlx_args, partitionable, constants_cache)
        mx.eval(*outputs)
        execution_seconds = time.perf_counter() - execution_start
        self.last_profile = {
            'trace': traced,
            'cachehit': not traced,
            'trace_seconds': trace_seconds,
            'execution_seconds': execution_seconds,
            'cache_size': len(self._cache),
            'primitive_counts': graph_stats['primitive_counts'],
            'constants_bytes': graph_stats['constants_bytes'],
            'input_shapes': [list(shape) for shape in leaf_shapes],
            'input_dtypes': [str(dtype) for dtype in leaf_dtypes],
            'output_bytes': graph_stats['output_bytes'],
        }
        return jax.tree_util.tree_unflatten(out_tree, outputs)
