# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MLX evaluator for JAX's partitionable Threefry2x32 random primitives."""

from __future__ import annotations

import math

import mlx.core as mx

_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)


def _threefry2x32(k0, k1, x0, x1):
    """Applies JAX's Threefry2x32 hash using MLX uint32 operations only."""
    ks = (k0, k1, k0 ^ k1 ^ mx.array(0x1BD11BDA, dtype=mx.uint32))
    x0, x1 = x0 + k0, x1 + k1
    for round_index in range(20):
        rotation = _ROTATIONS[round_index % len(_ROTATIONS)]
        x0 = x0 + x1
        x1 = ((x1 << rotation) | (x1 >> (32 - rotation))) ^ x0
        if round_index % 4 == 3:
            injection = (round_index + 1) // 4
            x0 = x0 + ks[injection % 3]
            x1 = x1 + ks[(injection + 1) % 3] + mx.array(injection,
                                                         dtype=mx.uint32)
    return x0, x1


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _counter_words(shape):
    """Returns the high and low words of JAX's flattened uint64 counter."""
    size = math.prod(shape)
    if size > 2**32:
        raise NotImplementedError(
            "MLX Threefry counters larger than 2**32 elements are unsupported")
    low = mx.arange(size, dtype=mx.uint32).reshape(shape)
    return mx.zeros_like(low), low


def _key_words(key):
    if len(key.shape) < 1 or key.shape[-1] != 2:
        raise ValueError(
            f"Threefry keys must have trailing uint32 dimension 2; got {key.shape}"
        )
    return key[..., 0], key[..., 1]


def _legacy_words(key, count):
    """Returns JAX's legacy-semantics Threefry count sequence."""
    if count > 2**32:
        raise NotImplementedError(
            "MLX Threefry counters larger than 2**32 elements are unsupported")
    key_prefix = tuple(key.shape[:-1])
    padded_count = count + count % 2
    counts = mx.arange(count, dtype=mx.uint32)
    counts = counts.reshape((1,) * len(key_prefix) + (count,))
    if padded_count != count:
        zero = mx.zeros((1,) * len(key_prefix) + (1,), dtype=mx.uint32)
        counts = mx.concatenate([counts, zero], axis=-1)
    half = padded_count // 2
    k0, k1 = _key_words(key)
    k0 = k0.reshape(key_prefix + (1,))
    k1 = k1.reshape(key_prefix + (1,))
    first, second = _threefry2x32(k0, k1, counts[..., :half], counts[...,
                                                                     half:])
    return mx.concatenate([first, second], axis=-1)[..., :count]


def _split(key, shape, partitionable):
    shape = _shape_tuple(shape)
    count = math.prod(shape)
    key_prefix = tuple(key.shape[:-1])
    if not partitionable:
        return _legacy_words(key, count * 2).reshape(key_prefix + shape + (2,))
    k0, k1 = _key_words(key)
    expand_dims = key_prefix + (1,) * len(shape)
    k0 = k0.reshape(expand_dims)
    k1 = k1.reshape(expand_dims)
    counts_hi, counts_lo = _counter_words(shape)
    first, second = _threefry2x32(k0, k1, counts_hi, counts_lo)
    result_shape = key_prefix + shape + (2,)
    return mx.stack([first, second], axis=-1).reshape(result_shape)


def _fold_in(key, datum):
    k0, k1 = _key_words(key)
    datum = mx.asarray(datum, dtype=mx.uint32)
    zero = mx.zeros_like(datum)
    return mx.stack(list(_threefry2x32(k0, k1, zero, datum)), axis=-1)


def _random_bits(key, bit_width, shape, partitionable):
    shape = _shape_tuple(shape)
    if bit_width not in (8, 16, 32, 64):
        raise TypeError(
            "Threefry random bits requires 8-, 16-, 32- or 64-bit width")
    if bit_width == 64:
        raise NotImplementedError(
            "MLX random_bits does not support JAX uint64 output precision")
    size = math.prod(shape)
    key_prefix = tuple(key.shape[:-1])
    if partitionable:
        k0, k1 = _key_words(key)
        expand_dims = key_prefix + (1,) * len(shape)
        k0 = k0.reshape(expand_dims)
        k1 = k1.reshape(expand_dims)
        counts_hi, counts_lo = _counter_words(shape)
        first, second = _threefry2x32(k0, k1, counts_hi, counts_lo)
        bits = first ^ second
    else:
        count = (bit_width * size + 31) // 32
        bits = _legacy_words(key, count)
        if bit_width != 32:
            shifts = mx.arange(32 // bit_width, dtype=mx.uint32) * bit_width
            bits = bits[..., :, None] >> shifts
            bits = bits & ((1 << bit_width) - 1)
            bits = bits.reshape(key_prefix + (count * (32 // bit_width),))
            bits = bits[..., :size]
    output_shape = key_prefix + shape
    if bit_width == 8:
        return bits.astype(mx.uint8).reshape(output_shape)
    if bit_width == 16:
        return bits.astype(mx.uint16).reshape(output_shape)
    return bits.reshape(output_shape)


def evaluate_random(name, args, params, out_avals):
    """Evaluates a supported JAX random primitive, returning MLX results."""
    if name in ("random_wrap", "random_unwrap"):
        return list(args)
    partitionable = params.get("_threefry_partitionable", True)
    if name == "random_split":
        return [_split(args[0], params["shape"], partitionable)]
    if name == "random_fold_in":
        return [_fold_in(args[0], args[1])]
    if name == "random_bits":
        return [
            _random_bits(args[0], int(params["bit_width"]), params["shape"],
                         partitionable)
        ]
    if name in ("threefry2x32", "threefry2x32_p"):
        if len(args) != 4:
            raise ValueError(
                f"threefry2x32 expects four arguments, got {len(args)}")
        return list(_threefry2x32(*args))
    raise NotImplementedError(f"MLX random evaluator does not support '{name}'")
