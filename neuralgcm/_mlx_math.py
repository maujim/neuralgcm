# Copyright 2026 Google LLC
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
"""MLX implementations of JAX primitives without direct MLX equivalents."""

import mlx.core as mx


def _integer_division(lhs, rhs, *, remainder):
    """Implements JAX integer div/rem with truncation toward zero."""
    dtype = lhs.dtype
    signed = mx.issubdtype(dtype, mx.signedinteger)
    if signed:
        unsigned_dtype = {
            mx.int8: mx.uint8,
            mx.int16: mx.uint16,
            mx.int32: mx.uint32,
            mx.int64: mx.uint64,
        }[dtype]
        lhs_negative = lhs < 0
        rhs_negative = rhs < 0
        lhs_bits = mx.view(lhs, unsigned_dtype)
        rhs_bits = mx.view(rhs, unsigned_dtype)
        lhs_magnitude = mx.where(lhs_negative,
                                 mx.subtract(mx.zeros_like(lhs_bits), lhs_bits),
                                 lhs_bits)
        rhs_magnitude = mx.where(rhs_negative,
                                 mx.subtract(mx.zeros_like(rhs_bits), rhs_bits),
                                 rhs_bits)
        # Avoid backend-specific divide-by-zero behavior; JAX's integer div/rem
        # both return zero for a zero divisor.
        divisor = mx.where(rhs_magnitude == 0,
                           mx.array(1, dtype=unsigned_dtype), rhs_magnitude)
        quotient_magnitude = mx.floor_divide(lhs_magnitude, divisor)
        quotient_negative = mx.logical_xor(lhs_negative, rhs_negative)
        quotient_bits = mx.where(
            quotient_negative,
            mx.subtract(mx.zeros_like(quotient_magnitude), quotient_magnitude),
            quotient_magnitude,
        )
        quotient = mx.view(quotient_bits, dtype)
        if remainder:
            result = mx.subtract(lhs, mx.multiply(quotient, rhs))
        else:
            result = quotient
        return mx.where(rhs == 0, mx.zeros_like(lhs), result)

    divisor = mx.where(rhs == 0, mx.array(1, dtype=dtype), rhs)
    quotient = mx.floor_divide(lhs, divisor)
    quotient = mx.where((lhs < 0) != (rhs < 0), quotient + 1, quotient)
    result = mx.subtract(lhs, mx.multiply(quotient,
                                          rhs)) if remainder else quotient
    return mx.where(rhs == 0, mx.zeros_like(lhs), result)


def _float_remainder(lhs, rhs):
    """Implements fmod using MLX remainder and the dividend's sign bit."""
    magnitude = mx.remainder(mx.abs(lhs), mx.abs(rhs))
    result_bits = mx.view(magnitude, mx.uint32)
    lhs_bits = mx.view(lhs, mx.uint32)
    sign_bit = mx.array(0x80000000, dtype=mx.uint32)
    sign = mx.bitwise_and(lhs_bits, sign_bit)
    return mx.view(mx.bitwise_or(result_bits, sign), lhs.dtype)


def _total_order_key(value):
    """Maps float32 bit patterns to monotonically ordered unsigned keys."""
    bits = mx.view(value, mx.uint32)
    sign_bit = mx.array(0x80000000, dtype=mx.uint32)
    return mx.where(
        (bits & sign_bit) != 0,
        mx.bitwise_invert(bits),
        mx.bitwise_xor(bits, sign_bit),
    )


def _compare_total(lhs, rhs, *, inclusive):
    lhs_key = _total_order_key(lhs)
    rhs_key = _total_order_key(rhs)
    if inclusive:
        return lhs_key <= rhs_key
    return lhs_key < rhs_key


def _nextafter(lhs, rhs):
    """Returns the adjacent float32 value from lhs in the direction of rhs."""
    lhs_bits = mx.view(lhs, mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    lhs_zero = lhs == 0
    lhs_negative = lhs < 0
    increase_bits = (lhs < rhs) == (~lhs_negative)
    stepped = mx.where(increase_bits, lhs_bits + one, lhs_bits - one)
    # Either signed zero steps directly to the least subnormal with rhs's sign.
    zero_step = mx.where(rhs < 0, mx.array(0x80000001, dtype=mx.uint32), one)
    result_bits = mx.where(lhs_zero, zero_step, stepped)
    result = mx.view(result_bits, mx.float32)
    result = mx.where(lhs == rhs, rhs, result)
    # The arithmetic path propagates NaNs without host-side scalar conversion.
    return mx.where(mx.isnan(lhs) | mx.isnan(rhs), lhs + rhs, result)


def evaluate_math(name, xs, params):
    """Evaluates a math primitive using MLX arrays only."""
    del params
    if name == 'div':
        lhs, rhs = xs
        if mx.issubdtype(lhs.dtype, mx.integer):
            return _integer_division(lhs, rhs, remainder=False)
        return mx.divide(lhs, rhs)
    if name == 'rem':
        lhs, rhs = xs
        if mx.issubdtype(lhs.dtype, mx.integer):
            return _integer_division(lhs, rhs, remainder=True)
        return _float_remainder(lhs, rhs)
    if name == 'lt_to':
        return _compare_total(xs[0], xs[1], inclusive=False)
    if name == 'le_to':
        return _compare_total(xs[0], xs[1], inclusive=True)
    if name == 'nextafter':
        return _nextafter(xs[0], xs[1])
    if name == 'erf_inv':
        return mx.erfinv(xs[0])
    if name == 'erfc':
        return mx.array(1, dtype=xs[0].dtype) - mx.erf(xs[0])
    raise NotImplementedError(f"Unsupported MLX math primitive '{name}'")
