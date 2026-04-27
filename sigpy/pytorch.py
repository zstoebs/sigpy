# -*- coding: utf-8 -*-
"""Functions for interoperability between sigpy and pytorch.

"""
import numpy as np

from sigpy import backend, config

if config.pytorch_enabled:
	import torch


__all__ = ["to_pytorch", 
		   "from_pytorch", 
		   "to_pytorch_function", 
		   "TorchCompat"]


def to_pytorch(array, requires_grad=True):  # pragma: no cover
	"""Zero-copy conversion from numpy/cupy array to pytorch tensor.

	For complex array input, returns a tensor with shape + [2],
	where tensor[..., 0] and tensor[..., 1] represent the real
	and imaginary.

	Args:
		array (numpy/cupy array): input.
		requires_grad(bool): Set .requires_grad output tensor
	Returns:
		PyTorch tensor.

	"""
	import torch
	from torch.utils.dlpack import from_dlpack

	device = backend.get_device(array)
	if not np.issubdtype(array.dtype, np.floating):
		with device:
			shape = array.shape
			array = array.view(dtype=array.real.dtype)
			array = array.reshape(shape + (2,))

	if device == backend.cpu_device:
		tensor = torch.from_numpy(array)
	else:
		tensor = from_dlpack(array.toDlpack())

	tensor.requires_grad = requires_grad
	return tensor.contiguous()


def from_pytorch(tensor, iscomplex=False):  # pragma: no cover
	"""Zero-copy conversion from pytorch tensor to numpy/cupy array.

	If iscomplex, then tensor must have the last dimension as 2,
	and the output will be viewed as a complex valued array.

	Args:
		tensor (PyTorch tensor): input.
		iscomplex (bool): whether input represents complex valued tensor.

	Returns:
		Numpy/cupy array.

	"""
	from torch.utils.dlpack import to_dlpack

	device = tensor.device
	if device.type == "cpu":
		output = tensor.detach().contiguous().numpy()
	else:
		if config.cupy_enabled:
			import cupy as cp

			output = cp.fromDlpack(to_dlpack(tensor.contiguous()))
		else:
			raise TypeError(
				"CuPy not installed, "
				"but trying to convert GPU PyTorch Tensor."
			)

	if iscomplex:
		if output.shape[-1] != 2:
			raise ValueError(
				"shape[-1] must be 2 when iscomplex is "
				"specified, but got {}".format(output.shape)
			)

		with backend.get_device(output):
			if output.dtype == np.float32:
				output = output.view(np.complex64)
			elif output.dtype == np.float64:
				output = output.view(np.complex128)

			output = output.reshape(output.shape[:-1])

	return output


def to_pytorch_function(
	linop, input_iscomplex=False, output_iscomplex=False
):  # pragma: no cover
	"""Convert SigPy Linop to PyTorch Function.

	The returned function can be treated as a native
	pytorch function performing the linop operator.
	The function can be backpropagated, applied on GPU arrays,
	and has minimal overhead as the underlying arrays
	are shared without copying.
	For complex valued input/output, the appropriate options
	should be set when calling the function.

	Args:
		linop (Linop): linear operator to be converted.
		input_iscomplex (bool): whether the PyTorch input
			represents complex tensor.
		output_iscomplex (bool): whether the PyTorch output
			represents complex tensor.

	Returns:
		torch.autograd.Function: equivalent PyTorch Function.

	"""
	import torch

	class LinopFunction(torch.autograd.Function):
		@staticmethod
		def forward(ctx, input):
			return to_pytorch(
				linop(from_pytorch(input, iscomplex=input_iscomplex))
			)

		@staticmethod
		def backward(ctx, grad_output):
			return to_pytorch(
				linop.H(from_pytorch(grad_output, iscomplex=output_iscomplex))
			)

	return LinopFunction


class TorchCompat:
	"""Wrapper to make torch module compatible with NumPy/CuPy API.

	Usage:
		device = TorchDevice(0)
		xp = device.xp  # Returns TorchCompat instance
		x = xp.array([1, 2, 3])  # Creates torch.tensor
		y = xp.fft.fft(x)  # Uses torch.fft.fft
	"""

	def __init__(self, torch_device):
		"""Initialize with a torch.device."""
		self._device = torch_device
		self._torch = torch

	# ======================
	# Array Creation
	# ======================

	def array(self, data, dtype=None):
		"""Create tensor (like np.array or cp.array)."""
		if dtype is not None:
			dtype = self._numpy_to_torch_dtype(dtype)
		tensor = torch.as_tensor(data, dtype=dtype, device=self._device)
		return tensor

	def zeros(self, shape, dtype=torch.float32):
		"""Create zeros tensor."""
		if isinstance(dtype, np.dtype):
			dtype = self._numpy_to_torch_dtype(dtype)
		return torch.zeros(shape, dtype=dtype, device=self._device)

	def ones(self, shape, dtype=torch.float32):
		"""Create ones tensor."""
		if isinstance(dtype, np.dtype):
			dtype = self._numpy_to_torch_dtype(dtype)
		return torch.ones(shape, dtype=dtype, device=self._device)

	def empty(self, shape, dtype=torch.float32):
		"""Create empty tensor."""
		if isinstance(dtype, np.dtype):
			dtype = self._numpy_to_torch_dtype(dtype)
		return torch.empty(shape, dtype=dtype, device=self._device)

	def arange(self, *args, dtype=None):
		"""Create range tensor (like np.arange)."""
		if dtype is not None and isinstance(dtype, np.dtype):
			dtype = self._numpy_to_torch_dtype(dtype)
		return torch.arange(*args, dtype=dtype, device=self._device)

	def linspace(self, start, end, steps, dtype=None, endpoint=True):
		"""Create linearly spaced tensor."""
		# PyTorch linspace doesn't have endpoint parameter
		# Adjust steps if endpoint=False
		if not endpoint:
			steps = steps + 1
		result = torch.linspace(
			start, end, steps, dtype=dtype, device=self._device
		)
		if not endpoint:
			result = result[:-1]
		return result

	# ======================
	# Random
	# ======================

	@property
	def random(self):
		"""Random number generation (NumPy-compatible API)."""
		return self._RandomCompat(self._device)

	class _RandomCompat:
		def __init__(self, device):
			self.device = device

		def normal(self, loc=0.0, scale=1.0, size=None):
			"""Generate normal random numbers (like np.random.normal)."""
			if size is None:
				return torch.randn([], device=self.device) * scale + loc
			return torch.randn(size, device=self.device) * scale + loc

		def uniform(self, low=0.0, high=1.0, size=None):
			"""Generate uniform random numbers."""
			if size is None:
				return torch.rand([], device=self.device) * (high - low) + low
			return torch.rand(size, device=self.device) * (high - low) + low

	# ======================
	# Linear Algebra
	# ======================

	@property
	def linalg(self):
		"""Linear algebra operations."""
		return self._LinalgCompat(self._device)

	class _LinalgCompat:
		def __init__(self, device):
			self.device = device

		def norm(self, x, ord=None, axis=None, keepdims=False):
			"""Compute norm (NumPy-compatible)."""
			if axis is None:
				return torch.linalg.norm(x, ord=ord)
			return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)

		def eig(self, x):
			"""Compute eigenvalues and eigenvectors."""
			return torch.linalg.eig(x)

		def eigh(self, x):
			"""Compute eigenvalues and eigenvectors of Hermitian matrix."""
			return torch.linalg.eigh(x)

	# ======================
	# FFT
	# ======================

	@property
	def fft(self):
		"""FFT operations."""
		return torch.fft

	# ======================
	# Math Operations
	# ======================

	def sum(self, x, axis=None, keepdims=False):
		"""Sum (NumPy-compatible)."""
		if axis is None:
			return torch.sum(x)
		return torch.sum(x, dim=axis, keepdim=keepdims)

	def abs(self, x):
		"""Absolute value."""
		return torch.abs(x)

	def sqrt(self, x):
		"""Square root."""
		return torch.sqrt(x)

	def exp(self, x):
		"""Exponential."""
		return torch.exp(x)

	def log(self, x):
		"""Natural logarithm."""
		return torch.log(x)

	def sin(self, x):
		return torch.sin(x)

	def cos(self, x):
		return torch.cos(x)

	def angle(self, x):
		"""Phase angle of complex tensor."""
		# PyTorch doesn't have native complex, need workaround
		if x.shape[-1] == 2:  # [real, imag] format
			return torch.atan2(x[..., 1], x[..., 0])
		return torch.angle(x)

	def conj(self, x):
		"""Complex conjugate."""
		if x.shape[-1] == 2:  # [real, imag] format
			result = x.clone()
			result[..., 1] = -result[..., 1]
			return result
		return torch.conj(x)

	def real(self, x):
		"""Real part."""
		if x.shape[-1] == 2:
			return x[..., 0]
		return torch.real(x)

	def imag(self, x):
		"""Imaginary part."""
		if x.shape[-1] == 2:
			return x[..., 1]
		return torch.imag(x)

	def vdot(self, x, y):
		"""Vector dot product (conjugate first)."""
		if x.shape[-1] == 2:  # Complex as [real, imag]
			# Need to compute x.conj() @ y
			x_conj = self.conj(x)
			return torch.sum(x_conj * y)
		return torch.vdot(x.flatten(), y.flatten())

	def expand_dims(self, x, axis):
		"""Expand dimensions."""
		return torch.unsqueeze(x, axis)

	def concatenate(self, arrays, axis=0):
		"""Concatenate arrays."""
		return torch.cat(arrays, dim=axis)

	def stack(self, arrays, axis=0):
		"""Stack arrays."""
		return torch.stack(arrays, dim=axis)

	def reshape(self, x, shape):
		"""Reshape array."""
		return x.reshape(shape)

	def transpose(self, x, axes=None):
		"""Transpose array."""
		if axes is None:
			return x.t()
		return x.permute(*axes)

	def sort(self, x, axis=-1):
		"""Sort array."""
		return torch.sort(x, dim=axis).values

	def argsort(self, x, axis=-1):
		"""Get sort indices."""
		return torch.argsort(x, dim=axis)

	def where(self, condition):
		"""Find where condition is True."""
		return torch.where(condition)

	def count_nonzero(self, x):
		"""Count non-zero elements."""
		return torch.count_nonzero(x)

	def flatnonzero(self, x):
		"""Return indices of non-zero flattened elements."""
		return torch.nonzero(x.flatten()).squeeze()

	def flip(self, x, axis=None):
		"""Flip array along axes."""
		if axis is None:
			axis = list(range(x.ndim))
		elif isinstance(axis, int):
			axis = [axis]
		return torch.flip(x, dims=axis)

	def pad(self, x, pad_width, mode="constant", constant_values=0):
		"""Pad array (limited compatibility)."""
		# PyTorch padding is different format
		# Convert numpy-style pad_width to torch format
		if mode == "constant":
			# Torch expects [left, right, top, bottom, ...]
			torch_pad = []
			for before, after in reversed(pad_width):
				torch_pad.extend([before, after])
			return torch.nn.functional.pad(x, torch_pad, value=constant_values)
		raise NotImplementedError(f"Padding mode {mode} not implemented")

	# ======================
	# Testing
	# ======================

	@property
	def testing(self):
		"""Testing utilities."""
		return self._TestingCompat()

	class _TestingCompat:
		@staticmethod
		def assert_allclose(actual, desired, rtol=1e-7, atol=0):
			"""Assert arrays are close."""
			torch.testing.assert_close(actual, desired, rtol=rtol, atol=atol)

		@staticmethod
		def assert_array_equal(x, y):
			"""Assert arrays are equal."""
			torch.testing.assert_close(x, y, rtol=0, atol=0)

	# ======================
	# Type Conversions
	# ======================

	@staticmethod
	def _numpy_to_torch_dtype(np_dtype):
		"""Convert NumPy dtype to PyTorch dtype."""
		mapping = {
			np.float16: torch.float16,
			np.float32: torch.float32,
			np.float64: torch.float64,
			np.int8: torch.int8,
			np.int16: torch.int16,
			np.int32: torch.int32,
			np.int64: torch.int64,
			np.uint8: torch.uint8,
			np.bool_: torch.bool,
			# Complex types need special handling
			np.complex64: torch.float32,  # Will be [N, 2]
			np.complex128: torch.float64,  # Will be [N, 2]
		}
		if isinstance(np_dtype, np.dtype):
			np_dtype = np_dtype.type
		return mapping.get(np_dtype, torch.float32)

	# ======================
	# Special Methods
	# ======================

	def asarray(self, data, dtype=None):
		"""Convert to tensor (like np.asarray)."""
		return self.array(data, dtype=dtype)

	def copyto(self, dst, src):
		"""Copy src to dst in-place."""
		dst.copy_(src)

	# Expose common torch functions directly
	def __getattr__(self, name):
		"""Fallback to torch for missing methods."""
		return getattr(torch, name)
	
	def complex_to_real(self, z):
		"""Convert complex array to [real, imag] stacked representation.
		
		Args:
			z: Complex-valued array of shape [..., ]
		
		Returns:
			Real-valued array of shape [..., 2]
		"""
		if not torch.is_complex(z):
			return z
		
		real = torch.real(z).unsqueeze(-1)
		imag = torch.imag(z).unsqueeze(-1)
		return torch.cat([real, imag], dim=-1)

	def real_to_complex(self, x):
		"""Convert [real, imag] representation to complex.
		
		Args:
			x: Real array of shape [..., 2]
		
		Returns:
			Complex array of shape [...]
		"""
		if x.shape[-1] != 2:
			raise ValueError("Last dimension must be 2 for complex conversion")
		
		return torch.complex(x[..., 0], x[..., 1])
