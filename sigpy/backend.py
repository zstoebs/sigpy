# -*- coding: utf-8 -*-
"""Functions and classes for getting and setting computing devices.

"""
from typing import Literal
import warnings
import numpy as np
from abc import ABC, abstractmethod

from sigpy import config, pytorch

if config.cupy_enabled:
	import cupy as cp

if config.mpi4py_enabled:
	from mpi4py import MPI

	if config.nccl_enabled:
		from cupy.cuda import nccl

if config.pytorch_enabled:
	import torch as pt

config.set_backend_preference("auto")

__all__ = [
	"Device",
	"CPUDevice",
	"CupyDevice",
	"TorchDevice",
	"cpu_device",
	"get_device",
	"get_array_module",
	"is_arraylike",
	"to_device",
	"copyto",
	"Communicator",
]


class Device(ABC):
	"""Abstract base class factory for compute devices.

	The array module for the corresponding device can be obtained via .xp.
	The Device object can be used as a context:

		>>> device = Device(2)
		>>> xp = device.xp  # xp is cupy.
		>>> with device:
		>>>     x = xp.array([1, 2, 3])
		>>>     x += 1

	Args:
		spec (int, str, Device, torch.Device, or cupy.cuda.Device): id > 0 represents
			the corresponding GPUs, and id = -1 represents CPU.

	Attributes:
		id (int): id = -1 represents CPU,
			and others represents the id_th GPUs.

	"""

	def __new__(cls, spec=None):
		"""Factory method that returns appropriate Device subclass.
		"""
		# If called on a subclass, use normal instantiation
		if cls is not Device:
			return super().__new__(cls)
		
		# Factory logic - determine which subclass to create
		if spec is None:
			return CPUDevice()
		
		# Handle Device instances (return as-is)
		if isinstance(spec, Device):
			return spec
		
		# Handle integers
		if isinstance(spec, int):
			if spec == -1:
				return CPUDevice()
			# Prefer CuPy for GPU if both backends available
			elif config.cupy_enabled:
				return CupyDevice.__new__(CupyDevice)
			elif config.pytorch_enabled:
				return TorchDevice.__new__(TorchDevice)
			else:
				raise ValueError(f"No GPU backend available for device {spec}")
		
		# Handle strings
		if isinstance(spec, str):
			spec = spec.lower()
			if spec == 'cpu':
				return CPUDevice()
			elif spec.startswith('cuda:'):
				return TorchDevice.__new__(TorchDevice)
			elif spec.startswith('cupy:'):
				return CupyDevice.__new__(CupyDevice)
			else:
				raise ValueError(f"Invalid device string: {spec}")
		
		# Handle native device objects
		if config.cupy_enabled and isinstance(spec, cp.cuda.Device):
			return CupyDevice.__new__(CupyDevice)
		
		if config.pytorch_enabled and isinstance(spec, pt.device):
			return TorchDevice.__new__(TorchDevice)
		
		raise ValueError(f"Cannot create Device from: {type(spec)}")

	def __init__(self, spec=None):
		if hasattr(self, 'id'):
			return
		
		# set id attr
		if isinstance(spec, int):
			id = spec
		elif isinstance(spec, str):
			if spec == "cpu":
				id = -1
			elif spec.startswith("cupy:"):
				if config.cupy_enabled:
					id = int(spec.split(":")[1])
				else:
					raise ValueError("cupy is not installed, but set device cupy.")
			elif spec.startswith("cuda:"):
				if config.pytorch_enabled:
					id = int(spec.split(":")[1])
				else:
					raise ValueError("torch is not installed, but set device cuda.")
			else:
				raise ValueError(
					f"Unsupported device string {spec}. Accepts 'cpu', 'cupy:<id>', or 'cuda:<id>'."
				)
		elif isinstance(spec, Device):
			id = spec.id
		elif config.pytorch_enabled and isinstance(spec, pt.device):
			if spec.type == "cpu":
				id = -1
			elif spec.type == "cuda":
				id = spec.index
			else:
				raise ValueError(
					"Unsupported torch device type {}, "
					"only cpu and cuda are supported.".format(spec.type)
				)
		elif config.cupy_enabled and isinstance(spec, cp.cuda.Device):
			id = spec.id
		else:
			raise ValueError(
				"Accepts int, Device, cupy.cuda.Device, or torch.device, got {}".format(
					spec
				)
			)

		self.id = id

	@property
	@abstractmethod
	def xp(self):
		"""Return the array module (numpy, cupy, or torch) for this device."""
		pass

	@property
	@abstractmethod
	def backend_name(self) -> str:
		"""Return backend name: 'numpy', 'cupy', or 'torch'."""
		pass

	@abstractmethod
	def use(self):
		"""Set this device as the current device for operations."""
		pass

	@abstractmethod
	def __enter__(self):
		"""Context manager entry."""
		pass

	@abstractmethod
	def __exit__(self, *args):
		"""Context manager exit."""
		pass

	def __int__(self):
		"""Int representation of the device, which is the id."""
		return self.id

	def __eq__(self, other):
		"""Equality operator."""
		if isinstance(other, int):
			return self.id == other
		elif isinstance(other, Device):
			return self.id == other.id and type(self) == type(other)
		return False

	def __ne__(self, other):
		"""Inequality operator."""
		return not self == other

	@abstractmethod
	def __repr__(self):
		pass


class CPUDevice(Device):
	"""CPU device 

	Uses NumPy.
	
	"""
	
	def __init__(self, spec=None):
		if not hasattr(self, 'id'):
			self.id = -1
	
	@property
	def xp(self):
		return np
	
	@property
	def backend_name(self) -> str:
		return 'numpy'
	
	def use(self):
		pass  # No-op for CPU
	
	def __enter__(self):
		return None
	
	def __exit__(self, *args):
		pass
	
	def __repr__(self):
		return "<CPU Device>"	


class CupyDevice(Device):
	"""Cupy device

	"""
	def __init__(self, spec):
		if not config.cupy_enabled:
			raise ValueError("CuPy not installed.")

		if not hasattr(self, 'id'):
			super().__init__(spec)

		if self.id < 0:
			raise ValueError(f"CupyDevice requires id >= 0, got {id}")
		
		self.device = cp.cuda.Device(self.id)
		
	@property
	def xp(self):
		return cp
	
	@property
	def backend_name(self) -> str:
		return 'cupy'
	
	def use(self):
		self.device.use()
	
	def __enter__(self):
		return self.device.__enter__()
	
	def __exit__(self, *args):
		self.device.__exit__(*args)
	
	def __repr__(self):
		return self.device.__repr__() 
	
	def __eq__(self, other):
		if isinstance(other, cp.cuda.Device):
			return self.id == other.id
		return super().__eq__(other)


class TorchDevice(Device):
	"""Torch device

	"""
	def __init__(self, spec):
		if not config.pytorch_enabled:
			raise ValueError("PyTorch not installed")
		
		# Parse spec if not already done
		if not hasattr(self, 'id'):
			super().__init__(spec)
		
		if self.id == -1:
			self.device = pt.device('cpu')
		else:
			if not pt.cuda.is_available():
				raise ValueError(f"CUDA not available for device {self.id}")
			self.device = pt.device(f'cuda:{self.id}')
		
		self._prev_device = None
		self._torch_compat = None
	
	@property
	def xp(self):
		if self._torch_compat is None:
			from sigpy.pytorch import TorchCompat
			self._torch_compat = TorchCompat(self.device)
		return self._torch_compat
	
	@property
	def backend_name(self) -> str:
		return 'torch'
	
	def use(self):
		if self.id >= 0:
			pt.cuda.set_device(self.id)
	
	def __enter__(self):
		if self.id >= 0:
			# torch dev doesn't have __enter__()
			self._prev_device = pt.cuda.current_device()
			pt.cuda.set_device(self.id)
		return self
	
	def __exit__(self, *args):
		if self.id >= 0:
			pt.cuda.set_device(self._prev_device)
	
	def __repr__(self):
		return self.device.__repr__()	
	
	def __eq__(self, other):
		if isinstance(other, pt.device):
			if other.type == 'cpu':
				return self.id == -1
			elif other.type == 'cuda':
				return self.id == other.index
		return super().__eq__(other)


cpu_device = CPUDevice()


def get_array_module(array):
	"""Gets an appropriate module from :mod:`numpy` or :mod:`cupy` or :mod:`torch`.

	This is almost equivalent to :func:`cupy.get_array_module`. The differences
	are that this function can be used even if cupy is not available.

	Args:
		array: Input array.

	Returns:
		module: :mod:`cupy`, :mod:`numpy`, or :mod:`torch` is returned based on input.
	"""
	if config.cupy_enabled and isinstance(array, cp.ndarray):
		return cp.get_array_module(array)
	elif config.pytorch_enabled and isinstance(array, pt.Tensor):
		return pt
	else:
		return np


def is_arraylike(array) -> bool:
	"""Check if input is array-like (numpy, cupy, or torch array).

	Args:
		array: Input to check.

	Returns:
		bool: True if array-like, False otherwise.
	"""
	if config.cupy_enabled and isinstance(array, cp.ndarray):
		return True
	if config.pytorch_enabled and isinstance(array, pt.Tensor):
		return True
	return isinstance(array, np.ndarray)


def get_device(array) -> Device:
	"""Get Device from input array.

	Args:
		array (array): Array.

	Returns:
		Device.

	"""
	if (config.cupy_enabled or config.pytorch_enabled) and isinstance(array, (cp.ndarray, pt.Tensor)):
		return Device(array.device)
	else:
		return cpu_device
	

def to_device(input, device=cpu_device):
	"""Move input to device. Does not copy if same device.

	Args:
		input (array): Input.
		device (int or Device or cupy.Device or torch.device): Output device.

	Returns:
		array: Output array placed in device.
	"""
	idevice = get_device(input)
	odevice = Device(device)

	if idevice == odevice:
		return input

	if odevice == cpu_device: # output is CPU
		if idevice.backend_name == 'cupy':
			with idevice:
				return input.get()
		elif idevice.backend_name == 'torch':
			return pytorch.from_pytorch(input.cpu())
		else:
			return np.asarray(input) 
	else: # devices are different and output is not CPU 
		with odevice:
			if odevice.backend_name == 'cupy':
				if idevice.backend_name == 'torch': 
					converted = pytorch.from_pytorch(input)
					if config.cupy_enabled and isinstance(converted, cp.ndarray):
						return converted
					return cp.asarray(converted) # edge case where from_pytorch returns numpy bc cupy not enabled 
				return cp.asarray(input) # input is numpy
			elif odevice.backend_name == 'torch':
				if idevice.backend_name == 'torch': 
					return input.to(odevice.device)
				elif idevice.backend_name == 'cupy':
					return pytorch.to_pytorch(input, requires_grad=False).to(odevice.device)
				return pt.as_tensor(input, device=odevice.device) # input is numpy
	
	raise ValueError(f"Unsupported device transfer from {idevice.backend_name} to {odevice.backend_name}")


def copyto(output, input):
	"""Copy from input to output. Input/output can be in different device.

	Args:
		input (array): Input.
		output (array): Output.

	"""
	odevice = get_device(output)
	converted = to_device(input, odevice)
	if odevice == cpu_device:
		np.copyto(output, converted)
	elif odevice.backend_name == 'cupy':
		with odevice:
			cp.copyto(output, converted)
	elif odevice.backend_name == 'torch':
		output.copy_(converted)
	else:
		raise ValueError(f"Unsupported output device: {odevice}")
	

class Communicator(object):
	"""Communicator for distributed computing using MPI.

	When NCCL is not installed, arrays are moved to CPU,
	then communicated through MPI, and moved back
	to original device.
	When mpi4py is not installed, the communicator errors.

	"""

	def __init__(self):
		if config.mpi4py_enabled:
			self.mpi_comm = MPI.COMM_WORLD
			self.size = self.mpi_comm.Get_size()
			self.rank = self.mpi_comm.Get_rank()
		else:
			self.size = 1
			self.rank = 0

		# Keep nccl comms for reuse
		if config.nccl_enabled:
			self.nccl_comms = {}

	def allreduce(self, input):
		"""All reduce operation in-place.

		Sums input across all nodes and broadcast back to each node.

		Args:
			input (array): input array.

		"""
		if self.size > 1:
			if config.nccl_enabled:
				device = get_device(input)
				devices = self.mpi_comm.allgather(device.id)
				if all([d >= 0 for d in devices]):
					nccl_comm = self._get_nccl_comm(device, devices)
					nccl_dtype, nccl_size = self._get_nccl_dtype_size(input)
					with device:
						nccl_comm.allReduce(
							input.data.ptr,
							input.data.ptr,
							nccl_size,
							nccl_dtype,
							nccl.NCCL_SUM,
							cp.cuda.Stream.null.ptr,
						)
						return

			cpu_input = to_device(input, cpu_device)
			self.mpi_comm.Allreduce(MPI.IN_PLACE, cpu_input)
			copyto(input, cpu_input)

	def reduce(self, input, root=0):
		"""Reduce operation in-place.

		Sums input across all nodes in root node.

		Args:
			input (array): input array.
			root (int): root node rank.

		"""
		if self.size > 1:
			cpu_input = to_device(input, cpu_device)
			if self.rank == root:
				self.mpi_comm.Reduce(MPI.IN_PLACE, cpu_input, root=root)
				copyto(input, cpu_input)
			else:
				self.mpi_comm.Reduce(cpu_input, None, root=root)

	def bcast(self, input, root=0, datatype=None):
		"""Broadcast from root to other nodes.

		Args:
			input (array): input array.
			root (int): root node rank.
			datatype (int): MPI datatype for broadcasting.

		"""
		if config.mpi4py_enabled:
			datatype = MPI.COMPLEX

		if self.size > 1:
			cpu_input = to_device(input, cpu_device)
			self.mpi_comm.Bcast((cpu_input, datatype), root=root)
			copyto(input, cpu_input)

	def gatherv(self, input, root=0):
		"""Gather with variable sizes operation.

		Gather inputs across all nodes to the root node,
		and vectorizes them.

		Args:
			input (array): input array.
			root (int): root node rank.

		Returns:
			array or None: vectorized array if rank==root else None.

		"""
		if self.size > 1:
			cpu_input = to_device(input, cpu_device)

			sizes = self.mpi_comm.gather(input.size, root=root)
			if self.rank == root:
				cpu_output = np.empty(sum(sizes), dtype=input.dtype)
				self.mpi_comm.Gatherv(
					cpu_input, [cpu_output, sizes], root=root
				)
				return to_device(cpu_output, get_device(input))
			else:
				self.mpi_comm.Gatherv(cpu_input, [None, sizes], root=root)
		else:
			return input

	def _get_nccl_comm(self, device, devices):
		if str(devices) in self.nccl_comms:
			return self.nccl_comms[str(devices)]

		if self.rank == 0:
			nccl_comm_id = nccl.get_unique_id()
		else:
			nccl_comm_id = None

		nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id)

		with device:
			nccl_comm = nccl.NcclCommunicator(
				self.size, nccl_comm_id, self.rank
			)
			self.nccl_comms[str(devices)] = nccl_comm

		return nccl_comm

	def _get_nccl_dtype_size(self, input):
		if input.dtype == np.float32:
			nccl_dtype = nccl.NCCL_FLOAT32
			nccl_size = input.size
		elif input.dtype == np.float64:
			nccl_dtype = nccl.NCCL_FLOAT64
			nccl_size = input.size
		elif input.dtype == np.complex64:
			nccl_dtype = nccl.NCCL_FLOAT32
			nccl_size = input.size * 2
		elif input.dtype == np.complex128:
			nccl_dtype = nccl.NCCL_FLOAT64
			nccl_size = input.size * 2
		else:
			raise ValueError(
				"dtype not supported, got {dtype}.".format(dtype=input.dtype)
			)

		return nccl_dtype, nccl_size
