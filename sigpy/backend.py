# -*- coding: utf-8 -*-
"""Functions and classes for getting and setting computing devices.

"""
from typing import Literal
import warnings
import numpy as np

from sigpy import config

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
	"get_device",
	"get_array_module",
	"cpu_device",
	"to_device",
	"copyto",
	"Communicator",
]


class Device(object):
	"""Device class.

	This class extends cupy.Device, with id > 0 representing the id_th GPU,
	and id = -1 representing CPU. cupy or torch must be installed to use GPUs.

	The array module for the corresponding device can be obtained via .xp.
	Similar to cupy.Device, the Device object can be used as a context:

		>>> device = Device(2)
		>>> xp = device.xp  # xp is cupy.
		>>> with device:
		>>>     x = xp.array([1, 2, 3])
		>>>     x += 1

	Args:
		id_or_device (int or str or Device or torch.Device or cupy.cuda.Device): id > 0 represents
			the corresponding GPUs, and id = -1 represents CPU.

	Attributes:
		id (int): id = -1 represents CPU,
			and others represents the id_th GPUs.

	"""

	def __init__(self, id_or_device):
		self.backend = config.preferred_backend
		if isinstance(id_or_device, int):
			id = id_or_device
		elif isinstance(id_or_device, str):
			if id_or_device == "cpu":
				id = -1
			elif id_or_device.startswith("cupy:"):
				if config.cupy_enabled:
					self.backend = "cupy"
					id = int(id_or_device.split(":")[1])
				else:
					raise ValueError("cupy is not installed, but set device cupy.")
			elif id_or_device.startswith("cuda:"):
				if config.pytorch_enabled:
					self.backend = "torch"
					id = int(id_or_device.split(":")[1])
				else:
					raise ValueError("torch is not installed, but set device cuda.")
			else:
				raise ValueError(
					f"Unsupported device string {id_or_device}. Accepts 'cpu', 'cupy:<id>', or 'cuda:<id>'."
				)
		elif isinstance(id_or_device, Device):
			self.backend = id_or_device.backend
			id = id_or_device.id
		elif config.pytorch_enabled and isinstance(id_or_device, pt.device):
			self.backend = "torch"
			if id_or_device.type == "cpu":
				id = -1
			elif id_or_device.type == "cuda":
				id = id_or_device.index
			else:
				raise ValueError(
					"Unsupported torch device type {}, "
					"only cpu and cuda are supported.".format(id_or_device.type)
				)
		elif config.cupy_enabled and isinstance(id_or_device, cp.cuda.Device):
			self.backend = "cupy"
			id = id_or_device.id
		else:
			raise ValueError(
				"Accepts int, Device, cupy.cuda.Device, or torch.device, got {}".format(
					id_or_device
				)
			)

		if id != -1:
			if self.backend == "torch": 
				self.device = pt.device(f"cuda:{id}")
			elif self.backend == "cupy":
				self.device = cp.cuda.Device(id)
			else:
				raise ValueError(
				f"Backend '{self.backend}' does not support GPU devices."
				)
		elif id == -1 and self.backend == "torch":
			self.device = pt.device("cpu")
		else: 
			self.device = None

		self.id = id

	@property
	def xp(self):
		"""module: numpy, cupy, or torch module for the device."""
		if isinstance(self.device, pt.device):
			return pt
		elif isinstance(self.device, cp.cuda.Device):
			return cp
		else:
			return np

	def use(self):
		"""Use computing device.

		All operations after use() will use the device.
		"""
		if self.id > 0:
			if self.backend == "cupy":
				self.device.use()
			elif self.backend == "torch":
				pt.cuda.set_device(self.device)

	def __int__(self):
		return self.id

	def __eq__(self, other):
		if isinstance(other, int):
			return self.id == other
		elif isinstance(other, str):
			if other == "cpu":
				return self.id == -1
			elif other.startswith("cupy:"):
				return self.backend == "cupy" and self.id == int(other.split(":")[1])
			elif other.startswith("cuda:"):
				return self.backend == "torch" and self.id == int(other.split(":")[1])
			else:
				return False
		elif isinstance(other, Device):
			return self.id == other.id
		elif isinstance(other, pt.device):
			return self.backend == "torch" and self.device == other
		elif isinstance(other, cp.cuda.Device):
			return self.backend == "cupy" and self.device == other
		else:
			return False

	def __ne__(self, other):
		return not self == other

	def __enter__(self):
		if self.id == -1:
			return None

		return self.device.__enter__()

	def __exit__(self, *args):
		if self.id == -1:
			pass
		else:
			self.device.__exit__()

	def __repr__(self):
		if self.id == -1:
			return "<CPU Device>"

		return self.device.__repr__()


cpu_device = Device(-1)


def get_array_module(array):
	"""Gets an appropriate module from :mod:`numpy` or :mod:`cupy` or :mod:`torch`.

	This is almost equivalent to :func:`cupy.get_array_module`. The differences
	are that this function can be used even if cupy is not available.

	Args:
		array: Input array.

	Returns:
		module: :mod:`cupy`, :mod:`numpy`, or :mod:`torch` is returned based on input.
	"""
	if config.cupy_enabled and config.preferred_backend == "cupy" and isinstance(array, cp.ndarray):
		return cp.get_array_module(array)
	elif config.pytorch_enabled and config.preferred_backend == "torch" and isinstance(array, pt.Tensor):
		return pt
	else:
		return np


def get_device(array):
	"""Get Device from input array.

	Args:
		array (array): Array.

	Returns:
		Device.

	"""
	if get_array_module(array) == np:
		return cpu_device
	else:
		return Device(array.device)


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

	if odevice == cpu_device:
		with idevice:
			return input.get()
	else:
		with odevice:
			if odevice.backend == "cupy":
				return cp.asarray(input)
			elif odevice.backend == "torch":
				return pt.as_tensor(input)


def copyto(output, input):
	"""Copy from input to output. Input/output can be in different device.

	Args:
		input (array): Input.
		output (array): Output.

	"""
	idevice = get_device(input)
	odevice = get_device(output)
	if idevice == cpu_device and odevice != cpu_device:
		with odevice:
			output.set(input)
	elif idevice != cpu_device and odevice == cpu_device:
		with idevice:
			np.copyto(output, input.get())
	else:
		idevice.xp.copyto(output, input)


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
