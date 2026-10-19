# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""State for pedestal models."""

import dataclasses
from typing import Self
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StructureState:
  """State for a pedestal structure model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationState:
  """State for a pedestal formation model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalState:
  """State for a pedestal model carried over from the previous timestep."""

  confinement_mode: array_typing.IntScalar = dataclasses.field(
      default_factory=lambda: jnp.asarray(
          pedestal_runtime_params_lib.ConfinementMode.L_MODE,
          dtype=jax_utils.get_int_dtype(),
      )
  )
  transition_start_time: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(-jnp.inf, dtype=jax_utils.get_dtype())
  )
  T_i_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  T_e_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  n_e_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  structure: StructureState = dataclasses.field(default_factory=StructureState)
  formation: FormationState = dataclasses.field(default_factory=FormationState)

  @classmethod
  def empty(
      cls,
      confinement_mode: array_typing.IntScalar = (
          pedestal_runtime_params_lib.ConfinementMode.L_MODE
      ),
      structure: StructureState | None = None,
      formation: FormationState | None = None,
  ) -> Self:
    """Returns an empty PedestalState."""
    return cls(
        confinement_mode=jnp.asarray(
            confinement_mode, dtype=jax_utils.get_int_dtype()
        ),
        transition_start_time=jnp.array(-jnp.inf, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        structure=structure if structure is not None else StructureState(),
        formation=formation if formation is not None else FormationState(),
    )
